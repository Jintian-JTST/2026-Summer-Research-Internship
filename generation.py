import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from constants import *

def generation(N):

    results = []
    count = 0

    while count < N:
        scale=int(min(10000,int(N-count)))#每loop生成1000个muon
        success = np.zeros(int(scale))


        T = np.random.exponential(scale=TAU_LAB,size=scale) # t_lab 是每个muon衰变的绝对时间，满足指数分布
        phase = OMEGA_A * T # 自旋进动相位


        # generation
        x= np.random.uniform(0, 1,size=scale)               # x_e 的归一化能量
        cos_theta = np.random.uniform(-1, 1,size=scale)     # cos(theta_e) 的范围是 [-1, 1]
        phi= np.random.uniform(0, 2*np.pi,size=scale)       # phi_e 的范围是 [0, 2pi]

        sin_theta = np.sqrt(1 - cos_theta**2)
        cos_alpha = (sin_theta * np.cos(phi) * np.sin(phase) + cos_theta * np.cos(phase)) # dot product
        
        pdf = (x**2) * (3 - 2*x + P_MU * cos_alpha * (2*x - 1))


        # MC
        y = np.random.uniform(0, 1+P_MU,size=scale) # 用于接受-拒绝的随机数
        accept=y<=pdf

        success[accept]=1
        T=T[accept]
        x=x[accept]
        cos_theta=cos_theta[accept]
        phi=phi[accept]

        sin_theta = np.sqrt(1 - cos_theta**2) #重新筛选cos 删了后面报错


        # MC and count
        num_accepted = len(T)
        needed = N - count
        if num_accepted > needed:
            T = T[:needed]
            x = x[:needed]
            cos_theta = cos_theta[:needed]
            phi= phi[:needed]
            num_accepted = needed # 更新实际接受的数量
            

        # ================================== #
        E = x * E_MAX                               # 计算正电子能量
        p= np.sqrt(np.maximum(E**2 - M_E**2, 0))    # 计算正电子动量大小

        px_prime = p * sin_theta * np.cos(phi)      # 计算正电子在 x 方向的动量分量
        py_prime = p * sin_theta * np.sin(phi)      # 计算正电子在 y 方向的动量分量
        pz_prime = p * cos_theta                    # 计算正电子在 z 方向的动量分量

        # position
        PHI_C = OMEGA_C * T                         # 磁场进动相位
        PosX=RADIUS * np.cos(PHI_C)                 # 计算正电子在 x 轴上的位置
        PosY=RADIUS * np.sin(PHI_C)                 # 计算正电子在 y 轴上的位置
        PosZ = np.zeros(num_accepted)               # 生成长度为 num_accepted 的全 0 数组

        # Four-momentum in
        E_lab = GAMMA * (E + BETA * pz_prime)       # 计算正电子在实验室参考系下的能量
        p_long_lab = GAMMA * (E + BETA * pz_prime)  # 切向动量被 Boost 这个实则和Elab一样大小（c=1）
        p_rad_lab = px_prime                        # 径向动量不变
        p_vert_lab = py_prime                       # 垂直方向动量不变
        px_lab = p_rad_lab * np.cos(PHI_C) - p_long_lab * np.sin(PHI_C)
        py_lab = p_rad_lab * np.sin(PHI_C) + p_long_lab * np.cos(PHI_C)
        pz_lab = p_vert_lab


        T=np.round(T,decimals=2)
        E_lab=np.round(E_lab,decimals=2)
        PosX=np.round(PosX,decimals=2)
        PosY=np.round(PosY,decimals=2)
        PosZ=np.round(PosZ,decimals=2)
        px_lab=np.round(px_lab,decimals=2)
        py_lab=np.round(py_lab,decimals=2)
        pz_lab=np.round(pz_lab,decimals=2)

        results.extend(zip(
            np.atleast_1d(T),
            np.atleast_1d(E_lab),
            np.atleast_1d(PosX),
            np.atleast_1d(PosY),
            np.atleast_1d(PosZ),
            np.atleast_1d(px_lab),
            np.atleast_1d(py_lab),
            np.atleast_1d(pz_lab)
        ))
    
        count += num_accepted
        print(f"Generated {count}/{N} events", end='\r')    
    print("")
    return results

detector_data = pd.DataFrame(generation(N_EVENTS), columns=['Time_us', 'Energy_MeV','PosX', 'PosY', 'PosZ', 'px_lab', 'py_lab', 'pz_lab'])
print("Data generation completed. Saving to",FILE_NAME)
detector_data.to_csv(FILE_NAME, index=False)
print("Data saved.")



