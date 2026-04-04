import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from constants import *

def generation(N):

    results = []
    count = 0

    while count < N:
        T = np.random.exponential(scale=TAU_LAB) # t_lab 是每个muon衰变的绝对时间，满足指数分布
        phase = OMEGA_A * T # 自旋进动相位

        x= np.random.uniform(0, 1) # x_e 的归一化能量
        cos_theta = np.random.uniform(-1, 1) # cos(theta_e) 的范围是 [-1, 1]
        phi= np.random.uniform(0, 2*np.pi) # phi_e 的范围是 [0, 2pi]
        sin_theta = np.sqrt(1 - cos_theta**2)
        cos_alpha = (sin_theta * np.cos(phi) * np.sin(phase) + cos_theta * np.cos(phase)) # 计算夹角余弦
        
        pdf = (x**2) * (3 - 2*x + P_MU * cos_alpha * (2*x - 1))

        y = np.random.uniform(0, 2) # 用于接受-拒绝的随机数，范围是 [0, PDF 的最大值]

        if y <= pdf:

            E = x * E_MAX # 计算正电子能量
            p= np.sqrt(np.maximum(E**2 - M_E**2, 0)) # 计算正电子动量大小，使用 np.maximum 防止浮点误差导致负数求平方根
            pz = p * cos_theta # 计算正电子在 z 方向的动量分量
            E_lab = GAMMA * (E + BETA * pz) # 计算正电子在实验室参考系下的能量

            results.append([T.round(2), E_lab.round(2)])
            count += 1
            print(f"\rProgress: {count}/{N} events generated", end="")
        else:
            continue
    print("")
    return results

detector_data = pd.DataFrame(generation(N_EVENTS), columns=['Time_us', 'Energy_MeV'])
print("Data generation completed. Saving to Data.csv...")
detector_data.to_csv("Data.csv", index=False)
print("Data saved.")



    