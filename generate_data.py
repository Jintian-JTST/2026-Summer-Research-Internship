# generate_data.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # 用于生成结构化的“探测器数据”表
from constants import *

print("Data generation in progress...")
def run_g2_toy_mc(n_events):
    
    # 1. 生成每个事件的绝对时间 t_lab 和自旋相位 theta_spin
    t_lab = np.random.exponential(scale=TAU_LAB, size=n_events) # t_lab 是每个muon 衰变的绝对时间，满足指数分布
    #phi_c = OMEGA_C * t_lab      # 储藏环回旋相位
    theta_spin = OMEGA_A * t_lab # 自旋进动相位
    
    #pos_x = RADIUS * np.cos(phi_c) # 正电子在储藏环中的位置，假设在 x-y 平面飞行
    #pos_y = RADIUS * np.sin(phi_c) 
    #pos_z = np.zeros(n_events) # 假设 z 方向没有位移，简化模型
    
    # 2. 准备空数组存储生成的正电子变量
    x_e = np.zeros(n_events)
    cos_theta_e = np.zeros(n_events)
    phi_e = np.zeros(n_events)
    
    # 建立一个掩码，True 表示该位置的粒子还没有成功生成
    pending_mask = np.ones(n_events, dtype=bool)# 初始化为全 True，表示所有事件都需要生成
    M_max = 1.0 + P_MU # PDF 的最大值，确保 y_rand 的范围足够覆盖 PDF 的所有可能值

    # 3. 舍选法循环，严格保证 t_lab 和生成变量的一一对应
    progress = 0.0
    while np.any(pending_mask):
        n_pending = np.sum(pending_mask) # 还需要生成多少个
        
        # 随机生成这批候选者的运动学参数
        x_rand = np.random.uniform(0, 1, n_pending) # x_e 的归一化能量
        c_rand = np.random.uniform(-1, 1, n_pending) # cos(theta_e) 的范围是 [-1, 1]
        p_rand = np.random.uniform(0, 2*np.pi, n_pending) # phi_e 的范围是 [0, 2pi]
        y_rand = np.random.uniform(0, M_max, n_pending) # 用于接受-拒绝的随机数，范围是 [0, PDF 的最大值]
        
        sin_theta_rand = np.sqrt(1 - c_rand**2)
        
        # 提取出目前【仍然 pending】的事件所对应的时间和自旋相位
        t_spin_pending = theta_spin[pending_mask]
        
        # 计算当前的夹角
        cos_theta_spin = (sin_theta_rand * np.cos(p_rand) * np.sin(t_spin_pending) + 
                          c_rand * np.cos(t_spin_pending)) # 计算夹角余弦
        
        pdf = (x_rand**2) * (3 - 2*x_rand + P_MU * cos_theta_spin * (2*x_rand - 1))
        
        accept = y_rand <= pdf
        
        # 找到在总数组中对应的绝对索引位置
        accepted_indices = np.where(pending_mask)[0][accept]
        
        # 将成功接受的数据填入总数组
        x_e[accepted_indices] = x_rand[accept]
        cos_theta_e[accepted_indices] = c_rand[accept]
        phi_e[accepted_indices] = p_rand[accept]
        
        # 更新掩码，把成功生成的标记为 False
        pending_mask[accepted_indices] = False

        if 100 * (1 - np.sum(pending_mask) / n_events) > progress + 5: # 每完成 1% 就更新一次进度显示
            progress = 100 * (1 - np.sum(pending_mask) / n_events)
            print(f"Progress: {progress:.1f}%")

    # 4. 计算静止系下的四动量
    E_rest = x_e * E_MAX
    p_rest = np.sqrt(np.maximum(E_rest**2 - M_E**2, 0))
    #sin_theta_e = np.sqrt(1 - cos_theta_e**2)
    
    # 计算正电子在缪子静止系下的动量分量
    #px_prime = p_rest * sin_theta_e * np.cos(phi_e)
    #py_prime = p_rest * sin_theta_e * np.sin(phi_e)
    pz_prime = p_rest * cos_theta_e
    
    # 5. 洛伦兹变换 (沿 z' 轴 boost)
    E_lab = GAMMA * (E_rest + BETA * pz_prime)
    #px_boost = px_prime
    #py_boost = py_prime
    #pz_boost = GAMMA * (pz_prime + BETA * E_rest)
    
    # 6. 旋转回实验室坐标系 (假设缪子沿 z 轴飞行，正电子的动量在 x'-z' 平面内)
    #px_lab = px_boost * np.cos(phi_c) - pz_boost * np.sin(phi_c)
    #py_lab = px_boost * np.sin(phi_c) + pz_boost * np.cos(phi_c)
    #pz_lab = py_boost 
    

    return (t_lab, 
            #pos_x, 
            #pos_y, 
            E_lab, 
            #px_lab, 
            #py_lab, 
            #pz_lab
            )

(t, 
 #x, 
 #y, 
 E, 
 #px, 
 #py, 
 #pz
 ) = run_g2_toy_mc(N_EVENTS)
print("Data generation finished. Preparing output...")
#  整理输出数据 (满足 "数据包括四动量、时间和位置" 的要求)

detector_data = pd.DataFrame({
    'Time_us': t,
    #'PosX_m': x, 'PosY_m': y,
    'E_MeV': E,
    #'Px_MeV': px, 'Py_MeV': py, 'Pz_MeV': pz
})
#print("\n--- 模拟的探测器截获数据前 5 行 ---")
#print(detector_data.head())

print("Saving simulated data to Feather...")
pandas_output_path = "simulated_detector_data.feather"
detector_data.to_feather(pandas_output_path)
print(f"Simulated data saved to {pandas_output_path}, with {len(detector_data)} events.")
print("Data generation completed.")