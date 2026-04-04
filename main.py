import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # 用于生成结构化的“探测器数据”表

# ==========================================
# 1. 物理常数与实验参数 (Toy 设定)
# ==========================================
M_MU = 105.66     # 缪子质量 MeV
M_E = 0.511       # 电子质量 MeV
E_MAX = (M_MU**2 + M_E**2) / (2 * M_MU)

GAMMA = 29.3      # 真实 g-2 实验的魔术伽马值 (Magic Gamma) 约等于 29.3
BETA = np.sqrt(1 - 1/GAMMA**2)
TAU_REST = 2.2    # 静止寿命 us
TAU_LAB = GAMMA * TAU_REST # 实验室寿命 us

# 频率参数 (为了 Toy MC 能在短时间内看出效果，使用夸张的频率)
OMEGA_C = 2.0     # 缪子动量在储藏环中的回旋角频率 (rad/us)
OMEGA_A = 0.5     # 反常自旋进动角频率 omega_a (rad/us) -> 决定了 wiggle 的频率
RADIUS = 7.0      # 假设的储藏环半径 (米)

N_EVENTS = 500000 # 模拟事件数 (数据量大一点 wiggle 会更平滑)
P_MU = 1.0        # 极化度


# ==========================================
# 2. 核心 Monte Carlo 生成器 (已修复时间关联 Bug)
# ==========================================
def run_g2_toy_mc(n_events):
    print("正在生成 MC 数据，请稍候...")
    
    # 1. 生成每个事件的绝对时间 t_lab 和自旋相位 theta_spin
    t_lab = np.random.exponential(scale=TAU_LAB, size=n_events)
    phi_c = OMEGA_C * t_lab      # 储藏环回旋相位
    theta_spin = OMEGA_A * t_lab # 自旋进动相位
    
    pos_x = RADIUS * np.cos(phi_c)
    pos_y = RADIUS * np.sin(phi_c)
    pos_z = np.zeros(n_events) 
    
    # 2. 准备空数组存储生成的正电子变量
    x_e = np.zeros(n_events)
    cos_theta_e = np.zeros(n_events)
    phi_e = np.zeros(n_events)
    
    # 建立一个掩码，True 表示该位置的粒子还没有成功生成
    pending_mask = np.ones(n_events, dtype=bool)
    M_max = 1.0 + P_MU
    
    # 3. 舍选法循环，严格保证 t_lab 和生成变量的一一对应
    while np.any(pending_mask):
        n_pending = np.sum(pending_mask) # 还需要生成多少个
        
        # 随机生成这批候选者的运动学参数
        x_rand = np.random.uniform(0, 1, n_pending)
        c_rand = np.random.uniform(-1, 1, n_pending)
        p_rand = np.random.uniform(0, 2*np.pi, n_pending)
        y_rand = np.random.uniform(0, M_max, n_pending)
        
        sin_theta_rand = np.sqrt(1 - c_rand**2)
        
        # 提取出目前【仍然 pending】的事件所对应的时间和自旋相位
        t_spin_pending = theta_spin[pending_mask]
        
        # 计算当前的夹角
        cos_theta_spin = (sin_theta_rand * np.cos(p_rand) * np.sin(t_spin_pending) + 
                          c_rand * np.cos(t_spin_pending))
        
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
        
    # 4. 计算静止系下的四动量
    E_rest = x_e * E_MAX
    p_rest = np.sqrt(np.maximum(E_rest**2 - M_E**2, 0))
    sin_theta_e = np.sqrt(1 - cos_theta_e**2)
    
    px_prime = p_rest * sin_theta_e * np.cos(phi_e)
    py_prime = p_rest * sin_theta_e * np.sin(phi_e)
    pz_prime = p_rest * cos_theta_e
    
    # 5. 洛伦兹变换 (沿 z' 轴 boost)
    E_lab = GAMMA * (E_rest + BETA * pz_prime)
    px_boost = px_prime
    py_boost = py_prime
    pz_boost = GAMMA * (pz_prime + BETA * E_rest)
    
    # 6. 坐标旋转回绝对系
    px_lab = px_boost * np.cos(phi_c) - pz_boost * np.sin(phi_c)
    py_lab = px_boost * np.sin(phi_c) + pz_boost * np.cos(phi_c)
    pz_lab = py_boost 
    
    return t_lab, pos_x, pos_y, pos_z, E_lab, px_lab, py_lab, pz_lab
# 运行模拟
t, x, y, z, E, px, py, pz = run_g2_toy_mc(N_EVENTS)

# ==========================================
# 3. 整理输出数据 (满足 "数据包括四动量、时间和位置" 的要求)
# ==========================================
# 将数据存入 Pandas DataFrame，模拟真实的 ROOT Tree 或 CSV 数据结构
detector_data = pd.DataFrame({
    'Time_us': t,
    'PosX_m': x, 'PosY_m': y, 'PosZ_m': z,
    'E_MeV': E,
    'Px_MeV': px, 'Py_MeV': py, 'Pz_MeV': pz
})

print("\n--- 模拟的探测器截获数据前 5 行 ---")
print(detector_data.head())

# ==========================================
# 4. 施加高能切阈值并绘制 Wiggle Plot
# ==========================================
THRESHOLD = 1800.0 # 设定能量阈值 1.8 GeV
high_energy_data = detector_data[detector_data['E_MeV'] > THRESHOLD]

fig, ax = plt.subplots(figsize=(10, 6))

# 画出时间直方图 (Wiggle Plot)
counts, bins, _ = ax.hist(high_energy_data['Time_us'], bins=120, range=(0, 150), 
                          color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)

ax.set_title(f'Toy MC: Muon g-2 Wiggle Plot (E > {THRESHOLD/1000:.1f} GeV)\nAnomalous Precession $\\omega_a$ = {OMEGA_A} rad/us', fontsize=14)
ax.set_xlabel('Time in Lab Frame ($\mu s$)', fontsize=12)
ax.set_ylabel('Number of High-Energy Positrons ($N_e$)', fontsize=12)
ax.set_yscale('log') # 真实实验中通常也会使用对数坐标，方便拟合
ax.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig('wiggle_plot.png', dpi=300)
print("\nPlot saved as wiggle_plot.png")
plt.show()