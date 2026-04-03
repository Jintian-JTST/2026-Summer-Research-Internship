import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 物理常数与模拟参数设置
# ==========================================
M_MU = 105.66 # 缪子质量 (MeV/c^2)
M_E = 0.511   # 电子质量 (MeV/c^2)
E_MAX = (M_MU**2 + M_E**2) / (2 * M_MU) # 静止系下正电子最大能量 (MeV)

N_PARTICLES = 100000 # 模拟衰变的缪子数量
P_MU = 1.0           # 缪子极化度 (0 到 1)
BETA = 0.999        # 缪子速度 (接近光速)
GAMMA = 1/np.sqrt(1-BETA**2) # 洛伦

# ==========================================
# 2. 阶段一：静止参考系下的舍选法模拟
# ==========================================
def generate_rest_frame_kinematics(n_samples, p_mu):
    """使用舍选法生成静止系下的 x 和 cos(theta)"""
    samples_x = []
    samples_c = []
    
    # 概率密度函数的最大值 (当 x=1, cos(theta)=1 时)
    M_max = 1.0 + p_mu 
    
    # 批量生成以提高效率
    while len(samples_x) < n_samples:
        needed = n_samples - len(samples_x)
        # 产生多余的随机数以减少循环次数
        x_rand = np.random.uniform(0, 1, needed * 3)
        c_rand = np.random.uniform(-1, 1, needed * 3)
        y_rand = np.random.uniform(0, M_max, needed * 3)
        
        # 计算 Michel 谱的 PDF 核
        pdf_val = (x_rand**2) * (3 - 2*x_rand + p_mu * c_rand * (2*x_rand - 1))
        
        # 舍选判定
        accept = y_rand <= pdf_val
        samples_x.extend(x_rand[accept])
        samples_c.extend(c_rand[accept])
        
    x = np.array(samples_x[:n_samples])
    c = np.array(samples_c[:n_samples])
    
    # 生成各向同性的方位角 phi
    phi = np.random.uniform(0, 2*np.pi, n_samples)
    
    return x, c, phi

# 获取随机生成的运动学变量
x, cos_theta, phi = generate_rest_frame_kinematics(N_PARTICLES, P_MU)

# 计算静止系下的四维动量
sin_theta = np.sqrt(1 - cos_theta**2)
E_rest = x * E_MAX
# 动量大小 p = sqrt(E^2 - m^2)，使用 np.maximum 防止浮点误差导致负数求平方根
p_rest = np.sqrt(np.maximum(E_rest**2 - M_E**2, 0)) 

px_rest = p_rest * sin_theta * np.cos(phi)
py_rest = p_rest * sin_theta * np.sin(phi)
pz_rest = p_rest * cos_theta

# ==========================================
# 3. 阶段二：洛伦兹变换到实验室参考系
# ==========================================
# 假设缪子沿 Z 轴高速飞行
E_lab = GAMMA * (E_rest + BETA * pz_rest)
px_lab = px_rest # 垂直方向动量不变
py_lab = py_rest # 垂直方向动量不变
pz_lab = GAMMA * (pz_rest + BETA * E_rest)

# ==========================================
# 4. 可视化结果对比
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左图：静止参考系
# 为了画图清晰，我们只画一小部分点 (例如 5000 个)
plot_n = 50000
ax1.scatter(pz_rest[:plot_n], px_rest[:plot_n], s=1, alpha=0.5, color='blue')
ax1.set_title('Rest Frame (pz vs px)\nAsymmetric due to Spin')
ax1.set_xlabel('pz (MeV/c)')
ax1.set_ylabel('px (MeV/c)')
ax1.grid(True)
ax1.axis('equal') # 保持比例一致，你会看到一个向右(pz>0)偏移的近似圆形

# 右图：实验室参考系
ax2.scatter(pz_lab[:plot_n], px_lab[:plot_n], s=1, alpha=0.5, color='red')
ax2.set_title(f'Lab Frame (Gamma={GAMMA})\nRelativistic Beaming Effect')
ax2.set_xlabel('pz (MeV/c)')
ax2.set_ylabel('px (MeV/c)')
ax2.grid(True)
# 注意：这里不能用 equal，因为 pz 被拉伸了极多

plt.tight_layout()
plt.show()


# ==========================================
# 4. 3D 可视化结果对比
# ==========================================
fig = plt.figure(figsize=(16, 7))

# 为了保证 3D 渲染流畅，限制画出的点的数量
plot_n = 4000 

# 左图：静止参考系 (3D)
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(pz_rest[:plot_n], px_rest[:plot_n], py_rest[:plot_n], s=2, alpha=0.4, color='blue')
ax1.set_title('Rest Frame (3D)\nAsymmetric due to Spin')
ax1.set_xlabel('pz (MeV/c) - Spin Direction')
ax1.set_ylabel('px (MeV/c)')
ax1.set_zlabel('py (MeV/c)')
# 强制坐标轴比例一致（部分较新的 matplotlib 版本支持此功能）
try:
    ax1.set_box_aspect([1, 1, 1])
except AttributeError:
    pass

# 右图：实验室参考系 (3D)
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(pz_lab[:plot_n], px_lab[:plot_n], py_lab[:plot_n], s=2, alpha=0.4, color='red')
ax2.set_title(f'Lab Frame (3D, Gamma={GAMMA})\nRelativistic Beaming Effect')
ax2.set_xlabel('pz (MeV/c) - Boost Direction')
ax2.set_ylabel('px (MeV/c)')
ax2.set_zlabel('py (MeV/c)')
try:
    # 实验室系下 pz 被拉长，这里设置一个长条形的视觉比例
    ax2.set_box_aspect([3, 1, 1]) 
except AttributeError:
    pass

plt.tight_layout()
plt.show()



# ==========================================
# 4. 第四步：筛选与画图 (应用能量阈值)
# ==========================================
import matplotlib.pyplot as plt

# 设定阈值: 1.7 GeV = 1700 MeV
THRESHOLD_MEV = 1700.0 

# 1. 生成布尔掩码 (True 表示满足条件，False 表示不满足)
mask = E_lab > THRESHOLD_MEV

# 2. 应用掩码，提取满足条件的电子数据
E_lab_filtered = E_lab[mask]
px_lab_filtered = px_lab[mask]
py_lab_filtered = py_lab[mask]
pz_lab_filtered = pz_lab[mask]

# (可选) 看看这些高能电子在静止系下本来是往哪个方向飞的？
pz_rest_filtered = pz_rest[mask] 

# 打印统计信息
print(f"总模拟粒子数: {N_PARTICLES}")
print(f"能量大于 {THRESHOLD_MEV/1000:.1f} GeV 的粒子数: {len(E_lab_filtered)}")
print(f"筛选保留率: {len(E_lab_filtered) / N_PARTICLES * 100:.2f}%\n")

# ==========================================
# 画图展示筛选前后的差异
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：实验室系下的能谱对比 (1D 直方图)
# 画出所有电子的能谱 (灰色背景)
ax1.hist(E_lab, bins=100, range=(0, 6000), color='gray', alpha=0.5, label='All Electrons')
# 画出筛选后的能谱 (红色)
ax1.hist(E_lab_filtered, bins=100, range=(0, 6000), color='red', alpha=0.7, label=f'E > {THRESHOLD_MEV/1000:.1f} GeV')
ax1.axvline(THRESHOLD_MEV, color='black', linestyle='--', label='Threshold')
ax1.set_title('Laboratory Frame Energy Spectrum')
ax1.set_xlabel('E_lab (MeV)')
ax1.set_ylabel('Counts')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右图：这些高能电子在“静止参考系”中的发射方向 (pz) 分布
# 这能揭示高能切的物理后果：我们筛选出的是哪一部分电子？
ax2.hist(pz_rest, bins=100, range=(-60, 60), color='gray', alpha=0.5, label='All Electrons (Rest Frame)')
ax2.hist(pz_rest_filtered, bins=100, range=(-60, 60), color='blue', alpha=0.7, label=f'Those with E_lab > {THRESHOLD_MEV/1000:.1f} GeV')
ax2.set_title('Rest Frame Longitudinal Momentum (pz_rest)\nWhat does the high-energy cut select?')
ax2.set_xlabel('pz_rest (MeV/c)')
ax2.set_ylabel('Counts')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()