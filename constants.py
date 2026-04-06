# constants.py
import numpy as np

M_MU = 105.658     # 缪子质量 MeV/c^2
M_E = 0.511       # 电子质量 MeV/c^2
E_MAX = (M_MU**2 + M_E**2) / (2 * M_MU) # 静止系下正电子获得最大能量 MeV

GAMMA = 29.3 # 洛伦兹因子
BETA = np.sqrt(1 - 1/GAMMA**2) # 根据 GAMMA 计算 BETA，确保物理一致性
TAU_REST = 2.2    # 静止寿命 μs
TAU_LAB = GAMMA * TAU_REST # 实验室寿命 μs

OMEGA_C = 42.11     # 缪子动量在储藏环中的回旋角频率 (rad/μs)
OMEGA_A = 1.439    # 反常自旋进动角频率 omega_a (rad/μs) -> 决定了 wiggle 的频率
#RADIUS = 7.112     # 储藏环半径 (m)

N_EVENTS = 50000 # 模拟事件数 (数据量大一点 wiggle 会更平滑)
P_MU = 0.95        # 极化度

THRESHOLD = 1700.0 # 设定能量阈值 1.7 GeV
TIME_MAX = 300.0    # 绘图和拟合的时间范围上限 (us)

FILE_NAME = "Data.csv"