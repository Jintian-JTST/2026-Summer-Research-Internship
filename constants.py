# constants.py
import numpy as np

M_MU = 105.658     # 缪子质量 MeV/c^2
M_E = 0.511       # 电子质量 MeV/c^2
E_MAX = (M_MU**2 + M_E**2) / (2 * M_MU) # 静止系下正电子获得最大能量 MeV

GAMMA = 29.3 # 洛伦兹因子
BETA = np.sqrt(1 - 1/GAMMA**2) # 根据 GAMMA 计算 BETA，确保物理一致性
TAU_REST = 2.2    # 静止寿命 s
TAU_LAB = GAMMA * TAU_REST # 实验室寿命 s

OMEGA_C = 42112502e-6     # 缪子动量在储藏环中的回旋角频率 (rad/s)
OMEGA_A = 1439358e-6     # 反常自旋进动角频率 omega_a (rad/s) -> 决定了 wiggle 的频率
RADIUS = 7.112     # 储藏环半径 (m)

N_EVENTS = 500000 # 模拟事件数 (数据量大一点 wiggle 会更平滑)
P_MU = 1.0        # 极化度

THRESHOLD = 1700.0 # 设定能量阈值 1.7 GeV
