# constants.py
import numpy as np

M_MU = 105.658     # 缪子质量 MeV/c^2
M_E = 0.511       # 电子质量 MeV/c^2
E_MAX = (M_MU**2 + M_E**2) / (2 * M_MU) # 静止系下正电子获得最大能量 MeV

BETA = 0.999      # 缪子速度 
GAMMA = 1/np.sqrt(1-BETA**2) # 洛伦兹因子
TAU_REST = 2.2e-6    # 静止寿命 s
TAU_LAB = GAMMA * TAU_REST # 实验室寿命 s

OMEGA_C = 42112502     # 缪子动量在储藏环中的回旋角频率 (rad/s)
OMEGA_A = 1439358     # 反常自旋进动角频率 omega_a (rad/s) -> 决定了 wiggle 的频率
RADIUS = 7.112     # 储藏环半径 (m)

N_EVENTS = 500000 # 模拟事件数 (数据量大一点 wiggle 会更平滑)
P_MU = 1.0        # 极化度
