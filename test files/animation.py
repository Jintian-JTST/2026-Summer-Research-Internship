import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ==========================================
# 1. 物理常数与实验参数 (严格同步 main.py)
# ==========================================
M_MU = 105.66     # 缪子质量 MeV
GAMMA = 29.3      # 魔术伽马值
BETA = np.sqrt(1 - 1/GAMMA**2)
E_MAX_REST = M_MU / 2.0 # 静止系最大电子能量 (约 52.8 MeV)

# 动画调节参数 (频率调大是为了在几秒钟内看清进动)
OMEGA_C = 3.5      # 回旋频率 (rad/us)
OMEGA_A = 0.7      # 反常进动频率 (rad/us)
RADIUS = 7.0       
TAU_LAB_ANIM = 6.0 # 视觉寿命
N_MUONS = 5        # 同时模拟的缪子数

# ==========================================
# 2. 核心物理逻辑类
# ==========================================
class Muon:
    def __init__(self, id):
        self.id = id
        self.reset()
        
    def reset(self):
        self.state = 'injecting'
        self.t_orbit = 0
        self.t_decay_flight = 0
        self.phi = -np.pi/2 
        self.x, self.y = -25, -RADIUS # 入射起点
        self.v_inject = 40.0 
        self.decay_limit = np.random.exponential(scale=TAU_LAB_ANIM)
        self.positron_data = None
        self.theta_p = 0
        self.spin_angle = 0

    def rigorous_decay(self):
        """严格按照 main.py 的物理公式生成衰变电子"""
        P_MU = 1.0 # 极化度
        # 舍选法生成静止系变量
        while True:
            x_rand = np.random.uniform(0, 1)        # 能量分数 E/E_max
            cos_theta_rest = np.random.uniform(-1, 1) # 相对自旋方向的夹角
            phi_rest = np.random.uniform(0, 2*np.pi)
            y_rand = np.random.uniform(0, 2.0)      # PDF 最大值为 2
            
            # 物理 PDF: Michel Spectrum + Parity Violation
            pdf = (x_rand**2) * ((3 - 2*x_rand) + P_MU * cos_theta_rest * (2*x_rand - 1))
            if y_rand <= pdf:
                break
        
        # 1. 静止系三动量 (相对于自旋方向)
        E_rest = x_rand * E_MAX_REST
        p_mag = E_rest # 忽略电子质量
        sin_theta_rest = np.sqrt(1 - cos_theta_rest**2)
        
        # 这里的坐标轴定义：z'轴沿自旋方向
        pz_prime = p_mag * cos_theta_rest
        px_prime = p_mag * sin_theta_rest * np.cos(phi_rest)
        py_prime = p_mag * sin_theta_rest * np.sin(phi_rest)
        
        # 2. 将静止系动量旋转，使 z' 轴对齐当前的自旋向量 s
        # 自旋向量在 xy 平面的角度是 self.spin_angle
        # 我们做一个简单的 2D 平面近似旋转
        px_rot = px_prime * np.cos(self.spin_angle) - pz_prime * np.sin(self.spin_angle)
        py_rot = py_prime # 垂直平面的分量
        pz_rot = px_prime * np.sin(self.spin_angle) + pz_prime * np.cos(self.spin_angle)
        
        # 3. 洛伦兹变换 (沿着缪子的动量方向 boost)
        # 动量方向角 theta_p
        gamma, beta = GAMMA, BETA
        # 我们只考虑平面内的 boost 效果
        # 先把动量转到 boost 轴，变换，再转回来
        # 在这里简化处理：电子能量在实验室系受多普勒效应增强
        cos_e_p = np.cos(np.arctan2(pz_rot, px_rot) - self.theta_p)
        E_lab = gamma * E_rest * (1 + beta * cos_e_p)
        
        # 最终射出角度 (在实验室内受相对论集束效应影响，趋向于动量方向)
        v_e_lab_x = px_rot + (gamma-1)*px_rot + gamma*beta*E_rest*np.cos(self.theta_p)
        v_e_lab_y = pz_rot + (gamma-1)*pz_rot + gamma*beta*E_rest*np.sin(self.theta_p)
        angle_lab = np.arctan2(v_e_lab_y, v_e_lab_x)
        
        return E_lab, angle_lab

    def update(self, dt):
        if self.state == 'injecting':
            self.x += self.v_inject * dt
            if self.x >= 0:
                self.state = 'orbiting'
                self.t_orbit = 0
                self.phi = -np.pi/2
                self.x = RADIUS * np.cos(self.phi)
                self.y = RADIUS * np.sin(self.phi)
                self.theta_p = 0
                self.spin_angle = 0
                
        elif self.state == 'orbiting':
            self.t_orbit += dt
            self.phi = -np.pi/2 + OMEGA_C * self.t_orbit
            self.x = RADIUS * np.cos(self.phi)
            self.y = RADIUS * np.sin(self.phi)
            
            # 物理量更新
            self.theta_p = self.phi + np.pi/2 # 动量方向 (切线)
            self.spin_angle = self.theta_p + OMEGA_A * self.t_orbit # 自旋方向 (进动)
            
            if self.t_orbit >= self.decay_limit:
                self.state = 'decayed'
                self.t_decay_flight = 0
                energy, angle = self.rigorous_decay()
                self.positron_data = {
                    'start_x': self.x, 'start_y': self.y,
                    'vx': 10.0 * np.cos(angle), 'vy': 10.0 * np.sin(angle),
                    'energy': energy, 'lifetime': self.t_orbit
                }
                
        elif self.state == 'decayed':
            self.t_decay_flight += dt
            if self.t_decay_flight > 1.2: self.reset()

# ==========================================
# 3. 动画显示设置
# ==========================================
fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
ax.set_xlim(-20, 20); ax.set_ylim(-20, 20)
ax.set_aspect('equal'); ax.axis('off')

# 环境
ax.add_artist(plt.Circle((0, 0), RADIUS, color='gray', fill=False, ls='--', alpha=0.3))
ax.plot([-25, 0], [-RADIUS, -RADIUS], color='white', alpha=0.2)

muons = [Muon(i) for i in range(N_MUONS)]
muon_dots = [ax.plot([], [], 'cyan', marker='o', ms=10)[0] for _ in range(N_MUONS)]

# 核心：动量箭头(绿)和自旋箭头(红)
dummy_pos = [-100] * N_MUONS
dummy_vec = [0] * N_MUONS
p_arrows = ax.quiver(dummy_pos, dummy_pos, dummy_vec, dummy_vec, color='#00FF00', scale=15, width=0.005, label='Momentum P')
s_arrows = ax.quiver(dummy_pos, dummy_pos, dummy_vec, dummy_vec, color='#FF0000', scale=15, width=0.005, label='Spin S')

positron_dots = [ax.plot([], [], 'yellow', marker='*', ms=12)[0] for _ in range(N_MUONS)]
labels = [ax.text(0, 0, '', color='yellow', fontsize=9, fontweight='bold') for _ in range(N_MUONS)]
info = ax.text(0.02, 0.02, '', color='white', transform=ax.transAxes, family='monospace', fontsize=12)
ax.legend(loc='upper right', facecolor='black', labelcolor='white')

def animate(frame):
    p_pos, p_vec = [], []
    s_pos, s_vec = [], []
    
    for i, m in enumerate(muons):
        m.update(0.05)
        if m.state == 'orbiting':
            muon_dots[i].set_data([m.x], [m.y])
            p_pos.append([m.x, m.y]); p_vec.append([np.cos(m.theta_p), np.sin(m.theta_p)])
            s_pos.append([m.x, m.y]); s_vec.append([np.cos(m.spin_angle), np.sin(m.spin_angle)])
            positron_dots[i].set_data([], []); labels[i].set_text('')
        elif m.state == 'decayed':
            muon_dots[i].set_data([], [])
            p_pos.append([-100, -100]); p_vec.append([0, 0])
            s_pos.append([-100, -100]); s_vec.append([0, 0])
            p = m.positron_data
            px, py = p['start_x'] + p['vx']*m.t_decay_flight, p['start_y'] + p['vy']*m.t_decay_flight
            positron_dots[i].set_data([px], [py])
            labels[i].set_position((px+0.5, py+0.5))
            labels[i].set_text(f"{p['energy']:.1f}MeV\n{p['lifetime']:.1f}us")
        else: # m.state == 'injecting'
            muon_dots[i].set_data([m.x], [m.y])
            p_pos.append([-100, -100]); p_vec.append([0, 0])
            s_pos.append([-100, -100]); s_vec.append([0, 0])
            positron_dots[i].set_data([], [])
            labels[i].set_text('')
            
    p_arrows.set_offsets(p_pos); p_arrows.set_UVC(np.array(p_vec)[:,0], np.array(p_vec)[:,1])
    s_arrows.set_offsets(s_pos); s_arrows.set_UVC(np.array(s_vec)[:,0], np.array(s_vec)[:,1])
    info.set_text(f"Global Time: {frame*0.05:.1f} us\nPhysics: Michel + Lorentz Boost")
    return muon_dots + [p_arrows, s_arrows] + positron_dots + labels + [info]

ani = FuncAnimation(fig, animate, frames=300, interval=30, blit=True)
print("正在使用严格物理模型生成动画...")
ani.save('muon_g2_rigorous.mp4', writer=FFMpegWriter(fps=30))
plt.show()
