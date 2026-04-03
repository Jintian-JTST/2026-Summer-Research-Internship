import numpy as np
import matplotlib.pyplot as plt

N = 30000  # 点多一点更平滑

def W(theta):
    return 0.5 + np.cos(theta)/6

theta_list = []
phi_list = []

# 接受-拒绝采样
while len(theta_list) < N:
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2*np.pi)

    if np.random.rand() < W(theta):
        theta_list.append(theta)
        phi_list.append(phi)

theta = np.array(theta_list)
phi = np.array(phi_list)

# 单位球面
X = np.sin(theta)*np.cos(phi)
Y = np.sin(theta)*np.sin(phi)
Z = np.cos(theta)

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X, Y, Z, s=1, alpha=0.6)

# 自旋方向（关键！）
ax.plot([0,0],[0,0],[-1.2,1.2], linewidth=3)

ax.set_box_aspect([1,1,1])
ax.set_title("Muon decay angular distribution (point cloud)")

plt.show()


plt.hist(np.cos(theta), bins=50)
plt.xlabel("cos(theta)")
plt.ylabel("counts")
plt.show()