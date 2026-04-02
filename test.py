import uproot
import numpy as np
import matplotlib.pyplot as plt

# 1) 打开 ROOT 文件
file = uproot.open("run6A.root")

# 2) 取出二维直方图
hist = file["et_spectrum"]

# 3) 取 bin 内容和 bin 边界
values = hist.values()
time_edges = hist.axis(0).edges()
energy_edges = hist.axis(1).edges()

# 4) 画二维能量-时间图
plt.figure(figsize=(8, 5))
plt.imshow(
    values.T,              # 转置后更符合常见显示习惯
    origin="lower",
    aspect="auto",
    extent=[time_edges[0], time_edges[-1], energy_edges[0], energy_edges[-1]]
)
plt.colorbar(label="Counts")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Energy-Time Histogram")
plt.show()

# 5) 设定能量阈值，做 wiggle plot
threshold = 1700  # 单位按你的数据来，常见是 MeV

# 找到阈值对应的能量 bin
energy_bin = np.searchsorted(energy_edges, threshold, side="left")

# 假设第 0 轴是时间，第 1 轴是能量：
counts = values[:, energy_bin:].sum(axis=1)

# 时间 bin 中心
time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])

# 6) 画 wiggle plot
plt.figure(figsize=(8, 4))
plt.plot(time_centers, counts, lw=1)
plt.xlabel("Time")
plt.ylabel(f"Counts (E > {threshold})")
plt.title("Wiggle Plot")
plt.show()