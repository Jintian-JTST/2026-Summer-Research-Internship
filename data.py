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
plt.figure(figsize=(10, 8)) # 进一步加大初始尺寸
plt.imshow(
    values.T,
    origin="lower",
    aspect="auto",
    extent=[time_edges[0], time_edges[-1], energy_edges[0], energy_edges[-1]],
    cmap="viridis" # 使用更亮眼的色阶
)

# --- 手动设置显示范围 (放大并去掉不需要的数据) ---
time_min, time_max = 0, 200      # 例如：只看 30us 到 400us
energy_min, energy_max = 1000, 3200 # 例如：只看 1000MeV 到 3200MeV

plt.xlim(time_min, time_max)
plt.ylim(energy_min, energy_max)

plt.colorbar(label="Counts")
plt.xlabel("Time ($\mu s$)")
plt.ylabel("Energy (MeV)")
plt.title(f"Energy-Time Histogram (Zoomed: {time_min}-{time_max} $\mu s$)")

# 尝试让窗口最大化显示
try:
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed') # 针对 Windows 系统的最大化
except:
    pass

plt.tight_layout() # 移除边距
plt.savefig("plot/energy_time_histogram.png", dpi=300, bbox_inches='tight')
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
plt.figure(figsize=(12, 6)) # 加大图表尺寸以适应放大显示
plt.plot(time_centers, counts, lw=1)

# 应用相同的时间显示范围
plt.xlim(time_min, time_max)

plt.xlabel("Time ($\mu s$)") # 添加单位
plt.ylabel(f"Counts (E > {threshold} MeV)") # 添加单位
plt.title(f"Wiggle Plot (Zoomed: {time_min}-{time_max} $\mu s$)")

plt.tight_layout()
plt.savefig("plot/wiggle_plot_run6A.png", dpi=300, bbox_inches='tight') # 保存图片
plt.show()