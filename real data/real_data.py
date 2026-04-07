# data.py
# from real data in run6A.root, extract the 4-momentum, time, and position information for each detected positron
import uproot
import numpy as np
import matplotlib.pyplot as plt

# 1) 打开 ROOT 文件
file = uproot.open("D:\\Users\\JTST\\Desktop\\Desktop\\SI\\2026-Summer-Research-Internship\\real data\\run6A.root")

# 2) 取出二维直方图
hist = file["et_spectrum"]

# 3) 取 bin 内容和 bin 边界
values = hist.values()
time_edges = hist.axis(0).edges()
energy_edges = hist.axis(1).edges()

'''# 4) 画二维能量-时间图
plt.figure(figsize=(12, 6)) # 加大图表尺寸以适应放大显示
plt.imshow(
    values.T,
    origin="lower",
    aspect="auto",
    extent=[time_edges[0], time_edges[-1], energy_edges[0], energy_edges[-1]],
    cmap="viridis" # 使用更亮眼的色阶
)'''

# --- 手动设置显示范围 (放大并去掉不需要的数据) ---
time_min, time_max = 0, 660      # 例如：只看 30us 到 400us
energy_min, energy_max = 1700, 3200 # 例如：只看 1000MeV 到 3200MeV

'''plt.xlim(time_min, time_max)
plt.ylim(energy_min, energy_max)
'''

'''plt.colorbar(label="Counts")
plt.xlabel("Time ($\mu s$)")
plt.ylabel("Energy (MeV)")
plt.title(f"Energy-Time Histogram (Zoomed: {time_min}-{time_max} $\mu s$)")
#plt.yscale("log") # 能量轴通常使用对数坐标，方便观察高能尾部的 wiggle 结构
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout() # 移除边距
plt.savefig("D:\\Users\\JTST\\Desktop\\Desktop\\SI\\2026-Summer-Research-Internship\\real data\\plot\\real_energy_time_histogram.png", dpi=300, bbox_inches='tight')
plt.show()'''

# 5) 设定能量阈值，做 wiggle plot
threshold = 1700  # 单位按你的数据来，常见是 MeV

print(len(time_edges) - 1)
print("Number of total positrons:", values.sum())


# 找到阈值对应的能量 bin
energy_bin = np.searchsorted(energy_edges, threshold, side="left")

# 假设第 0 轴是时间，第 1 轴是能量：
counts = values[:, energy_bin:].sum(axis=1)
time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])

print("Number of positrons above threshold:", values[:, energy_bin:].sum())

'''# 6) 画 wiggle plot
plt.figure(figsize=(12, 6)) # 加大图表尺寸以适应放大显示
plt.scatter(time_centers, counts, s=0.5)

# 应用相同的时间显示范围
plt.xlim(time_min, time_max)

plt.xlabel("Time ($\mu s$)") # 添加单位
plt.ylabel(f"Counts (E > {threshold} MeV)") # 添加单位
#plt.title(f"Wiggle Plot (Zoomed: {time_min}-{time_max} $\mu s$)")
plt.yscale("log") # 对数坐标更清晰地展示 wiggle 结构
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("D:\\Users\\JTST\\Desktop\\Desktop\\SI\\2026-Summer-Research-Internship\\real data\\plot\\real_wiggle_plot_run6A.png", dpi=300, bbox_inches='tight') # 保存图片
plt.show()

'''

'''#生成时间除以100的余数为横坐标的wiggle plot
time_mod = time_centers % 100
plt.figure(figsize=(12, 10))
plt.scatter(time_mod, counts, s=0.5)
plt.xlabel("Time mod 100 ($\mu s$)")
plt.ylabel(f"Counts (E > {threshold} MeV)")
#plt.title(f"Wiggle Plot (Time mod 100 $\mu s$)")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("D:\\Users\\JTST\\Desktop\\Desktop\\SI\\2026-Summer-Research-Internship\\real data\\plot\\real_wiggle_plot_time_mod_100.png", dpi=300, bbox_inches='tight')
plt.show()

'''







# ===== FFT of wiggle signal =====
import numpy as np
import matplotlib.pyplot as plt

# 取出有效区间
mask = (time_centers >= time_min) & (time_centers <= time_max) & (counts > 0)
t = time_centers[mask]
y = counts[mask].astype(float)

# 采样间隔 (单位: us)
dt = t[1] - t[0]
print("dt =", dt, "us")
print("Nyquist frequency =", 1 / (2 * dt), "1/us = MHz")

# ---- 去指数衰减趋势：log-linear 粗拟合 ----
# y ~ exp(a t + b)
coef = np.polyfit(t, np.log(y), 1)
trend = np.exp(np.polyval(coef, t))

# 残差信号：去掉整体衰减和 DC 分量
y_det = y / trend
y_det = y_det - np.mean(y_det)

# 加窗，减少谱泄漏
window = np.hanning(len(y_det))
y_win = y_det * window

# FFT
Y = np.fft.rfft(y_win)
freq = np.fft.rfftfreq(len(y_win), d=dt)   # 单位：1/us = MHz
amp = np.abs(Y)

# 画频谱
plt.figure(figsize=(12, 6))
plt.plot(freq, amp)
plt.xlim(0, 2.0)   # 先看 0~1 MHz，g-2 主峰通常在这附近
plt.xlabel("Frequency (MHz)")
plt.ylabel("FFT amplitude")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship\real data\plot\real_fft_spectrum_run6A.png", dpi=300, bbox_inches="tight")
plt.show()

# 打印主峰位置
peak_idx = np.argmax(amp[1:]) + 1   # 跳过直流分量
print("Peak frequency =", freq[peak_idx], "MHz")
print("Peak amplitude =", amp[peak_idx])