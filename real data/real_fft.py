# fft_left_style_only.py
import os
import sys
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq

# 如果 constants.py 在这个目录下，就保留这一行
sys.path.append(r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship")
from constants import TAU_LAB, OMEGA_A

# =========================
# 1. 读取 ROOT 数据
# =========================
root_path = r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship\real data\run6A.root"

try:
    with uproot.open(root_path) as f:
        hist = f["et_spectrum"]
        values = hist.values()
        time_edges = hist.axis(0).edges()
        energy_edges = hist.axis(1).edges()
except FileNotFoundError:
    print(f"Error: {root_path} not found.")
    raise SystemExit(1)

time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])
dt = time_centers[1] - time_centers[0]

# =========================
# 2. 设定能量阈值，提取 real data 的时间谱
# =========================
threshold = 1700  # MeV
energy_bin = np.searchsorted(energy_edges, threshold, side="left")
wiggle_real = np.sum(values[:, energy_bin:], axis=1)

# =========================
# 3. 读取 Toy MC
#    这里按你的文件名改；如果列名不同也在这里改
# =========================
toy_path = "Data.parquet"
try:
    import pandas as pd
    df = pd.read_parquet(toy_path)
except FileNotFoundError:
    print(f"Error: {toy_path} not found.")
    raise SystemExit(1)

T_toy = np.asarray(df["Time_us"])
E_toy = np.asarray(df["Energy_MeV"])

mask_toy = E_toy > threshold
wiggle_toy, _ = np.histogram(T_toy[mask_toy], bins=time_edges)

# =========================
# 4. 定义 5 参数拟合函数
# =========================
def five_param_fit(t, N0, tau, A, omega, phi):
    return N0 * np.exp(-t / tau) * (1 + A * np.cos(omega * t + phi))

# =========================
# 5. 分别拟合 real / toy
# =========================
p0_real = [np.max(wiggle_real), TAU_LAB, 0.3, OMEGA_A, 0.0]
p0_toy  = [np.max(wiggle_toy),  TAU_LAB, 0.3, OMEGA_A, 0.0]

popt_real, _ = curve_fit(
    five_param_fit,
    time_centers,
    wiggle_real,
    p0=p0_real,
    maxfev=20000
)

popt_toy, _ = curve_fit(
    five_param_fit,
    time_centers,
    wiggle_toy,
    p0=p0_toy,
    maxfev=20000
)

fit_real = five_param_fit(time_centers, *popt_real)
fit_toy = five_param_fit(time_centers, *popt_toy)

# =========================
# 6. 统一 residual 定义 + 去均值 + 加窗
# =========================
residual_real = (wiggle_real - fit_real) / np.sqrt(np.clip(fit_real, 1, None))
residual_toy  = (wiggle_toy  - fit_toy)  / np.sqrt(np.clip(fit_toy,  1, None))

residual_real -= np.mean(residual_real)
residual_toy  -= np.mean(residual_toy)

window = np.hanning(len(time_centers))
residual_real_win = residual_real * window
residual_toy_win  = residual_toy * window

# =========================
# 6.5 原始数据 FFT（未减残差）
# =========================
N = len(time_centers)
window_raw = np.hanning(len(time_centers))

wiggle_real_win = wiggle_real * window_raw

# 去掉 DC（否则0频会炸）
wiggle_real_win = wiggle_real_win - np.mean(wiggle_real_win)

fft_raw = np.abs(fft(wiggle_real_win)[:N // 2])


# =========================
# 7. FFT
# =========================
freq = fftfreq(N, d=dt)[:N // 2]
fft_real = np.abs(fft(residual_real_win)[:N // 2])
fft_toy  = np.abs(fft(residual_toy_win)[:N // 2])

plt.figure(figsize=(12, 4))

mask = freq > 0.1

# 原始 FFT（未减残差）
plt.plot(
    freq[mask],
    fft_raw[mask],
    label="Raw Signal FFT",
    lw=0.8,
    alpha=0.6
)

# 残差 FFT
plt.plot(
    freq[mask],
    fft_real[mask],
    label="Residual FFT",
    lw=0.8
)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.yscale("log")
plt.xlim(0.1, 3.0)
plt.ylim(1, 80000000)

plt.tight_layout()

save_dir = r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship\real data\plot"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "111fft_residual_left_style.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")

print(f"FFT plot saved to: {save_path}")
plt.show()



from scipy.signal import find_peaks

# 只在有效频率范围找（和画图一致）
mask = freq > 0.1
freq_valid = freq[mask]
fft_valid = fft_real[mask]

# 找峰（可以调 height / prominence 控制“显著性”）
peaks, properties = find_peaks(
    fft_valid,
    prominence=10  # 这个参数很关键，可以调（10~100之间试）
)

# 取最强的三个峰
peak_heights = fft_valid[peaks]
top3_idx = np.argsort(peak_heights)[-3:][::-1]

print("\n=== Top 3 Peaks (Real Data FFT) ===")
for i in top3_idx:
    print(f"Frequency = {freq_valid[peaks[i]]:.4f} MHz, Amplitude = {peak_heights[i]:.2f}")