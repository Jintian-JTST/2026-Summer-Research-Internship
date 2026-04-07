# fft_from_parquet.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# =========================
# 0. 常量
# =========================
sys.path.append(r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship")
from constants import TAU_LAB, OMEGA_A

toy_path = r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship\real data\Data.parquet"
threshold = 1700  # MeV

# 这个 bin 宽度要尽量和你原始时间分箱一致
# 如果你知道原来的 bin 宽度，改这里
time_bin_width_us = 0.15

# =========================
# 1. 读取 parquet
# =========================
try:
    df = pd.read_parquet(toy_path)
except FileNotFoundError:
    print(f"Error: {toy_path} not found.")
    raise SystemExit(1)

if "Time_us" not in df.columns or "Energy_MeV" not in df.columns:
    raise KeyError("Data.parquet must contain columns: 'Time_us' and 'Energy_MeV'")

T = np.asarray(df["Time_us"], dtype=float)
E = np.asarray(df["Energy_MeV"], dtype=float)

# =========================
# 2. 取 E > threshold 的时间谱
# =========================
mask = E > threshold
T_sel = T[mask]

t_min = 0.0
t_max = np.max(T_sel)
time_edges = np.arange(t_min, t_max + time_bin_width_us, time_bin_width_us)
time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])
dt = time_edges[1] - time_edges[0]

wiggle = np.histogram(T_sel, bins=time_edges)[0]

# =========================
# 3. 定义 5 参数拟合函数
# =========================
def five_param_fit(t, N0, tau, A, omega, phi):
    return N0 * np.exp(-t / tau) * (1 + A * np.cos(omega * t + phi))

# =========================
# 4. 拟合 raw wiggle
# =========================
p0 = [
    np.max(wiggle),
    TAU_LAB,
    0.3,
    OMEGA_A,
    0.0
]

print("Starting fit...")
try:
    popt, pcov = curve_fit(
        five_param_fit,
        time_centers,
        wiggle,
        p0=p0,
        maxfev=20000
    )
except RuntimeError as e:
    print(f"Fit failed to converge: {e}")
    raise SystemExit(1)

fit_curve = five_param_fit(time_centers, *popt)

fit_N, fit_tau, fit_A, fit_omega, fit_phi = popt
perr = np.sqrt(np.diag(pcov))

print("--- Fit Results ---")
print(f"N     = {fit_N:.3e} ± {perr[0]:.3e}")
print(f"tau   = {fit_tau:.6f} ± {perr[1]:.6f}")
print(f"A     = {fit_A:.4f} ± {perr[2]:.4f}")
print(f"omega = {fit_omega:.6f} ± {perr[3]:.6f} rad/us")
print(f"phi   = {fit_phi:.4f} ± {perr[4]:.4f} rad")
print("--------------------")

# =========================
# 5. Raw FFT
# =========================
window_raw = np.hanning(len(time_centers))
wiggle_raw_win = wiggle * window_raw
wiggle_raw_win = wiggle_raw_win - np.mean(wiggle_raw_win)

N = len(time_centers)
freq = fftfreq(N, d=dt)[:N // 2]
fft_raw = np.abs(fft(wiggle_raw_win)[:N // 2])

# =========================
# 6. Residual FFT
# =========================
# 统一用 pull residual
residual = (wiggle - fit_curve) / np.sqrt(np.clip(fit_curve, 1, None))
residual = residual - np.mean(residual)

window_res = np.hanning(len(time_centers))
residual_win = residual * window_res

fft_residual = np.abs(fft(residual_win)[:N // 2])

# =========================
# 7. 找峰（只对 residual FFT）
# =========================
mask_valid = freq > 0.1
freq_valid = freq[mask_valid]
fft_valid = fft_residual[mask_valid]

peaks, props = find_peaks(fft_valid, prominence=10)

peak_heights = fft_valid[peaks]
top3_idx = np.argsort(peak_heights)[-3:][::-1]

print("\n=== Top 3 Peaks (Residual FFT) ===")
for i in top3_idx:
    print(
        f"Frequency = {freq_valid[peaks[i]]:.4f} MHz, "
        f"Amplitude = {peak_heights[i]:.2f}"
    )

# =========================
# 8. 画图：raw FFT + residual FFT 同图
# =========================
plt.figure(figsize=(12, 4))

# 避免 log 轴下出现 0
raw_plot = np.clip(fft_raw, 1e-12, None)
res_plot = np.clip(fft_residual, 1e-12, None)

plt.plot(
    freq[mask_valid],
    raw_plot[mask_valid],
    label="Raw Signal FFT",
    lw=0.8,
    alpha=0.6
)

plt.plot(
    freq[mask_valid],
    res_plot[mask_valid],
    label="Residual FFT",
    lw=0.8
)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.yscale("log")
plt.xlim(0.1, 3.0)

ymax = max(np.max(raw_plot[mask_valid]), np.max(res_plot[mask_valid]))
plt.ylim(1, ymax * 1.2)

plt.tight_layout()

save_dir = r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship\real data\plot"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "fft_raw_and_residual_from_parquet.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")

print(f"\nFFT plot saved to: {save_path}")
plt.show()