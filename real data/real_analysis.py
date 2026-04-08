# fft_left_style_only.py
import os
import sys
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

import pandas as pd

NUM=5000
# 如果 constants.py 在这个目录下，就保留这一行
sys.path.append(r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship")
from constants import TAU_LAB, OMEGA_A

# =========================
# 1. 读取 ROOT 数据
# =========================
ROOT_FILE = r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship\real data\run6A.root"
#OUT_DIR = r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship\real data\plot"

THRESHOLD = 1700  # MeV
TIME_MIN = 10
TIME_MAX = 650.0

#os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Load data
# =========================
with uproot.open(ROOT_FILE) as file:
    hist = file["et_spectrum"]
    values = hist.values()
    time_edges = hist.axis(0).edges()
    energy_edges = hist.axis(1).edges()

bin_centers = 0.5 * (time_edges[:-1] + time_edges[1:])

energy_bin = np.searchsorted(energy_edges, THRESHOLD, side="left")
counts = values[:, energy_bin:].sum(axis=1)





# 拟合函数
def wiggle_fit_function(t, N, A, omega, phi_0):
    """
    N: Normalization constant
    A: Asymmetry parameter
    omega: Anomalous precession frequency (自由参数)
    phi_0: Initial phase
    """
    return N * np.exp(-t / TAU_LAB) * (1 + A * np.cos(omega * t - phi_0))

decay_factor = np.exp(-bin_centers[0] / TAU_LAB)

# 初始参数猜测 [N, A, omega, phi_0]
initial_N = counts[0] / decay_factor
initial_A = 0.4  
initial_omega = OMEGA_A  # 使用常数作为拟合的初始起点帮助收敛
initial_phi_0 = 0.0 

popt, pcov = curve_fit(wiggle_fit_function,
                       bin_centers[counts>0],
                       counts[counts>0],
                       p0=[initial_N, initial_A, initial_omega, initial_phi_0],
                       sigma=np.sqrt(counts[counts>0]),
                       absolute_sigma=True,maxfev=10000)

fit_N, fit_A, fit_omega, fit_phi_0 = popt
perr = np.sqrt(np.diag(pcov))
err_N, err_A, err_omega, err_phi_0 = perr


print("--- Fit Results ---")
print(f"N     = {fit_N:.3e} ± {err_N:.3e}")
print(f"A     = {fit_A:.4f} ± {err_A:.4f}")
print(f"omega = {fit_omega:.6f} ± {err_omega:.6f} rad/us")
print(f"phi_0 = {fit_phi_0:.4f} ± {err_phi_0:.4f} rad")
print("--------------------")
pd.DataFrame({
    'Parameter': ['N', 'A', 'omega (rad/us)', 'phi_0 (rad)'],
    'Value': [fit_N, fit_A, fit_omega, fit_phi_0],
    'Error': [err_N, err_A, err_omega, err_phi_0]
}).to_csv("fit_results.csv", index=False)



# 画图
plt.figure(figsize=(12, 4))
plt.scatter(bin_centers, counts, s=0.25)
plot_time = np.linspace(TIME_MIN, TIME_MAX, NUM)
fit_curve = wiggle_fit_function(plot_time, *popt)
plt.plot(plot_time, fit_curve, 'r-', linewidth=0.5, label="Fit")
#plt.title(f'Toy MC: Muon g-2 Wiggle Plot (E > {THRESHOLD/1000:.1f} GeV)\nAnomalous Precession $\\omega_a$ = {OMEGA_A} rad/us', fontsize=14)
plt.xlabel(r'Time in Lab Frame ($\mu s$)', fontsize=12)
plt.ylabel('Number of High-Energy Positrons ($N_e$)', fontsize=12)
plt.yscale('log') # 可选对数坐标
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
#plt.savefig('plot/wiggle_line_fit_plot_with_fit.png', dpi=300)
plt.show()




# 画图
plt.figure(figsize=(12, 4))
plt.scatter(bin_centers, counts, s=0.25)
'''plot_time = np.linspace(0, TIME_MAX, NUM)
fit_curve = wiggle_fit_function(plot_time, *popt)
plt.plot(plot_time, fit_curve, 'r-', linewidth=0.5, label="Fit")
'''
#plt.title(f'Toy MC: Muon g-2 Wiggle Plot (E > {THRESHOLD/1000:.1f} GeV)\nAnomalous Precession $\\omega_a$ = {OMEGA_A} rad/us', fontsize=14)
plt.xlabel(r'Time in Lab Frame ($\mu s$)', fontsize=12)
plt.ylabel('Number of High-Energy Positrons ($N_e$)', fontsize=12)
plt.yscale('log') # 可选对数坐标
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
#plt.savefig('plot/wiggle_line_fit_plot.png', dpi=300)
plt.show()





plt.figure(figsize=(12, 4))
#plt.scatter(bin_centers, counts, s=0.25)


fit_curve = wiggle_fit_function(bin_centers, *popt)
res = (counts - fit_curve)   # 或 fit_curve - counts，符号只差正
mask = (bin_centers >= 10.1) & (bin_centers <= TIME_MAX)
res = res[mask]
bin = bin_centers[mask]
plt.scatter(bin, res, s=0.25)

#plt.title(f'Toy MC: Muon g-2 Wiggle Plot (E > {THRESHOLD/1000:.1f} GeV)\nAnomalous Precession $\\omega_a$ = {OMEGA_A} rad/us', fontsize=14)
plt.xlabel(r'Time in Lab Frame ($\mu s$)', fontsize=12)
plt.ylabel('Residual', fontsize=12)
#plt.yscale('log') # 可选对数坐标
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('plot/real_residual.png', dpi=300)
plt.show()




#生成时间除以100的余数为横坐标的wiggle plot
time_mod = np.mod(bin_centers, 100)
fit_at_bins = wiggle_fit_function(bin_centers, *popt)

plt.figure(figsize=(12, 10))
plt.scatter(time_mod, counts, s=0.25)
#plt.scatter(time_mod, fit_at_bins, s=0.25, color='r', label='Fit')
plt.xlabel("Time mod 100 ($\\mu s$)")
plt.ylabel(f"Counts (E > {THRESHOLD} MeV)")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plot/wiggle_mod100.png", dpi=300, bbox_inches='tight')
plt.show()


#生成时间除以100的余数为横坐标的wiggle plot
time_mod = np.mod(bin_centers, 100)
bin = bin_centers[mask]

fit_at_bins = wiggle_fit_function(bin, *popt)

plt.figure(figsize=(12, 10))
plt.scatter(time_mod, counts, s=0.25)
plt.scatter(bin%100, fit_at_bins, s=0.25, color='r', label='Fit')
plt.xlabel("Time mod 100 ($\\mu s$)")
plt.ylabel(f"Counts (E > {THRESHOLD} MeV)")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plot/wiggle_mod100_with_fit.png", dpi=300, bbox_inches='tight')
plt.show()




mask = (bin_centers >= TIME_MIN) & (bin_centers <= TIME_MAX)

x_fft = bin_centers[mask]
counts_fft = counts[mask]
res_fft = (counts_fft - wiggle_fit_function(x_fft, *popt))

N_bins = len(res_fft)
dt = np.mean(np.diff(x_fft))   # 比手写 0.141 更稳

freq = fftfreq(N_bins, d=dt)[:N_bins // 2]

window_raw = np.hanning(N_bins)
wiggle_raw_win = (counts_fft - np.mean(counts_fft)) * window_raw
fft_raw = np.abs(fft(wiggle_raw_win)[:N_bins // 2])

window_res = np.hanning(N_bins)
residual_win = (res_fft - np.mean(res_fft)) * window_res
fft_residual = np.abs(fft(residual_win)[:N_bins // 2])

# =========================
# 7. 找峰（只对 residual FFT）
# =========================
mask_valid = freq > 0.1
freq_valid = freq[mask_valid]
fft_valid = fft_residual[mask_valid]

peaks, props = find_peaks(fft_valid, prominence=10)
peak_heights = fft_valid[peaks]

# 避免找到的峰不足3个时报错
n_peaks = min(3, len(peaks))
if n_peaks > 0:
    top_idx = np.argsort(peak_heights)[-n_peaks:][::-1]
    
    print("\n=== Top Peaks (Residual FFT) ===")
    for i in top_idx:
        print(
            f"Frequency = {freq_valid[peaks[i]]:.4f} MHz, "
            f"Amplitude = {peak_heights[i]:.2f}"
        )
else:
    print("\n=== No prominent peaks found in Residual FFT ===")

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
    lw=0.8,
    alpha=0.6
)

plt.xlabel("Frequency (MHz)")
plt.ylabel("Amplitude")
#plt.title("FFT: Raw Signal vs Residuals")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.yscale("log")
plt.xlim(0.1, 3.0)

ymax = max(np.max(raw_plot[mask_valid]), np.max(res_plot[mask_valid]))
plt.ylim(10, ymax * 1.2)

plt.tight_layout()

# 如需保存图片，取消下方注释
# save_dir = r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship\real data\plot"
# os.makedirs(save_dir, exist_ok=True)
# save_path = os.path.join(save_dir, "fft_raw_and_residual_from_parquet.png")
plt.savefig("plot/real_fft.png", dpi=300, bbox_inches="tight")
# print(f"\nFFT plot saved to: {save_path}")

plt.show()
