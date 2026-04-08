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



with uproot.open(ROOT_FILE) as file:
    hist = file["et_spectrum"]
    values = hist.values()
    time_edges = hist.axis(0).edges()
    energy_edges = hist.axis(1).edges()
root_bin_centers = 0.5 * (time_edges[:-1] + time_edges[1:])

root_energy_bin = np.searchsorted(energy_edges, THRESHOLD, side="left")
root_counts = values[:, root_energy_bin:].sum(axis=1)


# 拟合函数
def wiggle_fit_function(t, N, A, omega, phi_0):
    """
    N: Normalization constant
    A: Asymmetry parameter
    omega: Anomalous precession frequency (自由参数)
    phi_0: Initial phase
    """
    return N * np.exp(-t / TAU_LAB) * (1 + A * np.cos(omega * t - phi_0))

decay_factor = np.exp(-root_bin_centers[0] / TAU_LAB)

# 初始参数猜测 [N, A, omega, phi_0]
initial_N = root_counts[0] / decay_factor
initial_A = 0.4  
initial_omega = OMEGA_A  # 使用常数作为拟合的初始起点帮助收敛
initial_phi_0 = 0.0 

root_popt, root_pcov = curve_fit(wiggle_fit_function,
                       root_bin_centers[root_counts>0],
                       root_counts[root_counts>0],
                       p0=[initial_N, initial_A, initial_omega, initial_phi_0],
                       sigma=np.sqrt(root_counts[root_counts>0]),
                       absolute_sigma=True,maxfev=10000)

root_fit_N, root_fit_A, root_fit_omega, root_fit_phi_0 = root_popt
root_perr = np.sqrt(np.diag(root_pcov))
root_err_N, root_err_A, root_err_omega, root_err_phi_0 = root_perr


root_fit_curve = wiggle_fit_function(root_bin_centers, *root_popt)
root_res = (root_counts - root_fit_curve)   # 或 fit_curve - counts，符号只差正
root_mask = (root_bin_centers >= 10.1) & (root_bin_centers <= TIME_MAX)
root_res = root_res[root_mask]
root_bin = root_bin_centers[root_mask]


print("--- Fit Results ---")
print(f"N     = {root_fit_N:.8f} ± {root_err_N:.8f}")
print(f"A     = {root_fit_A:.8f} ± {root_err_A:.8f}")
print(f"omega = {root_fit_omega:.8f} ± {root_err_omega:.8f} rad/us")
print(f"phi_0 = {root_fit_phi_0:.8f} ± {root_err_phi_0:.8f} rad")
print("--------------------")
pd.DataFrame({
    'Parameter': ['N', 'A', 'omega (rad/us)', 'phi_0 (rad)'],
    'Value': [root_fit_N, root_fit_A, root_fit_omega, root_fit_phi_0],
    'Error': [root_err_N, root_err_A, root_err_omega, root_err_phi_0]
})#.to_csv("fit_results.csv", index=False)





toy_path = r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship\Data.parquet"
try:
    data = pd.read_parquet(toy_path)
except FileNotFoundError:
    print(f"Error: {toy_path} not found.")
    raise SystemExit(1)

if "Time_us" not in data.columns or "Energy_MeV" not in data.columns:
    raise KeyError("Data.parquet must contain columns: 'Time_us' and 'Energy_MeV'")


useful_data = data[data['Energy_MeV'] > THRESHOLD] # 只保留能量大于阈值的事件
counts, bin_edges = np.histogram(useful_data['Time_us'], bins=NUM, range=(0, TIME_MAX))

residuals = [bin_edges, counts]
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) # 计算 bin 的中心位置作为 x 轴坐标

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

fit_curve = wiggle_fit_function(bin_centers, *popt)
res = (counts - fit_curve)   # 或 fit_curve - counts，符号只差正
mask = (bin_centers >=  TIME_MIN) & (bin_centers <= TIME_MAX)
res = res[mask]
bin = bin_centers[mask]

print("--- Fit Results ---")
print(f"N     = {fit_N:.8f} ± {err_N:.8f}")
print(f"A     = {fit_A:.8f} ± {err_A:.8f}")
print(f"omega = {fit_omega:.8f} ± {err_omega:.8f} rad/us")
print(f"phi_0 = {fit_phi_0:.8f} ± {err_phi_0:.8f} rad")
print("--------------------")
pd.DataFrame({
    'Parameter': ['N', 'A', 'omega (rad/us)', 'phi_0 (rad)'],
    'Value': [fit_N, fit_A, fit_omega, fit_phi_0],
    'Error': [err_N, err_A, err_omega, err_phi_0]
}).to_csv("fit_results.csv", index=False)




plt.figure(figsize=(12, 4))
plt.scatter(bin, res, s=0.25)
plt.scatter(root_bin, root_res, s=0.25)

#plt.title(f'Toy MC: Muon g-2 Wiggle Plot (E > {THRESHOLD/1000:.1f} GeV)\nAnomalous Precession $\\omega_a$ = {OMEGA_A} rad/us', fontsize=14)
plt.xlabel(r'Time in Lab Frame ($\mu s$)', fontsize=12)
plt.ylabel('Residual', fontsize=12)
#plt.yscale('log') # 可选对数坐标
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
#plt.savefig('plot/comp_residual.png', dpi=300)
plt.show()









root_mask = (root_bin_centers >= TIME_MIN) & (root_bin_centers <= TIME_MAX)

root_x_fft = root_bin_centers[root_mask]
root_counts_fft = root_counts[root_mask]
root_res_fft = (root_counts_fft - wiggle_fit_function(root_x_fft, *root_popt))

root_N_bins = len(root_res_fft)
root_dt = np.mean(np.diff(root_x_fft))   # 比手写 0.141 更稳

root_freq = fftfreq(root_N_bins, d=root_dt)[:root_N_bins // 2]


root_window_res = np.hanning(root_N_bins)
root_residual_win = (root_res_fft - np.mean(root_res_fft)) * root_window_res
root_fft_residual = np.abs(fft(root_residual_win)[:root_N_bins // 2])

# =========================
# 7. 找峰（只对 residual FFT）
# =========================
root_mask_valid = root_freq > 0.1











dt = 0.06
N_bins = len(res) 
counts =counts[mask]
# Calculate frequencies once for both FFTs
freq = fftfreq(N_bins, d=dt)[:N_bins // 2]

# --- Raw FFT ---
window_raw = np.hanning(N_bins)
wiggle_raw_win = (counts - np.mean(counts)) * window_raw
fft_raw = np.abs(fft(wiggle_raw_win)[:N_bins // 2])

# --- Residual FFT ---
window_res = np.hanning(N_bins)
residual_win = (res - np.mean(res)) * window_res 
fft_residual = np.abs(fft(residual_win)[:N_bins // 2])

mask_valid = freq > 0.1









plt.figure(figsize=(12, 4))

root_plot = np.clip(root_fft_residual, 1e-12, None)
toy_plot = np.clip(fft_residual, 1e-12, None)


plt.plot(
    freq[mask_valid],
    toy_plot[mask_valid],
    label="Toy Residual FFT",
    lw=0.8,
    alpha=0.6
)

plt.plot(
    root_freq[root_mask_valid],
    root_plot[root_mask_valid],
    label="ROOT Residual FFT",
    lw=0.8,
    alpha=0.6
)


plt.xlabel("Frequency (MHz)")
plt.ylabel("Amplitude")
plt.legend()
plt.yscale("log")
plt.xlim(0.1, 3.0)
plt.grid(True, which="both", ls="-", alpha=0.5)

ymax = max(
    np.max(root_plot[root_mask_valid]),
    np.max(toy_plot[mask_valid])
)
plt.ylim(10, ymax * 1.2)

plt.tight_layout()
#plt.savefig("plot/comp_fft.png", dpi=300, bbox_inches="tight")
plt.show()