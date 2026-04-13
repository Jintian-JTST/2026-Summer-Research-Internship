import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from constants import *
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import uproot

ROOT_FILE = "run2.root"
TIME_MIN = 25
TIME_MAX = 655
NUM=int((TIME_MAX-TIME_MIN)/TIME_WIN)

# 读取 ROOT 数据
with uproot.open(ROOT_FILE) as file:
    hist = file["ET_raw"]
    values = hist.values()
    time_edges = hist.axis(0).edges()/1000  # 转换为 μs
    energy_edges = hist.axis(1).edges()
print('Total events in ROOT histogram:', values.sum())
bin_centers = 0.5 * (time_edges[:-1] + time_edges[1:])

energy_bin = np.searchsorted(energy_edges, THRESHOLD, side="left")
counts = values[:, energy_bin:].sum(axis=1)
# 后面和 anaylsis.py 一样



# 拟合函数
def wiggle_fit_function(t, N, A, omega, phi_0, tau): # FIVE PARAMETERS FUNCTION
    return N * np.exp(-t / tau) * (1 + A * np.cos(omega * t + phi_0))

decay_factor = np.exp(-bin_centers[0] / TAU_LAB)

# 初始参数猜测
initial_N = counts[0] / decay_factor
initial_A = 0.4  
initial_omega = OMEGA_A
initial_phi_0 = 0.0 
initial_tau=TAU_LAB

popt, pcov = curve_fit(wiggle_fit_function,
                       bin_centers[counts>0],
                       counts[counts>0],
                       p0=[initial_N, initial_A, initial_omega, initial_phi_0,initial_tau],
                       sigma=np.sqrt(counts[counts>0]),
                       absolute_sigma=True,maxfev=10000)

fit_N, fit_A, fit_omega, fit_phi_0, fit_tau = popt
perr = np.sqrt(np.diag(pcov))
err_N, err_A, err_omega, err_phi_0, err_tau = perr


print("--- Fit Results ---")
print(f"N     = {fit_N:.8f} ± {err_N:.8f}")
print(f"A     = {fit_A:.8f} ± {err_A:.8f}")
print(f"omega = {fit_omega:.8f} ± {err_omega:.8f} rad/us")
print(f"phi_0 = {fit_phi_0:.8f} ± {err_phi_0:.8f} rad")
print(f"tau = {fit_tau:.8f} ± {err_tau:.8f} us")
print("--------------------")
pd.DataFrame({
    'Parameter': ['N', 'A', 'omega (rad/us)', 'phi_0 (rad)', 'tau (us)'],
    'Value': [fit_N, fit_A, fit_omega, fit_phi_0, fit_tau],
    'Error': [err_N, err_A, err_omega, err_phi_0, err_tau]
}).to_csv("real_fit_results.csv", index=False)




# PLOT START ==================================================================


# ORIGINAL ONLY
plt.figure(figsize=(12, 4))
plt.scatter(bin_centers, counts, s=0.25)
plt.xlabel(r'Time in Lab Frame ($\mu s$)', fontsize=12)
plt.ylabel('Number of High-Energy Positrons ($N_e$)', fontsize=12)
plt.yscale('log') # 可选对数坐标
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.xlim(TIME_MIN-10, TIME_MAX+10)
plt.savefig('plot/REAL.png', dpi=300, bbox_inches='tight')
plt.show()


# FIT + ORIGINAL 
plt.figure(figsize=(12, 4))
plt.scatter(bin_centers, counts, s=0.25)
plot_time = np.linspace(TIME_MIN, TIME_MAX, NUM)
fit_curve = wiggle_fit_function(plot_time, *popt)
plt.plot(plot_time, fit_curve, 'r-', linewidth=0.5, label="Fit")
plt.xlabel(r'Time in Lab Frame ($\mu s$)', fontsize=12)
plt.ylabel('Number of High-Energy Positrons ($N_e$)', fontsize=12)
plt.yscale('log') # 可选对数坐标
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.xlim(TIME_MIN-10, TIME_MAX+10)
plt.savefig('plot/REAL_FIT.png', dpi=300, bbox_inches='tight')
plt.show()


# WIGGLE PLOT (NO FIT)
time_mod = bin_centers % 100
plt.figure(figsize=(12, 10))
plt.scatter(time_mod, counts, s=0.25)
plt.xlabel("Time mod 100 ($\mu s$)")
plt.ylabel(f"Counts (E > {THRESHOLD} MeV)")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.ylim(800, counts.max()*1.2)
plt.tight_layout()
plt.savefig("plot/REAL_WIGGLE.png", dpi=300, bbox_inches='tight')
plt.show()


# WIGGLE PLOT (WITH FIT)
time_mod = bin_centers % 100
plt.figure(figsize=(12, 10))       
plt.scatter(time_mod, counts, s=0.25)
plt.scatter(plot_time % 100, fit_curve,s=0.25, color='r', label="Fit")
plt.xlabel("Time mod 100 ($\mu s$)")
plt.ylabel(f"Counts (E > {THRESHOLD} MeV)")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.ylim(800, counts.max()*1.2)
plt.tight_layout()
plt.savefig("plot/REAL_WIGGLE_FIT.png", dpi=300, bbox_inches='tight')
plt.show()


# RESIDUAL
plt.figure(figsize=(12, 4))
fit_curve = wiggle_fit_function(bin_centers, *popt)
res = (counts - fit_curve) 
mask = (bin_centers >= TIME_MIN + 0.1) & (bin_centers <= TIME_MAX)
res = res[mask]
bin = bin_centers[mask]
plt.scatter(bin, res, s=0.25)
plt.xlabel(r'Time in Lab Frame ($\mu s$)', fontsize=12)
plt.ylabel('Residual', fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.xlim(TIME_MIN-10, TIME_MAX+10)
plt.savefig('plot/REAL_RES.png', dpi=300, bbox_inches='tight')
plt.show()
# 保存用于后续 FFT 的 masked residual
pd.DataFrame({
    "Time_us": bin,
    "Residual": res,
}).to_csv("real_residuals.csv", index=False)


# =========================
# FFT
# =========================
# 统一在同一个时间区间上做 FFT
x_fft = bin_centers[mask]
counts_fft = counts[mask]
fit_fft = wiggle_fit_function(x_fft, *popt)
res = counts_fft - fit_fft

dt = np.mean(np.diff(x_fft))
n = len(x_fft)

# 去均值 + 加窗
window = np.hanning(n)
raw_win = (counts_fft - np.mean(counts_fft)) * window
res_win = (res - np.mean(res)) * window

# 用 rfft / rfftfreq 更合适，只保留正频率
freq = np.fft.rfftfreq(n, d=dt)
fft_raw = np.abs(np.fft.rfft(raw_win))
fft_residual = np.abs(np.fft.rfft(res_win))

# =========================
# 找峰 (RES)
# =========================
mask_valid = freq > 0.1
freq_valid = freq[mask_valid]
fft_valid = fft_residual[mask_valid]

peaks, props = find_peaks(fft_valid, prominence=10)
peak_heights = fft_valid[peaks]

n_peaks = min(4, len(peaks))
if n_peaks > 0:
    top_idx = np.argsort(peak_heights)[-n_peaks:][::-1]
    print("\n=== Top Peaks (Residual FFT) ===")
    for i in top_idx:
        print(f"Frequency = {freq_valid[peaks[i]]:.4f} MHz, "f"Amplitude = {peak_heights[i]:.2f}")
else:
    print("\n=== No prominent peaks found in Residual FFT ===")

# =========================
# 画图：raw FFT + residual FFT 同图
# =========================
plt.figure(figsize=(12, 4))
raw_plot = np.clip(fft_raw, 1e-12, None)
res_plot = np.clip(fft_residual, 1e-12, None)
plt.plot(freq_valid, raw_plot[mask_valid], label="Raw Signal FFT", lw=0.8, alpha=0.6)
plt.plot(freq_valid, res_plot[mask_valid], label="Residual FFT", lw=0.8, alpha=0.6)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Amplitude")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.yscale("log")
plt.xlim(0.1, 3.0)
ymax = max(np.max(raw_plot[mask_valid]), np.max(res_plot[mask_valid]))
ymin = min(np.min(raw_plot[mask_valid]), np.min(res_plot[mask_valid]))
plt.ylim(ymin * 0.8, ymax * 1.2)
plt.tight_layout()
plt.savefig("plot/REAL_FFT.png", dpi=300, bbox_inches="tight")
plt.show()
