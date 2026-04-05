import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # 用于生成结构化的“探测器数据”表
from scipy.optimize import curve_fit
from constants import *

file= "Data_NEW.csv"

try:
    data = np.genfromtxt(file, delimiter=",", names=True)
except FileNotFoundError:
    print(f"Error: {file} not found. Please make sure the file is in the correct directory.")
    exit()


useful_data = data[data['Energy_MeV'] > THRESHOLD] # 只保留能量大于阈值的事件
counts, bin_edges = np.histogram(useful_data['Time_us'], bins=5000, range=(0, TIME_MAX))

residuals = [bin_edges, counts]
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) # 计算 bin 的中心位置作为 x 轴坐标

hist_data = {'Time_us': bin_centers, 'Counts': counts}
pd.DataFrame(hist_data).to_csv("Counts.csv", index=False)

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
plt.figure(figsize=(15, 5))
plt.scatter(bin_centers, counts, s=1)
plot_time = np.linspace(0, TIME_MAX, 5000)
fit_curve = wiggle_fit_function(plot_time, *popt)
plt.plot(plot_time, fit_curve, 'r-', linewidth=0.5, label="Fit")
plt.title(f'Toy MC: Muon g-2 Wiggle Plot (E > {THRESHOLD/1000:.1f} GeV)\nAnomalous Precession $\\omega_a$ = {OMEGA_A} rad/us', fontsize=14)
plt.xlabel(r'Time in Lab Frame ($\mu s$)', fontsize=12)
plt.ylabel('Number of High-Energy Positrons ($N_e$)', fontsize=12)
plt.yscale('log') # 可选对数坐标
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('plot/wiggle_line_fit_plot_with_fit.png', dpi=300)
plt.show()




# 画图
plt.figure(figsize=(15, 5))
plt.scatter(bin_centers, counts, s=1)
'''plot_time = np.linspace(0, TIME_MAX, 5000)
fit_curve = wiggle_fit_function(plot_time, *popt)
plt.plot(plot_time, fit_curve, 'r-', linewidth=0.5, label="Fit")
'''
plt.title(f'Toy MC: Muon g-2 Wiggle Plot (E > {THRESHOLD/1000:.1f} GeV)\nAnomalous Precession $\\omega_a$ = {OMEGA_A} rad/us', fontsize=14)
plt.xlabel(r'Time in Lab Frame ($\mu s$)', fontsize=12)
plt.ylabel('Number of High-Energy Positrons ($N_e$)', fontsize=12)
plt.yscale('log') # 可选对数坐标
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig('plot/wiggle_line_fit_plot.png', dpi=300)
plt.show()



#生成时间除以100的余数为横坐标的wiggle plot
time_mod = bin_centers % 100
plt.figure(figsize=(12, 10))
plt.scatter(time_mod, counts, s=1)
#plt.scatter(plot_time % 100, fit_curve,s=0.25, color='r', label="Fit")
plt.xlabel("Time mod 100 ($\mu s$)")
plt.ylabel(f"Counts (E > {THRESHOLD} MeV)")
plt.title(f"Wiggle Plot (Time mod 100 $\mu s$)")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plot/wiggle_plot_time_mod_100.png", dpi=300, bbox_inches='tight')
plt.show()


#生成时间除以100的余数为横坐标的wiggle plot
time_mod = bin_centers % 100
plt.figure(figsize=(12, 10))
plt.scatter(time_mod, counts, s=1)
plt.scatter(plot_time % 100, fit_curve,s=0.25, color='r', label="Fit")
plt.xlabel("Time mod 100 ($\mu s$)")
plt.ylabel(f"Counts (E > {THRESHOLD} MeV)")
plt.title(f"Wiggle Plot (Time mod 100 $\mu s$)")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plot/wiggle_plot_time_mod_100_with_fit.png", dpi=300, bbox_inches='tight')
plt.show()
