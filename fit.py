# fit.py
import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from constants import TAU_LAB, OMEGA_A, THRESHOLD, TIME_MAX

# --- 1. 读取数据 ---
try:
    file = open("wiggle_plot_data.csv", "r")
    data = np.genfromtxt(file, delimiter=",", names=True)
except FileNotFoundError:
    print("Error: wiggle_plot_data.csv not found. Please make sure the file is in the correct directory.")
    exit()

counts = data['Counts']

# --- 3. 选择拟合的时间范围 ---
# 我们使用这两个变量既做拟合范围，也做画图的 xlim
time_min = 0   # us, 避开早期的束流噪声
time_max = TIME_MAX  # us, 避开后期统计量太低的区域
time_centers = data['Time_us'] # 直接使用 CSV 中的时间数据，确保与 counts 完全对应
fit_mask = (data['Time_us'] >= time_min) & (data['Time_us'] <= time_max)
time_fit = data['Time_us'][fit_mask]
counts_fit = counts[fit_mask]

# 计算误差（泊松误差为 sqrt(N)，如果 counts 为 0 设为 1 防止除零报错）
counts_err = np.sqrt(counts_fit)
counts_err[counts_err == 0] = 1 

# --- 4. 定义包含 omega 作为自由参数的拟合函数 ---
def wiggle_fit_function(t, N, A, omega, phi_0):
    """
    N: Normalization constant
    A: Asymmetry parameter
    omega: Anomalous precession frequency (自由参数)
    phi_0: Initial phase
    """
    return N * np.exp(-t / TAU_LAB) * (1 + A * np.cos(omega * t - phi_0))

# --- 5. 执行拟合 ---
decay_factor = np.exp(-time_fit[0] / TAU_LAB)

# 初始参数猜测 [N, A, omega, phi_0]
initial_N = counts_fit[0] / decay_factor
initial_A = 0.4  
initial_omega = OMEGA_A  # 使用常数作为拟合的初始起点帮助收敛
initial_phi_0 = 0.0 
p0 = [initial_N, initial_A, initial_omega, initial_phi_0]

print("Starting fit...")
try:
    popt, pcov = curve_fit(
        wiggle_fit_function,
        time_fit,
        counts_fit,
        p0=p0,
        sigma=counts_err,
        absolute_sigma=True, 
        maxfev=10000 
    )
except RuntimeError as e:
    print(f"Fit failed to converge: {e}")
    exit()


# 提取拟合参数和误差
fit_N, fit_A, fit_omega, fit_phi_0 = popt
perr = np.sqrt(np.diag(pcov))
err_N, err_A, err_omega, err_phi_0 = perr

print("--- Fit Results ---")
print(f"N     = {fit_N:.3e} ± {err_N:.3e}")
print(f"A     = {fit_A:.4f} ± {err_A:.4f}")
print(f"omega = {fit_omega:.6f} ± {err_omega:.6f} rad/us")
print(f"phi_0 = {fit_phi_0:.4f} ± {err_phi_0:.4f} rad")
print("--------------------")


# --- 6. 画 wiggle plot (根据你提供的代码风格) ---
plt.figure(figsize=(12, 6))
plt.plot(time_centers, counts, lw=1, color='blue', label="Data")
plot_time = np.linspace(time_min, time_max, 1000)
fit_curve = wiggle_fit_function(plot_time, *popt)
plt.plot(plot_time, fit_curve, 'r-', linewidth=2, label="Fit")

# 应用相同的时间显示范围
plt.xlim(time_min, time_max)

# 添加带单位的标签 (使用原始字符串 r"" 防止转义字符报错)
plt.xlabel(r"Time ($\mu s$)") 
plt.ylabel(f"Counts (E > {THRESHOLD} MeV)") 
plt.title(f"Wiggle Plot and Fit (Zoomed: {time_min}-{time_max} $\mu s$)")
plt.yscale("log") 
plt.grid(True, which="both", ls="--", alpha=0.5)

# 在图表右上角添加拟合结果文本框
text_results = f"""Fit Parameters:
   $N = ${fit_N:.2e} $\pm$ {err_N:.1e}
   $A = ${fit_A:.3f} $\pm$ {err_A:.3f}
   $\omega = ${fit_omega:.5f} $\pm$ {err_omega:.5f} rad/$\mu$s
   $\phi_0 = ${fit_phi_0:.3f} $\pm$ {err_phi_0:.3f} rad"""
plt.text(0.65, 0.95, text_results, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', 
         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))

plt.legend(fontsize=12)
plt.tight_layout()

# 保存图片
os.makedirs("plot", exist_ok=True)
plt.savefig("plot/wiggle_plot_run6A_fit.png", dpi=300, bbox_inches='tight') 
print("Fit plot saved to plot/wiggle_plot_run6A_fit.png")

plt.show()