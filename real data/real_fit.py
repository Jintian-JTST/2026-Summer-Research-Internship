# fit.py
# for real data from run6A.root, with omega_a as a free parameter in the fit
import os
import sys
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.append(r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship")
from constants import TAU_LAB, OMEGA_A

# --- 1. 读取 ROOT 文件数据 ---
try:
    file = uproot.open("D:\\Users\\JTST\\Desktop\\Desktop\\SI\\2026-Summer-Research-Internship\\real data\\run6A.root")
    hist = file["et_spectrum"]
    values = hist.values()
    time_edges = hist.axis(0).edges()
    energy_edges = hist.axis(1).edges()
except FileNotFoundError:
    print("Error: run6A.root not found. Please make sure the ROOT file is in the correct directory.")
    exit()

# --- 2. 设定能量阈值，提取数据 (根据你提供的代码) ---
threshold = 1700  # MeV

# 找到阈值对应的能量 bin
energy_bin = np.searchsorted(energy_edges, threshold, side="left")

# 假设第 0 轴是时间，第 1 轴是能量：
counts = values[:, energy_bin:].sum(axis=1)

# 时间 bin 中心
time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])

# --- 3. 选择拟合的时间范围 ---
# 我们使用这两个变量既做拟合范围，也做画图的 xlim
time_min = 30   # us, 避开早期的束流噪声
time_max = 600  # us, 避开后期统计量太低的区域

fit_mask = (time_centers >= time_min) & (time_centers <= time_max)
time_fit = time_centers[fit_mask]
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
    # 这里的 TAU_LAB 从 constants.py 导入，作为固定常数
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
plt.figure(figsize=(12, 4))
plt.scatter(time_centers, counts, s=0.5, label="Data")
plot_time = np.linspace(time_min, time_max, 10000)
fit_curve = wiggle_fit_function(plot_time, *popt)
plt.plot(plot_time, fit_curve, 'r-', linewidth=0.5, label="Fit")

# 应用相同的时间显示范围
plt.xlim(0, 660)

# 添加带单位的标签 (使用原始字符串 r"" 防止转义字符报错)
plt.xlabel(r"Time ($\mu s$)") 
plt.ylabel(f"Counts (E > {threshold} MeV)") 
#plt.title(f"Wiggle Plot and Fit (Zoomed: {time_min}-{time_max} $\mu s$)")
plt.yscale("log") 
plt.grid(True, which="both", ls="--", alpha=0.5)

# 在图表右上角添加拟合结果文本框
text_results = f"""Fit Parameters:
   $N = ${fit_N:.2e} $\pm$ {err_N:.1e}
   $A = ${fit_A:.3f} $\pm$ {err_A:.3f}
   $\omega = ${fit_omega:.5f} $\pm$ {err_omega:.5f} rad/$\mu$s
   $\phi_0 = ${fit_phi_0:.3f} $\pm$ {err_phi_0:.3f} rad"""
#plt.text(0.65, 0.95, text_results, transform=plt.gca().transAxes,
#         fontsize=12, verticalalignment='top', 
#         bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))

#plt.legend(fontsize=12)
plt.tight_layout()

# 保存图片
os.makedirs("D:\\Users\\JTST\\Desktop\\Desktop\\SI\\2026-Summer-Research-Internship\\real data\\plot", exist_ok=True)
plt.savefig("D:\\Users\\JTST\\Desktop\\Desktop\\SI\\2026-Summer-Research-Internship\\real data\\plot\\real_wiggle_plot_run6A_fit.png", dpi=300, bbox_inches='tight') 
print("Fit plot saved to plot/real_wiggle_plot_run6A_fit.png")

plt.show()