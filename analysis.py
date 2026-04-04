import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # 用于生成结构化的“探测器数据”表
from scipy.optimize import curve_fit
from constants import *

file= "simulated_detector_data.csv"

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

'''plt.scatter(bin_centers, counts, s=1) 
plt.title(f'Toy MC: Muon g-2 Wiggle Plot (E > {THRESHOLD/1000:.1f} GeV)\nAnomalous Precession $\\omega_a$ = {OMEGA_A} rad/us', fontsize=14)
plt.xlabel(r'Time in Lab Frame ($\mu s$)', fontsize=12)
plt.ylabel('Number of High-Energy Positrons ($N_e$)', fontsize=12)
plt.yscale('log') # 可选对数坐标
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()
'''



#生成时间除以100的余数为横坐标的wiggle plot
time_mod = bin_centers % 100
plt.figure(figsize=(12, 10))
plt.scatter(time_mod, counts, s=1)
plt.xlabel("Time mod 100 ($\mu s$)")
plt.ylabel(f"Counts (E > {THRESHOLD} MeV)")
plt.title(f"Wiggle Plot (Time mod 100 $\mu s$)")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()