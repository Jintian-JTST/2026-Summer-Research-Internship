# plot.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from constants import *

# 确保图片保存目录存在
os.makedirs('plot', exist_ok=True)

detector_data = pd.read_csv('simulated_detector_data.csv')

high_energy_data = detector_data[
    (detector_data['E_MeV'] > THRESHOLD) &
    (detector_data['Time_us'] < 100)
]

# ==========================================
# 1. 绘制并保存直方图 (Histogram Wiggle Plot)
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

counts, bins, _ = ax.hist(high_energy_data['Time_us'], bins=1000, range=(0, 2*TAU_LAB),
                          color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)

# 保存数据为 CSV
result = pd.DataFrame({
    'Time_us': bins[:-1], # 取 bin 的左边缘
    'Counts': counts
})
save_path = "wiggle_plot_data.csv"
result.to_csv(save_path, index=False)

ax.set_title(f'Toy MC: Muon g-2 Wiggle Plot (E > {THRESHOLD/1000:.1f} GeV)\nAnomalous Precession $\\omega_a$ = {OMEGA_A} rad/us', fontsize=14)
ax.set_xlabel(r'Time in Lab Frame ($\mu s$)', fontsize=12) # 注意单位应该是微秒(us)
ax.set_ylabel('Number of High-Energy Positrons ($N_e$)', fontsize=12)
ax.set_yscale('log') # 真实实验中通常也会使用对数坐标，方便拟合
ax.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig('plot/wiggle_plot.png', dpi=300)
print("\nHistogram plot saved as plot/wiggle_plot.png")
plt.show()


# ==========================================
# 2. 绘制并保存散点图 (Scatter Wiggle Plot)
# ==========================================
# 计算 bin 的中心位置，用于画散点图会更精确
bin_centers = 0.5 * (bins[:-1] + bins[1:])

plt.figure(figsize=(10, 6))
# [修复] 这里的 x 改成了 bin_centers，与 counts 大小完全一致 (都是 400)
plt.scatter(bin_centers, counts, s=10, alpha=0.7, color='red') 

plt.title(f'Toy MC: Muon g-2 Wiggle Plot (E > {THRESHOLD/1000:.1f} GeV)\nAnomalous Precession $\\omega_a$ = {OMEGA_A} rad/us', fontsize=14)
plt.xlabel(r'Time in Lab Frame ($\mu s$)', fontsize=12)
plt.ylabel('Number of High-Energy Positrons ($N_e$)', fontsize=12)
plt.yscale('log') # 可选对数坐标
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

# [修复] 补充保存图片的命令
plt.savefig('plot/wiggle_scatter_plot.png', dpi=300)
print("Scatter plot saved as plot/wiggle_scatter_plot.png")
plt.show()