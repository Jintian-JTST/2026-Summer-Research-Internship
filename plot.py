# plot.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from constants import *

# 确保图片保存目录存在
os.makedirs('plot', exist_ok=True)

#detector_data = pd.read_feather('simulated_detector_data.feather')

detector_data = pd.read_csv('simulated_detector_data.csv')

high_energy_data = detector_data[
    (detector_data['Energy_MeV'] > THRESHOLD) &
    (detector_data['Time_us'] < TIME_MAX) # 只考虑时间范围内的数据，避免后期统计量过低的区域
]
time_min = 0   # us, 避开早期的束流噪声
time_max = TIME_MAX  # us, 避开后期统计量太低的区域
bins = np.linspace(0, TIME_MAX, 5001) # 400 bins from 0 to TIME_MAX us
counts, _ = np.histogram(high_energy_data['Time_us'], bins=bins)
# 保存数据为 CSV
result = pd.DataFrame({
    'Time_us': bins[:-1], # 取 bin 的左边缘
    'Counts': counts
})
save_path = "wiggle_plot_data.csv"
result.to_csv(save_path, index=False)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

plt.figure(figsize=(10, 6))
# [修复] 这里的 x 改成了 bin_centers，与 counts 大小完全一致 (都是 400)
plt.scatter(bin_centers, counts, s=1) 
plt.title(f'Toy MC: Muon g-2 Wiggle Plot (E > {THRESHOLD/1000:.1f} GeV)\nAnomalous Precession $\\omega_a$ = {OMEGA_A} rad/us', fontsize=14)
plt.xlabel(r'Time in Lab Frame ($\mu s$)', fontsize=12)
plt.ylabel('Number of High-Energy Positrons ($N_e$)', fontsize=12)
plt.yscale('log') # 可选对数坐标
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()

# [修复] 补充保存图片的命令
#plt.savefig('plot/wiggle_line_plot.png', dpi=300)
print("Line plot saved as plot/wiggle_line_plot.png")
plt.show()




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
#plt.savefig("plot/wiggle_plot_time_mod_100.png", dpi=300, bbox_inches='tight')
plt.show()
