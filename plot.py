# plot.py
import matplotlib.pyplot as plt
import pandas as pd

from generate_data import TAU_LAB, OMEGA_A

detector_data = pd.read_csv('simulated_detector_data.csv')

THRESHOLD = 1800.0 # 设定能量阈值 1.8 GeV
high_energy_data = detector_data[detector_data['E_MeV'] > THRESHOLD]

fig, ax = plt.subplots(figsize=(10, 6))

# 画出时间直方图 (Wiggle Plot)
counts, bins, _ = ax.hist(high_energy_data['Time_us'], bins=400, range=(0, 2*TAU_LAB),
                          color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)

ax.set_title(f'Toy MC: Muon g-2 Wiggle Plot (E > {THRESHOLD/1000:.1f} GeV)\nAnomalous Precession $\\omega_a$ = {OMEGA_A} rad/us', fontsize=14)
ax.set_xlabel('Time in Lab Frame ($s$)', fontsize=12)
ax.set_ylabel('Number of High-Energy Positrons ($N_e$)', fontsize=12)
#ax.set_yscale('log') # 真实实验中通常也会使用对数坐标，方便拟合
ax.grid(True, which="both", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig('plot/wiggle_plot.png', dpi=300)
print("\nPlot saved as plot/wiggle_plot.png")
plt.show()