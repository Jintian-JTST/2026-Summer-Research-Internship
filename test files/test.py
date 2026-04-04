import numpy as np
import matplotlib.pyplot as plt

TAU_LAB = 2.2e-6 # 2.2 microseconds

t_lab = np.random.exponential(scale=TAU_LAB, size=200000)

plt.figure(figsize=(10, 6))
plt.hist(t_lab, bins=100, range=(0, 2*TAU_LAB), color='skyblue', edgecolor='black')
plt.xlim(0, 2*TAU_LAB)
plt.title('Exponential Distribution of Muon Decay Times in Lab Frame', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Counts', fontsize=12)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()
