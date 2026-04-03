import numpy as np
import matplotlib.pyplot as plt

# 参数
N = 50000
t=2.2e-6  # muon lifetime in seconds
v=0.999 # muon velocity (close to speed of light)
gamma = 1/np.sqrt(1-v**2)  # Lorentz factor
tau = t * gamma  # dilated lifetime
    
omega = 2*np.pi*0.23
A = 0.4
phi = 0 * np.pi / 180  # convert degrees to radians

times = []

for _ in range(N):
    t = -tau * np.log(np.random.rand())

    weight = 1 + A * np.cos(omega * t + phi)

    if np.random.rand() < weight/2:
        E = np.random.rand()

        if E > 0.6:
            times.append(t)

plt.hist(times, bins=100)
plt.xlabel("time")
plt.ylabel("counts")
plt.title("Toy MC Wiggle Plot")
plt.show()