# mod100_only.py

import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =========================
# Config
# =========================
ROOT_FILE = r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship\real data\run6A.root"
OUT_DIR = r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship\real data\plot"

THRESHOLD = 1700  # MeV
TIME_MIN = 0.0
TIME_MAX = 660.0

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Load data
# =========================
with uproot.open(ROOT_FILE) as file:
    hist = file["et_spectrum"]
    values = hist.values()
    time_edges = hist.axis(0).edges()
    energy_edges = hist.axis(1).edges()

time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])

energy_bin = np.searchsorted(energy_edges, THRESHOLD, side="left")
counts = values[:, energy_bin:].sum(axis=1)

# =========================
# Fit model
# =========================
def wiggle_model(t, N0, tau, A, omega, phi):
    return N0 * np.exp(-t / tau) * (1 + A * np.cos(omega * t + phi))

mask = (
    (time_centers >= TIME_MIN) &
    (time_centers <= TIME_MAX) &
    (counts > 0)
)

t = time_centers[mask]
y = counts[mask]

sigma = np.sqrt(np.maximum(y, 1.0))

p0 = [
    y.max(),
    64.0,
    0.3,
    2.0 * np.pi * 0.23,
    0.0
]

popt, _ = curve_fit(
    wiggle_model,
    t, y,
    p0=p0,
    sigma=sigma,
    maxfev=5000
)

# 生成平滑拟合曲线
t_plot = np.linspace(TIME_MIN, TIME_MAX, 5000)
fit_plot = wiggle_model(t_plot, *popt)

# =========================
# mod 100 plot（唯一输出）
# =========================
plt.figure(figsize=(12, 10))

#plt.scatter(t_plot % 100, fit_plot, s=.5, color='red',label="Fit")
plt.scatter(t % 100, y, s=1, label="Data")

plt.xlabel("Time mod 100 ($\\mu s$)")
plt.ylabel(f"Counts (E > {THRESHOLD} MeV)")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()

plt.tight_layout()

save_path = os.path.join(OUT_DIR, "wiggle_mod100.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")

print(f"Saved to: {save_path}")
plt.show()

# ============================================================
# 6) Plot: wiggle + fit
# ============================================================
'''plt.figure(figsize=(12, 4))
plt.scatter(t, y, s=.8, label="Data")
plt.plot(t, fit, lw=1, color='red',label="Fit")
plt.xlim(TIME_MIN, TIME_MAX)
plt.xlabel("Time ($\\mu s$)")
plt.ylabel(f"Counts (E > {THRESHOLD} MeV)")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "wiggle_fit_run6A.png"), dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(12, 4))
plt.scatter(t, y, s=0.8, label="Data")
#plt.plot(t, fit, lw=2, label="Fit")
plt.xlim(TIME_MIN, TIME_MAX)
plt.xlabel("Time ($\\mu s$)")
plt.ylabel(f"Counts (E > {THRESHOLD} MeV)")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "wiggle_fit_run6A_with_fit.png"), dpi=300, bbox_inches="tight")
plt.show()
'''

# ============================================================
# 7) Plot: residual in time domain
# ============================================================
'''plt.figure(figsize=(12, 3))
plt.plot(t, residual, ".", ms=2)
plt.xlabel("Time ($\\mu s$)")
plt.ylabel("(data - fit) / fit")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "residual_time_run6A.png"), dpi=300, bbox_inches="tight")
plt.show()'''


# ============================================================
# 8) Plot: FFT of original wiggle + residual FFT together
# ============================================================
'''plt.figure(figsize=(12, 4))

#plt.plot(freq0, amp0, lw=1.0, label="Original wiggle FFT",color='black',alpha=0.5)
plt.plot(freq1, amp1, lw=1.0, label="Residual FFT")

plt.xlim(0, 3.0)
#plt.yscale("log")
plt.xlabel("Frequency (MHz)")
plt.ylabel("FFT amplitude")
plt.grid(True, which="both", ls="--", alpha=0.5)
'''
'''# Mark original peak
if np.isfinite(peak_freq0):
    plt.axvline(peak_freq0, color="C0", ls="--", lw=1.2)
    plt.scatter([peak_freq0], [peak_amp0], color="C0", zorder=5)
    plt.annotate(
        f"{peak_freq0:.3f} MHz",
        xy=(peak_freq0, peak_amp0),
        xytext=(peak_freq0 + 0.06, peak_amp0 * 1.4),
        arrowprops=dict(arrowstyle="->", lw=1, color="C0"),
        fontsize=9,
        color="C0"
    )
'''
'''# Mark residual peak
if np.isfinite(peak_freq1):
    plt.axvline(peak_freq1, color="C3", ls="--", lw=1.2)
    plt.scatter([peak_freq1], [peak_amp1], color="C3", zorder=5)
    plt.annotate(
        f"{peak_freq1:.3f} MHz",
        xy=(peak_freq1, peak_amp1),
        xytext=(peak_freq1 + 0.06, peak_amp1 * 1.4),
        arrowprops=dict(arrowstyle="->", lw=1, color="C3"),
        fontsize=9,
        color="C3"
    )'''
'''
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fft_original_and_residual_run6A.png"), dpi=300, bbox_inches="tight")
plt.show()
'''

# ============================================================
# 9) Optional: print top few peaks in each spectrum
'''# ============================================================
def print_top_peaks(freq_search, amp_search, label, n=5):
    if len(amp_search) == 0:
        print(f"\nNo data in {label} search band.")
        return

    prom = max(np.max(amp_search) * 0.02, 1e-12)
    peaks, _ = find_peaks(amp_search, prominence=prom)

    print(f"\n=== Top peaks: {label} ===")
    if len(peaks) == 0:
        idx = np.argmax(amp_search)
        print(f"1: f = {freq_search[idx]:.6f} MHz, amp = {amp_search[idx]:.6f}")
        return

    peak_freqs = freq_search[peaks]
    peak_amps = amp_search[peaks]
    order = np.argsort(peak_amps)[::-1]

    for i, idx in enumerate(order[:n], start=1):
        print(f"{i}: f = {peak_freqs[idx]:.6f} MHz, amp = {peak_amps[idx]:.6f}")

print_top_peaks(freq0_search, amp0_search, "original wiggle FFT")
print_top_peaks(freq1_search, amp1_search, "residual FFT")'''