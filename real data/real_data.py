# data_fft.py
# Read run6A.root, build wiggle plot, fit main oscillation,
# compute residuals, run FFT for both original wiggle and residual,
# auto-find peaks, and mark them on the plots.

import os
import uproot
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.signal import find_peaks


# ============================================================
# 0) Config
# ============================================================
ROOT_FILE = r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship\real data\run6A.root"
OUT_DIR = r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship\real data\plot"

THRESHOLD = 1700  # MeV
TIME_MIN = 0.0
TIME_MAX = 660.0

# FFT search band (MHz)
FFT_FMIN = 0.05
FFT_FMAX = 3.0

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1) Load histogram
# ============================================================
file = uproot.open(ROOT_FILE)
hist = file["et_spectrum"]

values = hist.values()
time_edges = hist.axis(0).edges()
energy_edges = hist.axis(1).edges()

n_time_bins = len(time_edges) - 1
print("Number of time bins:", n_time_bins)
print("Number of total positrons:", values.sum())

time_centers = 0.5 * (time_edges[:-1] + time_edges[1:])

energy_bin = np.searchsorted(energy_edges, THRESHOLD, side="left")
counts = values[:, energy_bin:].sum(axis=1)

print("Number of positrons above threshold:", values[:, energy_bin:].sum())

dt = time_centers[1] - time_centers[0]
print(f"dt = {dt:.6f} us")
print(f"Nyquist frequency = {1 / (2 * dt):.6f} MHz")


# ============================================================
# 2) Main wiggle fit model
# ============================================================
# N(t) = N0 * exp(-t/tau) * (1 + A*cos(omega*t + phi))
def wiggle_model(t, N0, tau, A, omega, phi):
    return N0 * np.exp(-t / tau) * (1.0 + A * np.cos(omega * t + phi))


# Select fit window and valid points
mask = (
    (time_centers >= TIME_MIN) &
    (time_centers <= TIME_MAX) &
    (counts > 0)
)

t = time_centers[mask].astype(float)
y = counts[mask].astype(float)

# Poisson-like weights
sigma = np.sqrt(np.maximum(y, 1.0))

# Initial guesses
p0 = [
    y.max(),             # N0
    64.0,                # tau (us)
    0.3,                 # A
    2.0 * np.pi * 0.23,  # omega (rad/us)
    0.0                  # phi
]

# Conservative bounds
bounds = (
    [0.0,   1.0,  0.0, 0.0,      -2*np.pi],
    [np.inf, 500.0, 1.0, 10.0,     2*np.pi]
)

popt, pcov = curve_fit(
    wiggle_model,
    t, y,
    p0=p0,
    sigma=sigma,
    absolute_sigma=False,
    bounds=bounds,
    maxfev=50000
)

fit = wiggle_model(t, *popt)

N0_fit, tau_fit, A_fit, omega_fit, phi_fit = popt
fa_fit = omega_fit / (2.0 * np.pi)

print("\n=== Fit parameters ===")
print(f"N0    = {N0_fit:.6e}")
print(f"tau   = {tau_fit:.6f} us")
print(f"A     = {A_fit:.6f}")
print(f"omega = {omega_fit:.6f} rad/us")
print(f"phi   = {phi_fit:.6f} rad")
print(f"fa    = {fa_fit:.6f} MHz")


# ============================================================
# 3) Helper: FFT + peak finding
# ============================================================
def fft_with_peaks(signal, dt_us, fmin=0.05, fmax=3.0, prominence_frac=0.02):
    """
    signal: 1D array
    dt_us: sampling interval in microseconds
    returns: freq, amp, peak_freq, peak_amp, freq_search, amp_search, peaks
    """
    sig = np.asarray(signal, dtype=float)
    sig = sig - np.mean(sig)
    window = np.hanning(len(sig))
    sig_win = sig * window

    Y = np.fft.rfft(sig_win)
    freq = np.fft.rfftfreq(len(sig_win), d=dt_us)  # MHz because dt is us
    amp = np.abs(Y)

    search_mask = (freq >= fmin) & (freq <= fmax)
    freq_search = freq[search_mask]
    amp_search = amp[search_mask]

    if len(amp_search) == 0:
        return freq, amp, np.nan, np.nan, freq_search, amp_search, np.array([])

    prom = max(np.max(amp_search) * prominence_frac, 1e-12)
    peaks, props = find_peaks(amp_search, prominence=prom)

    if len(peaks) > 0:
        best_local = peaks[np.argmax(amp_search[peaks])]
        peak_freq = freq_search[best_local]
        peak_amp = amp_search[best_local]
    else:
        idx = np.argmax(amp_search)
        peak_freq = freq_search[idx]
        peak_amp = amp_search[idx]

    return freq, amp, peak_freq, peak_amp, freq_search, amp_search, peaks


# ============================================================
# 4) FFT of the original wiggle
#    去掉指数趋势后再 FFT，避免 DC / envelope 主导
# ============================================================
coef = np.polyfit(t, np.log(np.maximum(y, 1.0)), 1)
trend = np.exp(np.polyval(coef, t))
y_det = y / trend

freq0, amp0, peak_freq0, peak_amp0, freq0_search, amp0_search, peaks0 = fft_with_peaks(
    y_det, dt, fmin=FFT_FMIN, fmax=FFT_FMAX, prominence_frac=0.02
)

print("\n=== Original wiggle FFT peak ===")
print(f"Peak frequency = {peak_freq0:.6f} MHz")
print(f"Peak amplitude = {peak_amp0:.6f}")


# ============================================================
# 5) Residuals
# ============================================================
# Relative residual; keep sign
residual = (y - fit) / fit
residual = residual - np.mean(residual)

freq1, amp1, peak_freq1, peak_amp1, freq1_search, amp1_search, peaks1 = fft_with_peaks(
    residual, dt, fmin=FFT_FMIN, fmax=FFT_FMAX, prominence_frac=0.02
)

print("\n=== Residual FFT peak ===")
print(f"Peak frequency = {peak_freq1:.6f} MHz")
print(f"Peak amplitude = {peak_amp1:.6f}")


# ============================================================
# 6) Plot: wiggle + fit
# ============================================================
plt.figure(figsize=(12, 6))
plt.scatter(t, y, s=2, label="Data")
plt.plot(t, fit, lw=2, label="Fit")
plt.xlim(TIME_MIN, TIME_MAX)
plt.xlabel("Time ($\\mu s$)")
plt.ylabel(f"Counts (E > {THRESHOLD} MeV)")
plt.yscale("log")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "wiggle_fit_run6A.png"), dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 7) Plot: residual in time domain
# ============================================================
plt.figure(figsize=(12, 5))
plt.plot(t, residual, ".", ms=2)
plt.xlabel("Time ($\\mu s$)")
plt.ylabel("(data - fit) / fit")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "residual_time_run6A.png"), dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 8) Plot: FFT of original wiggle + residual FFT together
# ============================================================
plt.figure(figsize=(12, 4))

#plt.plot(freq0, amp0, lw=1.0, label="Original wiggle FFT",color='black',alpha=0.5)
plt.plot(freq1, amp1, lw=1.0, label="Residual FFT")

plt.xlim(0, 3.0)
#plt.yscale("log")
plt.xlabel("Frequency (MHz)")
plt.ylabel("FFT amplitude")
plt.grid(True, which="both", ls="--", alpha=0.5)

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

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fft_original_and_residual_run6A.png"), dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 9) Optional: print top few peaks in each spectrum
# ============================================================
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
print_top_peaks(freq1_search, amp1_search, "residual FFT")