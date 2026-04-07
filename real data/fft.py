# compare_real_toy_residual_fft.py
# Real data is read from ROOT, toy data is read from CSV.
# Each dataset is processed independently:
#   1) load spectrum
#   2) fit wiggle model
#   3) compute residual = (data - fit) / fit
#   4) FFT the residual
# Finally, plot both residual FFTs on the same figure.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import uproot


# ============================================================
# 0) Config
# ============================================================
REAL_ROOT_FILE = r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship\run6A.root"
TOY_CSV_FILE   = r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship\Counts.csv"

OUT_DIR = r"D:\Users\JTST\Desktop\Desktop\SI\2026-Summer-Research-Internship\plot"

TIME_COL = "Time_us"
COUNT_COL = "Counts"

TIME_MIN = 0.0
TIME_MAX = 660.0

# FFT band in MHz
FFT_FMIN = 0.05
FFT_FMAX = 3.0

# If your ROOT file is a 2D energy-time histogram, set the threshold here.
# Use the same threshold as your toy analysis.
# If your ROOT file is already a 1D time spectrum, this value is ignored.
ENERGY_THRESHOLD = None   # e.g. 1.8

# If your ROOT histogram axes are known, leave these as-is.
# The code assumes axis 0 = time, axis 1 = energy for TH2 histograms.
ROOT_TIME_AXIS = 0
ROOT_ENERGY_AXIS = 1

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1) Model
# ============================================================
# N(t) = N0 * exp(-t/tau) * (1 + A*cos(omega*t + phi))
def wiggle_model(t, N0, tau, A, omega, phi):
    return N0 * np.exp(-t / tau) * (1.0 + A * np.cos(omega * t + phi))


# ============================================================
# 2) FFT helper
# ============================================================
def fft_with_peaks(signal, dt_us, fmin=0.05, fmax=3.0, prominence_frac=0.02):
    sig = np.asarray(signal, dtype=float)
    sig = sig - np.mean(sig)

    window = np.hanning(len(sig))
    sig_win = sig * window

    Y = np.fft.rfft(sig_win)
    freq = np.fft.rfftfreq(len(sig_win), d=dt_us)   # MHz because dt_us is in us

    # 归一化 FFT 幅度：除以窗函数和，避免幅度随点数/窗函数改变
    amp = (2.0 * np.abs(Y)) / np.sum(window)

    # DC 和 Nyquist 不应该乘 2
    amp[0] *= 0.5
    if len(amp) > 1 and len(sig_win) % 2 == 0:
        amp[-1] *= 0.5

    search_mask = (freq >= fmin) & (freq <= fmax)
    freq_search = freq[search_mask]
    amp_search = amp[search_mask]

    if len(amp_search) == 0:
        return freq, amp, np.nan, np.nan

    prom = max(np.max(amp_search) * prominence_frac, 1e-12)
    peaks, _ = find_peaks(amp_search, prominence=prom)

    if len(peaks) > 0:
        best_local = peaks[np.argmax(amp_search[peaks])]
        peak_freq = freq_search[best_local]
        peak_amp = amp_search[best_local]
    else:
        idx = np.argmax(amp_search)
        peak_freq = freq_search[idx]
        peak_amp = amp_search[idx]

    return freq, amp, peak_freq, peak_amp

# ============================================================
# 3) ROOT loader
# ============================================================
def load_root_spectrum(root_file, energy_threshold=None):
    """
    Load a spectrum from ROOT.

    Supports:
      - TH1: directly returns bin centers and counts
      - TH2: assumes axis 0 = time, axis 1 = energy, and projects to time
             using an optional energy threshold

    Returns:
      t_centers, counts, source_name
    """
    f = uproot.open(root_file)

    hist_name = None
    hist = None

    # Find first TH1/TH2-like object if not known in advance.
    for k in f.keys():
        obj = f[k]
        cls = getattr(obj, "classname", "")
        if cls.startswith("TH1") or cls.startswith("TH2"):
            hist_name = k
            hist = obj
            break

    if hist is None:
        raise ValueError(f"No TH1/TH2 histogram found in ROOT file: {root_file}")

    cls = hist.classname

    # TH1: already a 1D spectrum
    if cls.startswith("TH1"):
        edges = hist.axis(0).edges()
        values = hist.values(flow=False)

        t_centers = 0.5 * (edges[:-1] + edges[1:])
        counts = np.asarray(values, dtype=float)

        return t_centers, counts, hist_name

    # TH2: project to time axis
    if cls.startswith("TH2"):
        values = np.asarray(hist.values(flow=False), dtype=float)
        axis0 = hist.axis(ROOT_TIME_AXIS)
        axis1 = hist.axis(ROOT_ENERGY_AXIS)

        edges0 = axis0.edges()
        edges1 = axis1.edges()

        t_centers_0 = 0.5 * (edges0[:-1] + edges0[1:])
        t_centers_1 = 0.5 * (edges1[:-1] + edges1[1:])

        # assume axis 0 is time and axis 1 is energy
        # if the histogram is stored the other way around, swap ROOT_TIME_AXIS/ROOT_ENERGY_AXIS
        if energy_threshold is None:
            counts = values.sum(axis=1)
        else:
            energy_centers = t_centers_1
            mask_e = energy_centers >= energy_threshold
            counts = values[:, mask_e].sum(axis=1)

        return t_centers_0, counts, hist_name

    raise ValueError(f"Unsupported ROOT histogram type: {cls}")


# ============================================================
# 4) CSV loader
# ============================================================
def load_csv_spectrum(csv_file):
    df = pd.read_csv(csv_file)

    if TIME_COL not in df.columns or COUNT_COL not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{TIME_COL}' and '{COUNT_COL}'. "
            f"Found: {list(df.columns)}"
        )

    t = df[TIME_COL].to_numpy(dtype=float)
    y = df[COUNT_COL].to_numpy(dtype=float)

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    return t, y


# ============================================================
# 5) Single-dataset processing
# ============================================================
def process_dataset(t_all, y_all, label):
    t_all = np.asarray(t_all, dtype=float)
    y_all = np.asarray(y_all, dtype=float)

    mask = np.isfinite(t_all) & np.isfinite(y_all) & (y_all > 0)

    if TIME_MIN is not None:
        mask &= (t_all >= TIME_MIN)
    if TIME_MAX is not None:
        mask &= (t_all <= TIME_MAX)

    t = t_all[mask]
    y = y_all[mask]

    if len(t) < 4:
        raise ValueError(f"{label}: not enough valid data points after masking.")

    dt = np.median(np.diff(t))
    nyquist = 1.0 / (2.0 * dt)

    print(f"\n===== {label} =====")
    print(f"Number of time bins: {len(t)}")
    print(f"Number of total positrons: {np.sum(y_all):.6e}")
    print(f"Number of positrons above threshold: {np.sum(y):.7e}")
    print(f"dt = {dt:.6f} us")
    print(f"Nyquist frequency = {nyquist:.6f} MHz")

    sigma = np.sqrt(np.maximum(y, 1.0))

    # Initial guesses
    p0 = [
        y.max(),             # N0
        64.0,                # tau (us)
        0.3,                 # A
        2.0 * np.pi * 0.23,  # omega (rad/us)
        0.0                  # phi
    ]

    bounds = (
        [0.0,   1.0,  0.0, 0.0,       -2 * np.pi],
        [np.inf, 500.0, 1.0, 10.0,     2 * np.pi]
    )

    popt, pcov = curve_fit(
        wiggle_model,
        t,
        y,
        p0=p0,
        sigma=sigma,
        absolute_sigma=False,
        bounds=bounds,
        maxfev=50000
    )

    N0_fit, tau_fit, A_fit, omega_fit, phi_fit = popt
    fa_fit = omega_fit / (2.0 * np.pi)
    fit = wiggle_model(t, *popt)

    print("\n=== Fit parameters ===")
    print(f"N0    = {N0_fit:.6e}")
    print(f"tau   = {tau_fit:.6f} us")
    print(f"A     = {A_fit:.6f}")
    print(f"omega = {omega_fit:.6f} rad/us")
    print(f"phi   = {phi_fit:.6f} rad")
    print(f"fa    = {fa_fit:.6f} MHz")

    residual = (y - fit) / fit
    residual = residual - np.mean(residual)

    freq, amp, peak_freq, peak_amp = fft_with_peaks(
        residual,
        dt,
        fmin=FFT_FMIN,
        fmax=FFT_FMAX,
        prominence_frac=0.02
    )

    print("\n=== Residual FFT peak ===")
    print(f"Peak frequency = {peak_freq:.6f} MHz")
    print(f"Peak amplitude = {peak_amp:.6f}")

    return {
        "label": label,
        "t": t,
        "y": y,
        "fit": fit,
        "residual": residual,
        "dt": dt,
        "freq": freq,
        "amp": amp,
        "peak_freq": peak_freq,
        "peak_amp": peak_amp,
        "fa_fit": fa_fit,
    }


# ============================================================
# 6) Load and process real + toy
# ============================================================
real_t, real_y, real_source = load_root_spectrum(REAL_ROOT_FILE, energy_threshold=ENERGY_THRESHOLD)
toy_t, toy_y = load_csv_spectrum(TOY_CSV_FILE)

print(f"Real ROOT histogram: {real_source}")

real = process_dataset(real_t, real_y, "REAL DATA")
toy = process_dataset(toy_t, toy_y, "TOY DATA")


# ============================================================
# 7) Time-domain residual comparison
# ============================================================
'''plt.figure(figsize=(12, 5))
plt.plot(real["t"], real["residual"], ".", ms=2, alpha=0.6, label="Real residual")
plt.plot(toy["t"], toy["residual"], ".", ms=2, alpha=0.6, label="Toy residual")
plt.xlabel("Time ($\\mu$s)")
plt.ylabel("(data - fit) / fit")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "residual_time_compare.png"), dpi=300, bbox_inches="tight")
plt.show()
'''

# ============================================================
# 8) Combined residual FFT plot
# ============================================================
plt.figure(figsize=(12, 5))

eps = 1e-8
real_amp = np.maximum(real["amp"], eps)
toy_amp = np.maximum(toy["amp"], eps)

plt.plot(
    real["freq"], real_amp,
    lw=1.0,
    #color="black",
    alpha=0.8,
    label="Real residual FFT"
)

plt.plot(
    toy["freq"], toy_amp,
    lw=1.0,
    #color="blue",
    alpha=0.85,
    label="Toy residual FFT"
)

#plt.axvline(real["peak_freq"], linestyle="--", lw=1.0)
#plt.axvline(toy["peak_freq"], linestyle="--", lw=1.0)

plt.xlim(0, 3.0)
plt.yscale("log")
plt.xlabel("Frequency (MHz)")
plt.ylabel("FFT amplitude")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "residual_fft_compare.png"), dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 9) Peak listing
# ============================================================
def print_top_peaks(freq, amp, label, n=5):
    search_mask = (freq >= FFT_FMIN) & (freq <= FFT_FMAX)
    freq_search = freq[search_mask]
    amp_search = amp[search_mask]

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


print_top_peaks(real["freq"], real["amp"], "REAL residual FFT")
print_top_peaks(toy["freq"], toy["amp"], "TOY residual FFT")



# ============================================================
# Average magnitude in selected FFT band
# ============================================================

def average_magnitude_exclude_peak(freq, amp, fmin, fmax):
    mask = (freq >= fmin) & (freq <= fmax)
    amp_sel = amp[mask]

    if len(amp_sel) == 0:
        return np.nan

    # 去掉最大峰
    amp_sel = amp_sel[amp_sel < np.max(amp_sel)]

    return np.mean(amp_sel)

real_avg_mag = average_magnitude_exclude_peak(real["freq"], real["amp"], FFT_FMIN, FFT_FMAX)
toy_avg_mag  = average_magnitude_exclude_peak(toy["freq"], toy["amp"], FFT_FMIN, FFT_FMAX)

print("\n=== Average FFT magnitude (in band) ===")
print(f"Real data  : {real_avg_mag:.6f}")
print(f"Toy data   : {toy_avg_mag:.6f}")