import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

TOY_RES_FILE = "residuals.csv"
ROOT_RES_FILE = "real_residuals.csv"


def load_residual(csv_path):
    df = pd.read_csv(csv_path)

    required_cols = {"Time_us", "Residual"}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"{csv_path} must contain columns: {required_cols}")

    x = df["Time_us"].values
    y = df["Residual"].values
    return x, y


def compute_fft(x, y, freq_min=0.1):
    x = np.asarray(x)
    y = np.asarray(y)

    n = len(y)
    if n < 4:
        raise ValueError("Not enough points for FFT.")

    dt = np.mean(np.diff(x))
    freq = fftfreq(n, d=dt)[: n // 2]

    window = np.hanning(n)
    y_win = (y - np.mean(y)) * window
    amp = np.abs(fft(y_win)[: n // 2])

    mask = freq > freq_min
    return freq, amp, mask


toy_x, toy_res = load_residual(TOY_RES_FILE)
root_x, root_res = load_residual(ROOT_RES_FILE)



# ============================================================
# RESIDUAL
# ============================================================
plt.figure(figsize=(12, 4))
plt.scatter(toy_x, toy_res, s=0.25, label="Toy Residual")
plt.scatter(root_x, root_res, s=0.25, label="ROOT Residual")
plt.xlabel(r"Time in Lab Frame ($\mu s$)")
plt.ylabel("Residual")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plot/COMP_RES.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# FFT
# ============================================================
toy_freq, toy_fft, toy_mask = compute_fft(toy_x, toy_res, .1)
root_freq, root_fft, root_mask = compute_fft(root_x, root_res, .1)

plt.figure(figsize=(12, 4))
toy_plot = np.clip(toy_fft, 1e-12, None)
root_plot = np.clip(root_fft, 1e-12, None)
plt.plot(toy_freq[toy_mask],toy_plot[toy_mask],label="Toy Residual FFT",lw=0.8,alpha=0.6,)
plt.plot(root_freq[root_mask],root_plot[root_mask],label="ROOT Residual FFT",lw=0.8,alpha=0.6,)
plt.xlabel("Frequency (MHz)")
plt.ylabel("Amplitude")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.yscale("log")
plt.xlim(0.1, 3.0)
ymax = max(np.max(toy_plot[toy_mask]),np.max(root_plot[root_mask]),)
plt.ylim(10, ymax * 1.2)
plt.tight_layout()
plt.savefig("plot/COMP_FFT.png", dpi=300, bbox_inches="tight")
plt.show()