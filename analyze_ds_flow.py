import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# --- File paths ---
base = Path(r"C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\ds flow files\hk_results_local")
theta_file = base / "theta_mean.csv"
meta_file = base / "ensemble_meta.csv"
partial_file = base / "theta_partial.csv"

# --- Load data ---
theta = pd.read_csv(theta_file)
print(f"Loaded theta_mean.csv: {theta.shape} rows")

# Expect columns: t, Theta_mean (or similar)
t_col = [c for c in theta.columns if "t" in c.lower()][0]
theta_col = [c for c in theta.columns if "theta" in c.lower()][0]

t = np.array(theta[t_col])
Theta = np.array(theta[theta_col])

# --- Normalize and smooth ---
Theta /= Theta[0]
Theta = np.maximum(Theta, 1e-20)
Theta_smooth = gaussian_filter1d(Theta, sigma=3)

# --- Compute spectral dimension ---
# d_s(t) = -2 * d log(Theta) / d log(t)
log_t = np.log(t)
log_Theta = np.log(Theta_smooth)
ds = -2.0 * np.gradient(log_Theta, log_t)

# --- Save results ---
ds_out = pd.DataFrame({"t": t, "d_s": ds})
ds_out.to_csv(base / "ds_mean.csv", index=False)
print(f"Saved spectral-dimension curve → {base / 'ds_mean.csv'}")

# --- Plot ---
fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

# Heat trace
ax[0].loglog(t, Theta, label="Raw heat trace", alpha=0.5)
ax[0].loglog(t, Theta_smooth, label="Smoothed", color="k", lw=1.5)
ax[0].set_ylabel(r"$\Theta(t)$ (norm.)")
ax[0].set_title("Heat Trace vs Diffusion Time")
ax[0].legend()

# Spectral dimension
ax[1].semilogx(t, ds, color="C3", lw=2)
ax[1].set_xlabel(r"Diffusion scale $t$")
ax[1].set_ylabel(r"Spectral dimension $d_s(t)$")
ax[1].grid(True, which="both", ls="--", alpha=0.3)
ax[1].set_title("Spectral Dimension Flow")

plt.tight_layout()
plt.savefig(base / "spectral_dimension_ds.png", dpi=300)
plt.show()

print("✅ Done — plot saved as spectral_dimension_ds.png")