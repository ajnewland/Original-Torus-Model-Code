#!/usr/bin/env python3
"""
make_ds_ax_ay_map.py
Combine latent_z_merged3.csv and ds_mean.csv into a map (ax, ay, ds).
"""

import pandas as pd
import numpy as np
import os

latent = r"C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\geom_fit_many\latent_z_merged3.csv"
dsfile = r"C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\ds flow files\hk_results_local\ds_mean.csv"
out = r"C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Results\ds flow files\hk_results_local\ds_ax_ay_map.csv"

df_lat = pd.read_csv(latent)
df_ds  = pd.read_csv(dsfile)

# Pick usable columns
ax_col = [c for c in df_lat.columns if "ax" in c.lower()][0]
ay_col = [c for c in df_lat.columns if "ay" in c.lower()][0]
z_col  = [c for c in df_lat.columns if c.lower() in ("z","z_pred")][0]
ds_col = [c for c in df_ds.columns if "ds" in c.lower()][0]

# Sort both by z
df_lat = df_lat.sort_values(z_col).reset_index(drop=True)
ds_vals = pd.to_numeric(df_ds[ds_col], errors="coerce").dropna().values
nL, nD = len(df_lat), len(ds_vals)
print(f"[INFO] latent points={nL}, ds_mean points={nD}")

# Interpolate or repeat ds values to match length
if nD < nL:
    x_old = np.linspace(0,1,nD)
    x_new = np.linspace(0,1,nL)
    ds_interp = np.interp(x_new, x_old, ds_vals)
else:
    ds_interp = ds_vals[:nL]

df_lat["ds"] = ds_interp
df_lat[["ax","ay","ds"]] = df_lat[[ax_col,ay_col,"ds"]]
df_lat[["ax","ay","ds"]].to_csv(out, index=False)
print(f"[WROTE] {out}")