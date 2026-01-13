# -*- coding: utf-8 -*-
"""
Calibrate masses using per-point spectral dimension map d_s(ax, ay),
automatically adapted for the user's existing file conventions.

Compatible with:
  species,m_GeV,logm,q_target,z_target,ax,ay,z_pred,abs_err,ok,note,expansions
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from math import isfinite

def load_sector_alphas(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    alpha_col = None
    for cand in ["alpha", "alpha_raw", "alpha_norm"]:
        if cand in cols:
            alpha_col = cols[cand]
            break
    sec_col = cols.get("sector", list(df.columns)[0])
    sec2a = {}
    for _, row in df.iterrows():
        sec = str(row[sec_col]).strip().lower()
        try:
            a = float(row[alpha_col])
        except Exception:
            a = np.nan
        if sec and isfinite(a):
            sec2a[sec] = a
    return sec2a

def infer_sector(species):
    s = str(species).lower()
    if s in ["e","mu","tau"]: return "leptons"
    if s in ["u","c","t"]: return "up"
    if s in ["d","s","b"]: return "down"
    if s.startswith("nu"): return "neutrinos"
    if s in ["w","z","h"]: return "bosons"
    return "unknown"

def rescale_ds(v):
    v = np.asarray(v, dtype=float)
    finite = np.isfinite(v)
    if not np.any(finite):
        raise ValueError("No finite ds values to rescale.")
    vmin, vmax = np.percentile(v[finite], [2,98])
    if vmax <= vmin: vmax = vmin + 1e-6
    return 1.0 + 3.0 * (np.clip(v, vmin, vmax) - vmin)/(vmax - vmin + 1e-12)

def mass_law(z, alpha, ds_eff, beta=0.0, gamma=-0.05):
    return np.exp(alpha*z + beta + gamma*ds_eff)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", required=True)
    ap.add_argument("--latent", required=True)
    ap.add_argument("--dsmap", required=True)
    ap.add_argument("--sectorslopes", required=True)
    ap.add_argument("--outcsv", required=True)
    ap.add_argument("--outpng", required=True)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--gamma_min", type=float, default=-0.12)
    ap.add_argument("--gamma_max", type=float, default=-0.02)
    ap.add_argument("--gamma_step", type=float, default=0.01)
    ap.add_argument("--boson_tolerance", type=float, default=0.05)
    args = ap.parse_args()

    locked = pd.read_csv(args.locked)
    # adapt column names
    if "m_GeV" in locked.columns:
        locked = locked.rename(columns={"m_GeV": "m_PDG_GeV"})
    if "z_target" in locked.columns:
        locked = locked.rename(columns={"z_target": "z_pred"})
    if "sector" not in locked.columns:
        locked["sector"] = locked["species"].apply(infer_sector)

    dsmap = pd.read_csv(args.dsmap)
    ax_col, ay_col = "ax", "ay"
    ds_col = [c for c in dsmap.columns if c.lower().startswith("ds") or "spectral" in c.lower()][0]
    dsmap["ds_eff"] = rescale_ds(dsmap[ds_col])

    sec2a = load_sector_alphas(args.sectorslopes)

    points = dsmap[[ax_col, ay_col]].values
    values = dsmap["ds_eff"].values

    locked["ds_eff"] = griddata(points, values, locked[["ax","ay"]], method="linear")
    locked["ds_eff"].fillna(method="bfill", inplace=True)
    locked["alpha_used"] = locked["sector"].apply(lambda s: sec2a.get(s, np.median(list(sec2a.values()))))

    gammas = np.arange(args.gamma_min, args.gamma_max+1e-12, args.gamma_step)
    best_gamma, best_score, best_df = None, np.inf, None

    for g in gammas:
        df = locked.copy()
        df["m_pred_GeV"] = mass_law(df["z_pred"], df["alpha_used"], df["ds_eff"], beta=args.beta, gamma=g)
        df["abs_dlog"] = np.abs(np.log(df["m_pred_GeV"]/df["m_PDG_GeV"]))
        boson_mean = df.loc[df["sector"]=="bosons","abs_dlog"].mean()
        ferm_mean  = df.loc[df["sector"].isin(["up","down","leptons"]),"abs_dlog"].mean()
        score = ferm_mean + (10*max(0,boson_mean-args.boson_tolerance))
        if score < best_score:
            best_score, best_gamma, best_df = score, g, df

    print(f"[SELECT] gamma={best_gamma:.3f} (score={best_score:.3f})")
    best_df.to_csv(args.outcsv, index=False)
    print(f"[WROTE] {args.outcsv}")

    # Plot d_s field
    fig, ax = plt.subplots(figsize=(6,5))
    grid_ax = np.linspace(dsmap[ax_col].min(), dsmap[ax_col].max(), 200)
    grid_ay = np.linspace(dsmap[ay_col].min(), dsmap[ay_col].max(), 200)
    X, Y = np.meshgrid(grid_ax, grid_ay)
    Z = griddata(points, values, (X, Y), method="linear")
    im = ax.imshow(Z, origin="lower", extent=[grid_ax.min(),grid_ax.max(),grid_ay.min(),grid_ay.max()], aspect="auto")
    plt.colorbar(im, ax=ax, label=r"$d_s(a_x,a_y)$")
    for _,r in best_df.iterrows():
        ax.text(r["ax"],r["ay"],r["species"],fontsize=7)
    ax.set_xlabel(r"$a_x$")
    ax.set_ylabel(r"$a_y$")
    ax.set_title("Spectral-dimension field with locked species")
    plt.tight_layout()
    plt.savefig(args.outpng,dpi=160)
    print(f"[PLOT] {args.outpng}")

if __name__ == "__main__":
    main()