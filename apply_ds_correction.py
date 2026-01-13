#!/usr/bin/env python3
"""
Apply spectral-dimension (d_s) correction to torus-based mass predictions.

Inputs
------
1) --locked : CSV with species locks (must include at least: species, ax, ay, z_pred).
              If `m_GeV` is present, it will be used as PDG comparison (optional).
2) --ds     : CSV with spectral-dimension flow (must have columns like: t, ds OR just ds).
3) --latent : CSV with the latent z grid (optional; used to set z_min/z_max more robustly).
4) --sector_slopes : optional per-sector slope table (columns: sector, alpha), to use per-sector α.

Model
-----
log m_pred = alpha * z_pred + beta + gamma * (ds_eff - 4)

- alpha, beta: base linear map (anchor-free default: alpha=4.4, beta=0.0)
- gamma: RG-like tilt from d_s flow (default 0.30; tune 0.25–0.35)
- ds_eff: chosen from ds_mean via mapping z_pred -> index in ds(t)

Mapping z -> ds
---------------
u = (z_pred - z_min)/(z_max - z_min)  in [0,1]
idx = round( u * (len(ds)-1) )
ds_eff = ds[idx]

Outputs
-------
- outcsv: table with species, ax, ay, z_pred, ds_eff, logm_pred, m_pred_GeV, m_PDG_GeV, rel_err
- a quick PNG plot (if --plot) comparing predicted vs PDG (log-scale)

Example (Windows):
------------------
python "C:\\...\\apply_ds_correction.py" ^
  --locked  "C:\\...\\Predicted Masses\\all_particles_locked.csv" ^
  --ds      "C:\\...\\ds_mean.csv" ^
  --latent  "C:\\...\\latent_z_merged3.csv" ^
  --outcsv  "C:\\...\\ds_corrected_masses.csv" ^
  --alpha 4.4 --beta 0.0 --gamma 0.30 --plot
"""

import argparse, os, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_ds(ds_path: str) -> np.ndarray:
    ds_df = pd.read_csv(ds_path)
    cols = [c.lower() for c in ds_df.columns]
    if "ds" in cols:
        ds = ds_df.iloc[:, cols.index("ds")].astype(float).values
    else:
        # try to find a column that looks like ds
        cand = None
        for c in ds_df.columns:
            if c.lower().startswith("d") and "s" in c.lower():
                cand = c
                break
        if cand is None:
            raise ValueError(f"Could not find 'ds' column in {ds_path}.")
        ds = ds_df[cand].astype(float).values
    # Safety: clip to a reasonable band
    ds = np.clip(ds, 1.0, 5.0)
    return ds

def load_locked(locked_path: str) -> pd.DataFrame:
    df = pd.read_csv(locked_path)
    # normalize column names (keep originals too)
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    need = ["species", "ax", "ay", "z_pred"]
    for k in need:
        if k not in df.columns:
            raise ValueError(f"'{k}' column missing in locked CSV: {locked_path}")
    return df

def load_latent_bounds(latent_path: str, fallback_zs: np.ndarray):
    if latent_path and os.path.exists(latent_path):
        lat = pd.read_csv(latent_path)
        if "z" in lat.columns:
            zmin = np.nanmin(lat["z"].values)
            zmax = np.nanmax(lat["z"].values)
            if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
                return zmin, zmax
    # fallback to observed z_pred range
    zmin = float(np.nanmin(fallback_zs))
    zmax = float(np.nanmax(fallback_zs))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        raise ValueError("Could not determine a valid (z_min, z_max).")
    return zmin, zmax

def maybe_load_sector_slopes(path: str):
    if not path:
        return None
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Expect columns like: sector, alpha
    cols = [c.lower() for c in df.columns]
    if "sector" in cols and "alpha" in cols:
        out = {}
        for _,row in df.iterrows():
            sec = str(row[df.columns[cols.index("sector")]]).strip().lower()
            try:
                a = float(row[df.columns[cols.index("alpha")]])
                out[sec] = a
            except:
                pass
        return out
    return None

def map_sector(name: str) -> str:
    # crude sector mapping by species label
    n = name.strip().lower()
    if n in ("h","w","z","photon","gluon"):
        return "bosons"
    if n in ("e","mu","tau"):
        return "leptons"
    if n in ("u","c","t"):
        return "up"
    if n in ("d","s","b"):
        return "down"
    if n.startswith("nu"):
        return "neutrinos"
    return "unknown"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", required=True)
    ap.add_argument("--ds",     required=True)
    ap.add_argument("--latent", default=None)
    ap.add_argument("--sector_slopes", default=None,
                    help="Optional CSV with columns [sector, alpha] for per-sector α.")
    ap.add_argument("--alpha", type=float, default=4.4, help="Global alpha (slope) if no per-sector table.")
    ap.add_argument("--beta",  type=float, default=0.0, help="Global beta (offset).")
    ap.add_argument("--gamma", type=float, default=0.30, help="RG-like weight for (ds-4). Try 0.25–0.35.")
    ap.add_argument("--outcsv", required=True)
    ap.add_argument("--plot", action="store_true", help="Write a quick PDG vs Pred comparison plot if PDG present.")
    args = ap.parse_args()

    # Load data
    ds_vec = load_ds(args.ds)
    df = load_locked(args.locked)
    sector_alpha = maybe_load_sector_slopes(args.sector_slopes)

    # Determine z-range (prefer full latent grid)
    zmin, zmax = load_latent_bounds(args.latent, df["z_pred"].values)

    # Prepare outputs
    rows = []
    has_pdg = "m_GeV" in df.columns

    for _, row in df.iterrows():
        species = str(row["species"])
        ax = float(row["ax"])
        ay = float(row["ay"])
        z  = float(row["z_pred"])

        # map z -> ds index
        u = 0.0 if zmax == zmin else (z - zmin) / (zmax - zmin)
        u = float(np.clip(u, 0.0, 1.0))
        idx = int(round(u * (len(ds_vec)-1)))
        ds_eff = float(ds_vec[idx])

        # choose alpha (per sector if provided)
        alpha = args.alpha
        sector = map_sector(species)
        if sector_alpha and sector in sector_alpha:
            alpha = float(sector_alpha[sector])

        beta  = args.beta
        gamma = args.gamma

        logm_pred = alpha*z + beta + gamma*(ds_eff - 4.0)
        m_pred = math.exp(logm_pred)

        m_pdg = float(row["m_GeV"]) if has_pdg and not pd.isna(row["m_GeV"]) else np.nan
        rel_err = (m_pred - m_pdg)/m_pdg if (has_pdg and np.isfinite(m_pdg) and m_pdg>0) else np.nan

        rows.append({
            "species": species, "sector": sector, "ax": ax, "ay": ay,
            "z_pred": z, "ds_eff": ds_eff,
            "alpha_used": alpha, "beta": beta, "gamma": gamma,
            "logm_pred": logm_pred, "m_pred_GeV": m_pred,
            "m_PDG_GeV": m_pdg, "rel_err": rel_err
        })

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.outcsv)), exist_ok=True)
    out.to_csv(args.outcsv, index=False)
    print(f"[WROTE] {args.outcsv}")

    if args.plot and has_pdg and out["m_PDG_GeV"].notna().any():
        # log-log scatter
        fig = plt.figure(figsize=(6.0,5.2))
        sel = out["m_PDG_GeV"] > 0
        xp = out.loc[sel, "m_PDG_GeV"].values
        yp = out.loc[sel, "m_pred_GeV"].values
        lab = out.loc[sel, "species"].values
        plt.loglog(xp, yp, "o")
        # diagonal
        lo = min(xp.min(), yp.min())*0.8
        hi = max(xp.max(), yp.max())*1.2
        xs = np.logspace(np.log10(lo), np.log10(hi), 200)
        plt.loglog(xs, xs, "--")
        for i,(xx,yy,ll) in enumerate(zip(xp,yp,lab)):
            plt.text(xx*1.05, yy*0.95, ll, fontsize=8)
        plt.xlabel("PDG mass [GeV]")
        plt.ylabel("Predicted mass (d_s-corrected) [GeV]")
        plt.title("PDG vs Predicted (with spectral-dimension correction)")
        png = os.path.splitext(args.outcsv)[0] + "_pdg_vs_pred.png"
        plt.tight_layout()
        plt.savefig(png, dpi=150)
        print(f"[WROTE] {png}")