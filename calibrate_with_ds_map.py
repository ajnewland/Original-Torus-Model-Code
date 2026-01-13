# -*- coding: utf-8 -*-
"""
Calibrate masses using a per-point spectral-dimension map d_s(ax, ay),
with rescaling to [1, 4], sector-specific slopes, and a tiny gamma sweep.
Keeps W/Z/H near PDG while improving fermion sectors—no PDG anchors in the fit.

Inputs:
  --locked        CSV with locked particles (species, ax, ay, z_pred, m_PDG_GeV, sector, ...)
  --latent        CSV with latent grid (ax, ay, z)
  --dsmap         CSV with spectral dimension map (ax, ay, ds_raw) at grid points
  --sectorslopes  CSV with sector slopes; must contain sector + alpha (any of: alpha, alpha_raw, alpha_norm)

Outputs:
  outcsv          CSV with predictions and diagnostics
  outpng          PNG with d_s heatmap and particle labels

Usage (example at end of file).
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from math import isfinite
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def load_sector_alphas(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    # sector column
    if "sector" not in cols:
        raise ValueError("sector_slopes CSV needs a 'sector' column.")
    # alpha column (accept a few common names)
    alpha_col = None
    for cand in ["alpha", "alpha_raw", "alpha_norm"]:
        if cand in cols:
            alpha_col = cols[cand]
            break
    if alpha_col is None:
        raise ValueError("sector_slopes CSV needs an alpha column (alpha / alpha_raw / alpha_norm).")

    sec2a = {}
    for _, row in df.iterrows():
        sec = str(row[cols["sector"]]).strip().lower()
        try:
            a = float(row[alpha_col])
        except Exception:
            a = np.nan
        if sec and isfinite(a):
            sec2a[sec] = a
    if not sec2a:
        raise ValueError("No valid (sector, alpha) pairs found.")
    return sec2a

def pick_alpha_for_sector(sec, sec2a):
    # normalize sector name
    s = str(sec).strip().lower()
    # common aliases
    aliases = {
        "lepton": "leptons",
        "leptons": "leptons",
        "up": "up",
        "ups": "up",
        "down": "down",
        "downs": "down",
        "boson": "bosons",
        "bosons": "bosons",
        "neutrino": "neutrinos",
        "neutrinos": "neutrinos",
    }
    s = aliases.get(s, s)
    # default fallback to median of available alphas (except neutrinos if missing)
    if s in sec2a:
        return sec2a[s]
    vals = [v for k, v in sec2a.items() if k != "neutrinos" and isfinite(v)]
    return float(np.median(vals)) if vals else 4.4

def rescale_ds_to_1_4(ds_raw):
    # robust rescale of ds_raw to [1, 4] with small eps to avoid degenerate range
    v = np.asarray(ds_raw, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("No finite ds_raw values to rescale.")
    vmin, vmax = np.percentile(v, [2, 98])  # trim mild outliers
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax

def mass_law_logm(z, alpha, ds_eff, beta=0.0, gamma=-0.05):
    """
    Core geometric mass law in log-space (no anchors).
      log m_pred = alpha * z + beta + gamma * ds_eff
    where ds_eff = d_s(ax, ay) in [1, 4] (rescaled from Hutch map).
    """
    return alpha * z + beta + gamma * ds_eff

def summarize(df):
    # sector-wise mean |Δlog m| on fermions and on bosons separately
    has_pdg = df["m_PDG_GeV"].notna() & (df["m_PDG_GeV"] > 0)
    df = df.copy()
    df["dlog"] = np.abs(np.log(df["m_pred_GeV"] / df["m_PDG_GeV"]))
    by = df[has_pdg].groupby("sector")["dlog"].mean().sort_values()
    return by

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", required=True)
    ap.add_argument("--latent", required=True)
    ap.add_argument("--dsmap", required=True)
    ap.add_argument("--sectorslopes", required=True)
    ap.add_argument("--outcsv", required=True)
    ap.add_argument("--outpng", default=None)
    ap.add_argument("--gamma_min", type=float, default=-0.12)
    ap.add_argument("--gamma_max", type=float, default=-0.02)
    ap.add_argument("--gamma_step", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--boson_tolerance", type=float, default=0.05, help="Max mean |Δlog m| for bosons to accept.")
    args = ap.parse_args()

    outcsv = Path(args.outcsv)
    outcsv.parent.mkdir(parents=True, exist_ok=True)

    # Load
    locked = pd.read_csv(args.locked)
    latent = pd.read_csv(args.latent)
    dsmap  = pd.read_csv(args.dsmap)
    sec2a  = load_sector_alphas(args.sectorslopes)

    # Sanity: columns
    need_locked = {"species","ax","ay","z_pred","m_PDG_GeV","sector"}
    if not need_locked.issubset(set(c.lower() for c in locked.columns.str.lower())):
        raise SystemExit(f"[ERR] locked CSV must contain {sorted(need_locked)}. Got: {list(locked.columns)}")

    # Normalize column names
    def normcols(df):
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        return df
    locked = normcols(locked)
    latent = normcols(latent)
    dsmap  = normcols(dsmap)

    # Ensure dsmap has ax, ay, ds_raw
    ds_cols = {c.lower(): c for c in dsmap.columns}
    for req in ["ax","ay"]:
        if req not in ds_cols:
            raise SystemExit("[ERR] dsmap needs columns: ax, ay, (and one ds column).")
    # find a ds column
    ds_name = None
    for cand in ["ds","d_s","ds_raw","spectral_d","spectral_dimension"]:
        if cand in ds_cols:
            ds_name = ds_cols[cand]
            break
    if ds_name is None:
        # if third column exists, assume it's ds
        other = [c for c in dsmap.columns if c not in (ds_cols["ax"], ds_cols["ay"])]
        if other:
            ds_name = other[0]
        else:
            raise SystemExit("[ERR] could not find a ds column in dsmap.")

    ax_col = ds_cols["ax"]; ay_col = ds_cols["ay"]

    # Rescale ds_raw -> ds_eff in [1,4]
    vmin, vmax = rescale_ds_to_1_4(dsmap[ds_name].values)
    ds_eff_vals = 1.0 + 3.0 * (np.clip(dsmap[ds_name].values, vmin, vmax) - vmin) / (vmax - vmin + 1e-12)
    dsmap["ds_eff"] = ds_eff_vals

    # Interpolator for ds_eff on (ax, ay)
    points = dsmap[[ax_col, ay_col]].values
    values = dsmap["ds_eff"].values

    # Build locked table with ds_eff and alpha per sector
    Z = []
    for _, row in locked.iterrows():
        ax = float(row["ax"]); ay = float(row["ay"])
        z  = float(row["z_pred"])
        # nearest/linear interpolation; try linear then fallback to nearest
        ds_here = griddata(points, values, (ax, ay), method="linear")
        if not isfinite(ds_here):
            ds_here = griddata(points, values, (ax, ay), method="nearest")
        Z.append(ds_here)
    locked["ds_eff"] = np.array(Z, dtype=float)
    locked["alpha_used"] = [
        pick_alpha_for_sector(row["sector"], sec2a) for _, row in locked.iterrows()
    ]

    # Small sweep over gamma (and fixed beta) to minimize fermion error
    gammas = np.arange(args.gamma_min, args.gamma_max + 1e-12, args.gamma_step)
    best = None
    records = []

    # Helper to compute predictions and errors
    def predict(beta, gamma):
        df = locked.copy()
        df["logm_pred"] = df.apply(
            lambda r: mass_law_logm(r["z_pred"], r["alpha_used"], r["ds_eff"], beta=beta, gamma=gamma),
            axis=1
        )
        df["m_pred_GeV"] = np.exp(df["logm_pred"])
        # sector diagnostics
        df["abs_dlog"] = np.abs(np.log(df["m_pred_GeV"]/df["m_PDG_GeV"]))
        boson_mean = df.loc[df["sector"].str.lower()=="bosons","abs_dlog"].mean()
        ferm_mean  = df.loc[df["sector"].str.lower().isin(["up","down","leptons"]),"abs_dlog"].mean()
        return df, boson_mean, ferm_mean

    for g in gammas:
        df_try, bmean, fmean = predict(args.beta, g)
        ok_b = (bmean <= args.boson_tolerance)  # keep bosons pinned
        score = fmean + (0 if ok_b else 1.0)  # penalize if bosons drift
        records.append((g, bmean, fmean, score))
        if best is None or score < best[-1]:
            best = (g, bmean, fmean, score, df_try)

    g_best, bmean, fmean, score, df_best = best
    print(f"[SELECT] gamma={g_best:.3f}  (boson mean |Δlog|={bmean:.3f}, fermion mean |Δlog|={fmean:.3f})")

    # Write CSV
    df_out = df_best[[
        "species","sector","ax","ay","z_pred","ds_eff","alpha_used"
    ]].copy()
    df_out["beta"]        = args.beta
    df_out["gamma"]       = g_best
    df_out["logm_pred"]   = df_best["logm_pred"]
    df_out["m_pred_GeV"]  = df_best["m_pred_GeV"]
    df_out["m_PDG_GeV"]   = df_best["m_PDG_GeV"]
    df_out.to_csv(outcsv, index=False)
    print(f"[WROTE] {outcsv}")

    # Print sector summary
    print("[SECTOR mean |Δlog m|]")
    for sec, v in df_best.groupby(df_best["sector"].str.lower())["abs_dlog"].mean().sort_values().items():
        print(f"  {sec:9s}  {v:.3f}")

    # Optional figure
    if args.outpng:
        # coarse heatmap of ds_eff over latent grid extent
        fig, axp = plt.subplots(figsize=(6.5,5.5))
        # build a grid large enough for a decent image
        # base on dsmap points
        ax_min, ax_max = dsmap[ax_col].min(), dsmap[ax_col].max()
        ay_min, ay_max = dsmap[ay_col].min(), dsmap[ay_col].max()
        xx = np.linspace(ax_min, ax_max, 200)
        yy = np.linspace(ay_min, ay_max, 200)
        XX, YY = np.meshgrid(xx, yy)
        ZZ = griddata(points, values, (XX, YY), method="linear")
        # nearest fill for NaNs
        mask = ~np.isfinite(ZZ)
        if mask.any():
            ZZ[mask] = griddata(points, values, (XX[mask], YY[mask]), method="nearest")

        im = axp.imshow(ZZ, extent=[ax_min, ax_max, ay_min, ay_max],
                        origin="lower", aspect="auto")
        plt.colorbar(im, ax=axp, label=r"$d_s(a_x,a_y)$ (rescaled to [1,4])")
        # overlay particle labels
        for _, r in df_best.iterrows():
            axp.plot(r["ax"], r["ay"], "o", ms=4)
            axp.text(r["ax"]+0.005, r["ay"]+0.003, str(r["species"]), fontsize=7)
        axp.set_xlabel(r"$a_x$")
        axp.set_ylabel(r"$a_y$")
        axp.set_title("Spectral-dimension field with locked species")
        outpng = Path(args.outpng)
        outpng.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(outpng, dpi=160)
        print(f"[PLOT] {outpng}")

if __name__ == "__main__":
    main()