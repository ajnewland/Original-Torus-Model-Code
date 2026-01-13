#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calibrate masses using a spectral-dimension (d_s) map defined on (a_x, a_y),
with sector-specific slopes. Robust to column-name differences and missing fields.

Usage (example):
python calibrate_with_dsmap_safe_v2.py ^
  --locked "...\all_particles_locked.csv" ^
  --dsmap  "...\ds_ax_ay_map.csv" ^
  --sectorslopes "...\sector_slopes.csv" ^
  --outcsv "...\masses_with_dsmap_safe.csv" ^
  --outpng "...\ds_field_with_species.png" ^
  --lambda_ds 0.10 --ds_lo 0.06 --ds_hi 4.10 --gamma 0.0 --beta 0.0 --nn_k 3 --freeze_neutrinos

Author: you
"""

import argparse
import math
import sys
import os
import warnings

import numpy as np
import pandas as pd

# Optional plotting (always available in your Python install)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

try:
    from scipy.spatial import cKDTree as KDTree
    _HAVE_KD = True
except Exception:
    _HAVE_KD = False
    KDTree = None


# ----------------------------- utils -----------------------------

def _lc(x):
    return str(x).strip().lower()

SPECIES_SECTOR = {
    # up
    "u":"up","c":"up","t":"up",
    # down
    "d":"down","s":"down","b":"down",
    # leptons
    "e":"leptons","mu":"leptons","tau":"leptons",
    # neutrinos
    "nu1":"neutrinos","nu2":"neutrinos","nu3":"neutrinos",
    # bosons
    "h":"bosons","w":"bosons","z":"bosons"
}

DEFAULT_ALPHAS = {
    "up":        4.361511210581376,
    "down":      4.456469846755816,
    "leptons":   4.425327846103062,
    "neutrinos": 1.1287864401326415,   # from your alpha_norm
    "bosons":    3.4520009308762725
}

def infer_sector(species):
    s = _lc(species)
    return SPECIES_SECTOR.get(s, "unknown")

def safe_log_ratio(pred, pdg, eps=1e-15):
    """Return |log(pred) - log(pdg)| safely. If pdg<=0, return np.inf."""
    if pdg is None or pdg <= 0:
        return np.inf
    p = max(pred, eps)
    return abs(math.log(p) - math.log(pdg))

def nearest_ds(ax, ay, ds_df, k=3):
    """kNN average of ds at (ax, ay)."""
    if len(ds_df) == 0:
        return np.nan
    # Build static KDTree
    if "KDTree" not in nearest_ds.__dict__:
        if _HAVE_KD:
            nearest_ds.KDTree = KDTree(ds_df[["ax","ay"]].values)
        else:
            nearest_ds.KDTree = None
    if nearest_ds.KDTree is not None:
        dists, idxs = nearest_ds.KDTree.query([ax, ay], k=min(k, len(ds_df)))
        if np.isscalar(idxs):
            vals = [ds_df.iloc[idxs]["ds"]]
        else:
            vals = ds_df.iloc[np.atleast_1d(idxs)]["ds"].values
        return float(np.mean(vals))
    # Fallback brute force
    pts = ds_df[["ax","ay"]].values
    d2 = (pts[:,0]-ax)**2 + (pts[:,1]-ay)**2
    order = np.argsort(d2)[:min(k,len(d2))]
    return float(ds_df.iloc[order]["ds"].mean())

def load_sector_alphas(sector_csv):
    df = pd.read_csv(sector_csv)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    have_norm = ("alpha_norm" in df.columns) and df["alpha_norm"].notna().any()
    alphas = {}
    for _,row in df.iterrows():
        sec = _lc(row.get("sector",""))
        if not sec:
            continue
        a = None
        if have_norm and not pd.isna(row.get("alpha_norm", np.nan)):
            a = float(row["alpha_norm"])
        elif not pd.isna(row.get("alpha_raw", np.nan)):
            a = float(row["alpha_raw"])
        if a is not None and np.isfinite(a):
            alphas[sec] = a
    # Fill missing with defaults
    for sec,defv in DEFAULT_ALPHAS.items():
        if sec not in alphas:
            alphas[sec] = defv
    return alphas

def normalize_locked_columns(df):
    """
    Accept your 'locked' CSV with columns:
      ['species','m_GeV','logm','q_target','z_target','ax','ay','z_predi','abs_err','ok','note','expansions']
    and normalize to a standard set:
      species, sector, ax, ay, z_pred, m_PDG_GeV
    """
    cols = {c:_lc(c) for c in df.columns}
    # Species
    if "species" not in cols.values():
        raise ValueError("[ERR] 'locked' must have a 'species' column.")
    # ax, ay
    ax_col = next((c for c in df.columns if _lc(c)=="ax"), None)
    ay_col = next((c for c in df.columns if _lc(c)=="ay"), None)
    if ax_col is None or ay_col is None:
        raise ValueError("[ERR] 'locked' missing 'ax'/'ay' columns.")

    # z_pred: accept 'z_pred' or 'z_predi' or 'z'
    cand_z = [c for c in df.columns if _lc(c) in ("z_pred","z_predi","z")]
    if not cand_z:
        raise ValueError("[ERR] 'locked' missing z_pred/z_predi/z.")
    z_col = cand_z[0]

    # mass: accept 'm_PDG_GeV' or 'm_GeV'
    m_col = next((c for c in df.columns if _lc(c) in ("m_pdg_gev","m_gev")), None)
    if m_col is None:
        raise ValueError("[ERR] 'locked' missing m_PDG_GeV or m_GeV.")

    out = pd.DataFrame({
        "species": df["species"].astype(str).str.strip(),
        "ax":      pd.to_numeric(df[ax_col], errors="coerce"),
        "ay":      pd.to_numeric(df[ay_col], errors="coerce"),
        "z_pred":  pd.to_numeric(df[z_col], errors="coerce"),
        "m_PDG_GeV": pd.to_numeric(df[m_col], errors="coerce")
    })
    # sector (if present)
    if "sector" in [ _lc(c) for c in df.columns ]:
        sec_col = next(c for c in df.columns if _lc(c)=="sector")
        out["sector"] = df[sec_col].astype(str).str.strip().str.lower()
    else:
        out["sector"] = out["species"].map(lambda s: infer_sector(s))

    return out

def robust_clip(x, lo, hi):
    try:
        return float(np.clip(x, lo, hi))
    except Exception:
        return np.nan

# ----------------------------- main -----------------------------

def main():
    p = argparse.ArgumentParser(description="Calibrate masses using d_s map + sector slopes (robust columns).")
    p.add_argument("--locked", required=True, help="CSV with species locks (ax, ay, z_pred*, mass).")
    p.add_argument("--dsmap",  required=True, help="CSV with columns: ax, ay, ds.")
    p.add_argument("--sectorslopes", required=True, help="CSV with sector alphas (alpha_norm or alpha_raw).")
    p.add_argument("--outcsv", required=True, help="Output CSV path.")
    p.add_argument("--outpng", required=True, help="Output PNG of ds field with species overlay.")
    p.add_argument("--lambda_ds", type=float, default=0.10, help="kNN smoothing weight (used via nn_k; keep for compat).")
    p.add_argument("--nn_k", type=int, default=3, help="k neighbors for ds averaging.")
    p.add_argument("--ds_lo", type=float, default=0.06, help="Lower clamp for ds.")
    p.add_argument("--ds_hi", type=float, default=4.10, help="Upper clamp for ds.")
    p.add_argument("--beta", type=float, default=0.0, help="Global additive offset in log-mass.")
    p.add_argument("--gamma", type=float, default=0.0, help="Optional linear bias in z (logm += gamma * z_pred).")
    p.add_argument("--freeze_neutrinos", action="store_true", help="Exclude neutrinos from summaries (kept in CSV).")
    args = p.parse_args()

    print("[ARGS]", vars(args))

    # Load locked
    L_raw = pd.read_csv(args.locked)
    L = normalize_locked_columns(L_raw)

    # Load ds map
    DS = pd.read_csv(args.dsmap)
    DS = DS.rename(columns={c:c.strip() for c in DS.columns})
    need_ds = {"ax","ay","ds"}
    if not need_ds.issubset(set(DS.columns)):
        raise ValueError(f"[ERR] dsmap must have columns {need_ds}. Got: {list(DS.columns)}")
    DS = DS.dropna(subset=["ax","ay","ds"]).copy()
    DS["ax"] = pd.to_numeric(DS["ax"], errors="coerce")
    DS["ay"] = pd.to_numeric(DS["ay"], errors="coerce")
    DS["ds"] = pd.to_numeric(DS["ds"], errors="coerce")
    DS = DS.dropna()

    # Load sector slopes
    alphas = load_sector_alphas(args.sectorslopes)
    print("[INFO] sector alphas:", alphas)

    # Compute ds_eff for each species via kNN averaging
    ds_eff = []
    for _,row in L.iterrows():
        dse = nearest_ds(row["ax"], row["ay"], DS, k=max(1, args.nn_k))
        dse = robust_clip(dse, args.ds_lo, args.ds_hi)
        ds_eff.append(dse)
    L["ds_eff"] = ds_eff

    # Map alpha per sector
    L["sector_lc"] = L["sector"].astype(str).str.strip().str.lower()
    L["alpha_used"] = L["sector_lc"].map(lambda s: alphas.get(s, np.nan))
    # Fallback unknowns
    if L["alpha_used"].isna().any():
        warnings.warn("[WARN] Unknown sector(s) found; using up-quark alpha as fallback.")
        L.loc[L["alpha_used"].isna(), "alpha_used"] = DEFAULT_ALPHAS["up"]

    # Predict log-mass
    # Core relation: log m_pred = alpha * (ds_eff * z_pred) + beta + gamma * z_pred
    L["logm_pred"] = (
        L["alpha_used"] * (L["ds_eff"] * L["z_pred"])
        + (args.gamma * L["z_pred"])
        + args.beta
    )
    L["m_pred_GeV"] = np.exp(L["logm_pred"])

    # Errors
    # abs Δ log m and fractional mass error
    with np.errstate(divide="ignore", invalid="ignore"):
        L["abs_dlog"] = np.abs(np.log(L["m_pred_GeV"]) - np.log(L["m_PDG_GeV"]))
        L["rel_err"]  = np.abs(L["m_pred_GeV"] - L["m_PDG_GeV"]) / L["m_PDG_GeV"].replace(0, np.nan)

    # Clean columns (rename before slicing)  ### FIXED
    L = L.rename(columns={"sector_lc":"sector"})
    keep = ["species","sector","ax","ay","z_pred","ds_eff","alpha_used",
            "m_pred_GeV","logm_pred","m_PDG_GeV","abs_dlog","rel_err"]
    out = L[keep].copy()

    # Write CSV
    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    out.to_csv(args.outcsv, index=False)
    print("[WROTE]", args.outcsv)

    # Summary (optionally freeze neutrinos)
    SUM = out.copy()
    if args.freeze_neutrinos:
        SUM = SUM[~SUM["sector"].isin(["neutrinos"])].copy()

    def _median_safe(x):
        x = x.replace([np.inf, -np.inf], np.nan).dropna()
        return float(x.median()) if len(x) else np.nan

    def _mean_safe(x):
        x = x.replace([np.inf, -np.inf], np.nan).dropna()
        return float(x.mean()) if len(x) else np.nan

    print("[SUMMARY]")
    print("  count =", len(SUM))
    print("  median |Δ log m| =", f"{_median_safe(SUM['abs_dlog']):.4f}")
    print("  mean   |Δ log m| =", f"{_mean_safe(SUM['abs_dlog']):.4f}")
    print("  median % mass error =", f"{100.0*_median_safe(SUM['rel_err']):.2f}%")
    print("  mean   % mass error =", f"{100.0*_mean_safe(SUM['rel_err']):.2f}%")

    # Worst offenders by |Δ log m|
    W = SUM.replace([np.inf,-np.inf], np.nan).dropna(subset=["abs_dlog"]).copy()
    W = W.sort_values("abs_dlog", ascending=False).head(5)
    if len(W):
        print("[WORST by |Δ log m|]")
        for _,r in W.iterrows():
            print(f"  {r['species']:>5s}  m_pred={r['m_pred_GeV']:.6g}  PDG={r['m_PDG_GeV']:.6g}  |Δlog|={r['abs_dlog']:.4f}  %err={100.0*r['rel_err']:.2f}%")

    # Plot ds field + species
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=140)
    if len(DS):
        sc = ax.scatter(DS["ax"], DS["ay"], c=DS["ds"], s=28, alpha=0.7,
                        cmap="viridis", norm=Normalize(vmin=DS["ds"].min(), vmax=DS["ds"].max()),
                        edgecolors="none")
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"$d_s(a_x,a_y)$", fontsize=10)
    # overlay species
    for _,r in out.iterrows():
        ax.plot(r["ax"], r["ay"], "o", ms=5, mec="k", mfc="none", alpha=0.9)
        ax.text(r["ax"], r["ay"], f" {r['species']}", fontsize=8, va="center", ha="left")
    ax.set_xlabel(r"$a_x$")
    ax.set_ylabel(r"$a_y$")
    ax.set_title(r"$d_s$ field with species overlay")
    ax.grid(True, alpha=0.2)
    os.makedirs(os.path.dirname(args.outpng), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.outpng, bbox_inches="tight")
    plt.close(fig)
    print("[PLOT]", args.outpng)

if __name__ == "__main__":
    # Nicer warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        main()
    except Exception as e:
        print("FATAL:", e)
        sys.exit(1)