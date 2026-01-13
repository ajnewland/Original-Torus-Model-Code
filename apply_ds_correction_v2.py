#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_ds_correction_v2.py

Anchor-free (fermions) mass calibration with:
- robust column detection
- optional (ax,ay)->ds map (nearest-neighbor)
- sensible fallback if ds is a 1-column series
- neutrino handling via Δm^2 splittings (no absolute anchor)
- optional boson-only beta fit

Formula (non-neutrino):
  log m_pred = α_sector * z_pred + β + γ * (ds_eff - 1)

Neutrinos (normal ordering by default):
  choose m0 (lightest); m1=m0; m2=sqrt(m0^2 + Δm21); m3=sqrt(m0^2 + Δm31)
  assign by neutrino z-order, but DO NOT use sector α/beta (only for ordering).
  You can override m0 and splittings via CLI.

Usage:
  python apply_ds_correction_v2.py ^
    --locked "...\all_particles_locked.csv" ^
    --ds_map "...\ds_ax_ay_map.csv"  (optional; cols: ax,ay,ds) ^
    --ds "...\ds_mean.csv"           (fallback) ^
    --latent "...\latent_z_merged3.csv" ^
    --sector_slopes "...\sector_slopes.csv" ^
    --alpha 4.4 --gamma -0.05 --beta 0.0 ^
    --fit_bosons ^
    --nu_mode normal --nu_m0 1e-12 --dm21 7.53e-5 --dm31 2.44e-3 ^
    --outcsv "...\masses_anchor_free_heattrace.csv"
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------- helpers ----------

def read_csv_flex(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def pick(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

def infer_sector(species: str) -> str:
    s = (species or "").strip().lower()
    if s in {"u","c","t"}: return "up"
    if s in {"d","s","b"}: return "down"
    if s in {"e","mu","tau"}: return "leptons"
    if s.startswith("nu") or s in {"v1","v2","v3"}: return "neutrinos"
    if s in {"w","z","h"}: return "bosons"
    return "unknown"

def load_alphas(slopes_path: str) -> Dict[str,float]:
    df = read_csv_flex(slopes_path)
    s_col = pick(df, ["sector","Sector"])
    if not s_col:
        raise ValueError(f"No 'sector' column in {slopes_path}")
    # preference order:
    cand = ["alpha","alpha_eff","alpha_norm","alpha_raw","alpha_raw_fit"]
    out = {}
    for _,r in df.iterrows():
        sec = str(r[s_col]).strip().lower()
        if not sec or sec=="nan": continue
        a = None
        for c in cand:
            if c in df.columns:
                v = r[c]
                try:
                    v = float(v)
                    if np.isfinite(v):
                        a = v
                        # neutrinos: prefer normalized if present
                        if sec=="neutrinos" and c in ("alpha_norm","alpha_eff"):
                            break
                        break
                except: pass
        if a is not None:
            # guard absurd neutrino raw
            if sec=="neutrinos" and a>100:
                if "alpha_norm" in df.columns:
                    try:
                        nv = float(r["alpha_norm"])
                        if np.isfinite(nv): a = nv
                    except: pass
            out[sec]=float(a)
    return out

def nearest_ds(ax: float, ay: float, ds_map: pd.DataFrame) -> float:
    # simple NN (fast enough for small maps)
    dx = ds_map["ax"] - ax
    dy = ds_map["ay"] - ay
    i = (dx*dx + dy*dy).values.argmin()
    return float(ds_map["ds"].iloc[i])

def summarize(df: pd.DataFrame, tag: str):
    need = {"species","m_PDG_GeV","m_pred_GeV"}
    if not need.issubset(df.columns):
        print(f"[SUMMARY] ({tag}) missing cols.")
        return
    sub = df[(df["m_PDG_GeV"]>0) & df["m_pred_GeV"].apply(np.isfinite)].copy()
    if sub.empty:
        print(f"[SUMMARY] ({tag}) no comparable rows.")
        return
    err = np.abs(np.log(sub["m_pred_GeV"]) - np.log(sub["m_PDG_GeV"]))
    print(f"[SUMMARY] ({tag}) n={len(sub)}, median|Δlog m|={np.median(err):.4f}, mean|Δlog m|={np.mean(err):.4f}")
    sub["_err"]=err
    print("[SUMMARY] worst 5:")
    for _,r in sub.sort_values("_err", ascending=False).head(5).iterrows():
        print(f"  {r['species']:>5s}  pred={r['m_pred_GeV']:.6g}  PDG={r['m_PDG_GeV']:.6g}  |Δlog|={r['_err']:.3f}")

def fit_beta_bosons(df: pd.DataFrame) -> float:
    m = df["species"].str.lower().isin(["w","z","h"])
    sub = df.loc[m].copy()
    sub = sub[(sub["m_PDG_GeV"]>0) & sub["logm_pred"].apply(np.isfinite)]
    if sub.empty: return 0.0
    return float(np.mean(np.log(sub["m_PDG_GeV"]) - sub["logm_pred"]))

# ---------- neutrinos ----------
def neutrino_masses(mode: str, m0: float, dm21: float, dm31: float) -> Tuple[float,float,float]:
    # normal ordering by default
    if mode.lower()=="normal":
        m1 = m0
        m2 = math.sqrt(m0*m0 + dm21)
        m3 = math.sqrt(m0*m0 + dm31)
        return m1,m2,m3
    elif mode.lower()=="inverted":
        # approximate inverted: m3 = m0; m1, m2 heavier
        m3 = m0
        m1 = math.sqrt(m0*m0 + abs(dm31))
        m2 = math.sqrt(m1*m1 + dm21)
        return m1,m2,m3
    else:
        raise ValueError("nu_mode must be 'normal' or 'inverted'")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", required=True)
    ap.add_argument("--ds_map", required=False, help="CSV with columns ax,ay,ds (preferred)")
    ap.add_argument("--ds", required=False, help="Fallback 1-col ds_mean.csv")
    ap.add_argument("--latent", required=False)
    ap.add_argument("--sector_slopes", required=True)
    ap.add_argument("--alpha", type=float, default=4.4)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=-0.05, help="Recommended small negative")
    ap.add_argument("--fit_bosons", action="store_true")
    ap.add_argument("--nu_mode", default="normal", choices=["normal","inverted"])
    ap.add_argument("--nu_m0", type=float, default=1e-12)
    ap.add_argument("--dm21", type=float, default=7.53e-5)   # eV^2 → GeV^2 (we’ll treat numerically; log ratios OK)
    ap.add_argument("--dm31", type=float, default=2.44e-3)
    ap.add_argument("--outcsv", required=True)
    args = ap.parse_args()

    print("[ARGS]", vars(args))

    # locked
    L = read_csv_flex(args.locked)
    sp = pick(L, ["species","label","name"]); zc = pick(L, ["z_pred","z","z_target"])
    mc = pick(L, ["m_GeV","pdg_mass","mass_GeV"])
    axc = pick(L, ["ax","alpha_x","a_x"]); ayc = pick(L, ["ay","alpha_y","a_y"])
    if not (sp and zc and mc):
        raise ValueError("locked CSV missing species/z_pred/mass column")

    df = pd.DataFrame({
        "species": L[sp].astype(str),
        "z_pred": pd.to_numeric(L[zc], errors="coerce"),
        "m_PDG_GeV": pd.to_numeric(L[mc], errors="coerce"),
    })
    if axc: df["ax"]=pd.to_numeric(L[axc], errors="coerce")
    if ayc: df["ay"]=pd.to_numeric(L[ayc], errors="coerce")
    sec_col = pick(L, ["sector"])
    df["sector"] = (L[sec_col].astype(str).str.lower() if sec_col else df["species"].apply(infer_sector))

    # sector alphas
    alphas = load_alphas(args.sector_slopes)
    def alpha_of(s): return float(alphas.get(s.strip().lower(), args.alpha))
    df["alpha_used"] = df["sector"].apply(alpha_of)

    # ds map
    ds_map_df = None
    if args.ds_map and os.path.exists(args.ds_map):
        tmp = read_csv_flex(args.ds_map)
        axm = pick(tmp, ["ax","alpha_x","a_x"])
        aym = pick(tmp, ["ay","alpha_y","a_y"])
        dsm = pick(tmp, ["ds","d_s","spectral_dimension"])
        if not (axm and aym and dsm):
            raise ValueError("ds_map must have ax, ay, ds columns")
        ds_map_df = pd.DataFrame({"ax":pd.to_numeric(tmp[axm], errors="coerce"),
                                  "ay":pd.to_numeric(tmp[aym], errors="coerce"),
                                  "ds":pd.to_numeric(tmp[dsm], errors="coerce")}).dropna()
        print(f"[INFO] ds_map loaded: {len(ds_map_df)} points")
    else:
        # fall back to 1-col ds
        if args.ds and os.path.exists(args.ds):
            D = read_csv_flex(args.ds)
            dsc = pick(D, ["ds","d_s","spectral_dimension"])
            if not dsc:
                raise ValueError("Could not find ds column in ds file")
            ds_vals = pd.to_numeric(D[dsc], errors="coerce").dropna().values
            ds_global = float(np.median(ds_vals)) if len(ds_vals)>0 else 1.0
            print(f"[INFO] using global ds={ds_global:.3f} (median)")
        else:
            ds_global = 1.0
            print("[INFO] no ds provided; using ds=1.0")

    # build predictions per species (non-neutrino first)
    df["ds_eff"] = 1.0
    if ds_map_df is not None and {"ax","ay"}.issubset(df.columns):
        df["ds_eff"] = [nearest_ds(ax,ay,ds_map_df) if np.isfinite(ax) and np.isfinite(ay) else 1.0
                        for ax,ay in zip(df["ax"],df["ay"])]
    elif "ds_global" in locals():
        df["ds_eff"] = ds_global

    df["beta"] = args.beta
    df["gamma"] = args.gamma

    # neutrino handling: compute absolute masses from splittings, assign by z order
    nu_mask = df["sector"].eq("neutrinos")
    nu_df = df.loc[nu_mask, ["species","z_pred"]].copy()
    m1,m2,m3 = neutrino_masses(args.nu_mode, args.nu_m0, args.dm21, args.dm31)
    # order neutrinos by z_pred ascending (lightest more negative z)
    nu_order = nu_df.sort_values("z_pred").index.tolist()
    nu_masses = [m1,m2,m3]  # already normal ordering
    # assign in sorted order
    for idx, mass in zip(nu_order, nu_masses):
        df.loc[idx, "m_pred_GeV"] = float(mass)
        df.loc[idx, "logm_pred"] = float(np.log(mass))

    # non-neutrinos by affine + ds
    non_nu = ~nu_mask
    df.loc[non_nu, "logm_pred"] = (
        df.loc[non_nu, "alpha_used"] * df.loc[non_nu, "z_pred"]
        + df.loc[non_nu, "beta"]
        + df.loc[non_nu, "gamma"] * (df.loc[non_nu, "ds_eff"] - 1.0)
    )
    df.loc[non_nu, "m_pred_GeV"] = np.exp(df.loc[non_nu, "logm_pred"])

    # write pre-fit
    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    df.to_csv(args.outcsv, index=False)
    print("[WROTE]", args.outcsv)
    summarize(df, "pre-fit")

    # optional boson beta fit (recompute non-neutrinos only)
    if args.fit_bosons:
        bfit = fit_beta_bosons(df)
        print(f"[FIT] beta (W/Z/H) = {bfit:.6f}")
        df2 = df.copy()
        mask_adjust = ~nu_mask  # do not alter neutrinos
        df2.loc[mask_adjust, "logm_pred"] = df2.loc[mask_adjust, "logm_pred"] + bfit
        df2.loc[mask_adjust, "m_pred_GeV"] = np.exp(df2.loc[mask_adjust, "logm_pred"])
        out2 = args.outcsv.replace(".csv","_fitbosons.csv")
        df2.to_csv(out2, index=False)
        print("[WROTE]", out2)
        summarize(df2, "post-fit (W/Z/H pinned)")

    # quick neutrino Δm² check
    if nu_df.shape[0] == 3:
        ms = np.sort(df.loc[nu_mask,"m_pred_GeV"].values)
        dm21_pred = ms[1]**2 - ms[0]**2
        dm31_pred = ms[2]**2 - ms[0]**2
        print(f"[NU] predicted Δm21={dm21_pred:.3e}, Δm31={dm31_pred:.3e} (target {args.dm21:.3e}, {args.dm31:.3e})")

    # report sector alphas
    print("\n=== Sector α used ===")
    for s in sorted(df["sector"].unique()):
        a = df.loc[df["sector"].eq(s),"alpha_used"].iloc[0]
        print(f"  {s:10s}  α={a:.6f}")

if __name__=="__main__":
    main()