#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_ds_correction_flex.py
--------------------------------
Robust, ready-to-run mass calibration without anchors:
- Handles varied column names in your CSVs
- Flexible sector slope ingestion (alpha/alpha_eff/alpha_norm/alpha_raw)
- Uses ds_mean.csv to add a controlled spectral-dimension correction
- Optional global β fit on W/Z/H

Formula:
  log m_pred = α_sector * z_pred + β + γ * (ds_eff - 1)

Usage (example):
  python apply_ds_correction_flex.py ^
    --locked  "...\all_particles_locked.csv" ^
    --ds      "...\ds_mean.csv" ^
    --latent  "...\latent_z_merged3.csv" ^
    --sector_slopes "...\sector_slopes.csv" ^
    --alpha 4.4 --gamma 0.08 --beta 0.0 ^
    --fit_bosons ^
    --outcsv  "...\masses_anchor_free_heattrace.csv"
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ------------------------------ Helpers ------------------------------

def read_csv_flex(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # strip spaces in headers & values
    df.columns = [str(c).strip() for c in df.columns]
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df


def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_sector_from_species(species: str) -> str:
    s = species.strip().lower()
    up = {"u", "c", "t"}
    down = {"d", "s", "b"}
    leptons = {"e", "mu", "tau"}
    bosons = {"w", "z", "h"}
    neutrinos = {"nu", "nu1", "nu2", "nu3", "v1", "v2", "v3"}

    if s in up:
        return "up"
    if s in down:
        return "down"
    if s in leptons:
        return "leptons"
    # neutrino labels vary; handle prefixes
    if s in neutrinos or s.startswith("nu"):
        return "neutrinos"
    if s in bosons:
        return "bosons"
    # fallback (rare)
    return "unknown"


def load_sector_alphas(slopes_path: str) -> Dict[str, float]:
    """
    Accepts a wide variety of sector slope files. Preference:
      alpha -> alpha_eff -> alpha_norm -> alpha_raw
    For neutrinos: if alpha_norm exists (finite and reasonable), use it.
    """
    df = read_csv_flex(slopes_path)
    # normalize 'sector' column name
    sector_col = pick_first_existing(df, ["sector", "Sector", "SECTOR"])
    if not sector_col:
        raise ValueError(f"Could not find a 'sector' column in {slopes_path}. Columns: {list(df.columns)}")
    # candidate alpha columns (in priority order)
    alpha_cols = ["alpha", "alpha_eff", "alpha_norm", "alpha_raw", "alpha_raw_fit"]

    alphas = {}
    for _, row in df.iterrows():
        sec = str(row[sector_col]).strip().lower()
        if sec == "" or sec == "nan":
            continue
        # find first usable alpha
        alpha_val = None
        for cand in alpha_cols:
            if cand in df.columns:
                val = row[cand]
                try:
                    if pd.notna(val) and np.isfinite(float(val)):
                        alpha_val = float(val)
                        # For neutrinos, prefer a "normalized" finite alpha if present
                        if sec == "neutrinos" and cand in ("alpha_norm", "alpha_eff"):
                            break
                        # For non-neutrinos, alpha (or alpha_eff) is fine
                        # Fallthrough picks first available in priority order
                        break
                except Exception:
                    pass

        if alpha_val is None:
            # Nothing useful found; skip
            continue

        # Heuristic: If sec=='neutrinos' and alpha looks absurdly large (like 1e4),
        # try to replace with alpha_norm if present:
        if sec == "neutrinos" and alpha_val > 100.0:
            if "alpha_norm" in df.columns and pd.notna(row["alpha_norm"]):
                try:
                    nv = float(row["alpha_norm"])
                    if np.isfinite(nv):
                        alpha_val = nv
                except Exception:
                    pass

        alphas[sec] = alpha_val

    return alphas


def map_ds_from_z(z: float, zmin: float, zmax: float, ds_series: np.ndarray) -> float:
    """
    Map z in [zmin, zmax] to an index in [0, N-1] and sample ds mean value.
    """
    if zmax <= zmin:
        return 1.0
    x = (z - zmin) / (zmax - zmin)
    x = min(max(x, 0.0), 1.0)
    idx = int(round(x * (len(ds_series) - 1)))
    return float(ds_series[idx])


def fit_global_beta_on_bosons(df: pd.DataFrame) -> float:
    """
    Return a global beta offset so that W/Z/H geometric mean error is zero.
    We minimize mean(log m_pdg - log m_pred) over W/Z/H only.
    """
    mask = df["species"].str.lower().isin(["w", "z", "h"])
    ref = df.loc[mask].copy()
    if ref.empty:
        return 0.0
    if not all(col in ref.columns for col in ["m_PDG_GeV", "logm_pred"]):
        return 0.0
    # avoid invalid rows
    ref = ref[(ref["m_PDG_GeV"] > 0) & ref["logm_pred"].apply(np.isfinite)]
    if ref.empty:
        return 0.0
    return float(np.mean(np.log(ref["m_PDG_GeV"].values) - ref["logm_pred"].values))


def summarize_errors(df: pd.DataFrame, label: str):
    if not {"species", "m_PDG_GeV", "m_pred_GeV"}.issubset(df.columns):
        print(f"[SUMMARY] ({label}) Missing columns to compute errors.")
        return
    dff = df.copy()
    dff = dff[(dff["m_PDG_GeV"] > 0) & dff["m_pred_GeV"].apply(np.isfinite)]
    if dff.empty:
        print(f"[SUMMARY] ({label}) No comparable rows.")
        return
    rel = np.abs(np.log(dff["m_pred_GeV"]) - np.log(dff["m_PDG_GeV"]))
    print(f"[SUMMARY] ({label}) count={len(dff)}, median |Δlog m|={np.median(rel):.4f}, mean |Δlog m|={np.mean(rel):.4f}")
    # Show a few largest errors
    dff["_err"] = rel
    worst = dff.sort_values("_err", ascending=False).head(5)
    print("[SUMMARY] worst 5 by |Δlog m|:")
    for _, r in worst.iterrows():
        print(f"  {r['species']:>5s}  m_pred={r['m_pred_GeV']:.6g}  PDG={r['m_PDG_GeV']:.6g}  |Δlog|={r['_err']:.4f}")


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", required=True, help="all_particles_locked.csv")
    ap.add_argument("--ds", required=True, help="ds_mean.csv (has ds/d_s/spectral_dimension column)")
    ap.add_argument("--latent", required=False, help="latent_z_merged*.csv (used for z-range sanity)")
    ap.add_argument("--sector_slopes", required=True, help="sector_slopes.csv (any of alpha/alpha_eff/alpha_norm/alpha_raw)")
    ap.add_argument("--alpha", type=float, default=4.4, help="fallback α if sector not found (default 4.4)")
    ap.add_argument("--beta", type=float, default=0.0, help="global β offset (applied before optional boson fit)")
    ap.add_argument("--gamma", type=float, default=0.0, help="weight for ds correction term (ds_eff - 1)")
    ap.add_argument("--fit_bosons", action="store_true", help="fit a single global β using W/Z/H only")
    ap.add_argument("--outcsv", required=True, help="output CSV")

    args = ap.parse_args()
    print("[INFO] script:", os.path.abspath(sys.argv[0]))
    print("[ARGS]", vars(args))

    # 1) Load locked (species + z_pred + PDG masses)
    dfL = read_csv_flex(args.locked)
    # Flexible columns
    species_col = pick_first_existing(dfL, ["species", "label", "name"])
    z_col = pick_first_existing(dfL, ["z_pred", "z", "z_target"])
    m_pdg_col = pick_first_existing(dfL, ["m_GeV", "pdg_mass", "mass_GeV"])
    ax_col = pick_first_existing(dfL, ["ax", "alpha_x", "a_x"])
    ay_col = pick_first_existing(dfL, ["ay", "alpha_y", "a_y"])
    sector_col = pick_first_existing(dfL, ["sector"])

    needed = [species_col, z_col, m_pdg_col]
    if any(c is None for c in needed):
        raise ValueError(f"[ERROR] Missing required columns in locked CSV. Found={list(dfL.columns)}; "
                         f"needed species~{species_col}, z~{z_col}, PDG mass~{m_pdg_col}")

    df = pd.DataFrame({
        "species": dfL[species_col].astype(str),
        "z_pred": pd.to_numeric(dfL[z_col], errors="coerce"),
        "m_PDG_GeV": pd.to_numeric(dfL[m_pdg_col], errors="coerce"),
    })
    if ax_col: df["ax"] = pd.to_numeric(dfL[ax_col], errors="coerce")
    if ay_col: df["ay"] = pd.to_numeric(dfL[ay_col], errors="coerce")
    if sector_col:
        df["sector"] = dfL[sector_col].astype(str).str.strip().str.lower()
    else:
        df["sector"] = df["species"].apply(infer_sector_from_species)

    # 2) Load ds series
    dsdf = read_csv_flex(args.ds)
    ds_col = pick_first_existing(dsdf, ["ds", "d_s", "spectral_dimension", "Ds", "DS"])
    if not ds_col:
        raise ValueError(f"[ERROR] Could not find ds column in {args.ds}. Columns: {list(dsdf.columns)}")
    ds_series = pd.to_numeric(dsdf[ds_col], errors="coerce").dropna().values
    if len(ds_series) < 5:
        print("[WARN] ds series is very short; ds correction will be noisy.")
    ds_min, ds_max = float(np.min(ds_series)), float(np.max(ds_series))
    print(f"[INFO] ds length={len(ds_series)}, min={ds_min:.3f}, max={ds_max:.3f}")

    # 3) Latent for z range (optional but helpful)
    zmin, zmax = float(df["z_pred"].min()), float(df["z_pred"].max())
    if args.latent and os.path.exists(args.latent):
        ldf = read_csv_flex(args.latent)
        lz_col = pick_first_existing(ldf, ["z", "z_pred"])
        if lz_col:
            zmin = min(zmin, float(pd.to_numeric(ldf[lz_col], errors="coerce").min()))
            zmax = max(zmax, float(pd.to_numeric(ldf[lz_col], errors="coerce").max()))
    print(f"[INFO] z-range: {zmin:.6f} .. {zmax:.6f}")

    # 4) Sector alphas
    sector_alphas = load_sector_alphas(args.sector_slopes)
    print("[INFO] sector alphas (ingested):", json.dumps(sector_alphas, indent=2))
    # fill missing with fallback alpha
    def alpha_for(sec: str) -> float:
        sec = (sec or "").strip().lower()
        return float(sector_alphas.get(sec, args.alpha))
    df["alpha_used"] = df["sector"].apply(alpha_for)

    # 5) ds_eff mapping + prediction
    df["ds_eff"] = df["z_pred"].apply(lambda z: map_ds_from_z(z, zmin, zmax, ds_series))
    df["beta"] = float(args.beta)
    df["gamma"] = float(args.gamma)
    df["logm_pred"] = df["alpha_used"] * df["z_pred"] + df["beta"] + df["gamma"] * (df["ds_eff"] - 1.0)
    df["m_pred_GeV"] = np.exp(df["logm_pred"])

    # 6) Write first-pass CSV
    outcsv = args.outcsv
    os.makedirs(os.path.dirname(outcsv), exist_ok=True)
    df.to_csv(outcsv, index=False)
    print("[WROTE]", outcsv)

    summarize_errors(df, "pre-fit")

    # 7) Optional global β fit on W/Z/H only (keeps fermions anchor-free)
    if args.fit_bosons:
        beta_fit = fit_global_beta_on_bosons(df)
        print(f"[FIT] global beta on W/Z/H = {beta_fit:.6f}")
        dff = df.copy()
        dff["logm_pred"] = dff["logm_pred"] + beta_fit
        dff["m_pred_GeV"] = np.exp(dff["logm_pred"])
        dff["beta"] = dff["beta"] + beta_fit
        out2 = outcsv.replace(".csv", "_fitbosons.csv")
        dff.to_csv(out2, index=False)
        print("[WROTE]", out2)
        summarize_errors(dff, "post-fit (W/Z/H pinned)")

    # 8) Small report
    print("\n=== Per-sector α used ===")
    for sec in sorted(df["sector"].unique()):
        a = df.loc[df["sector"] == sec, "alpha_used"].iloc[0]
        print(f"  {sec:10s}  α = {a:.6f}")
    print("\nDone.")


if __name__ == "__main__":
    main()