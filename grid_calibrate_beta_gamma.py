#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid-search calibration for (beta, gamma[, lambda_ds]) using W/Z/H as anchors.
- Reads your 'locked' CSV (with z_pred or z_predi, ax, ay, species, sector, m_GeV/logm).
- Reads a ds(a_x, a_y) map and interpolates ds_eff by k-NN (inverse-distance).
- Reads per-sector slope alphas from sector_slopes.csv (alpha_raw or alpha_norm for neutrinos).
- Predicts masses via:
      log m_pred = alpha_sector * (lambda_ds * ds_eff * z_pred) + gamma * z_pred + beta
- Objective: minimize mean absolute log error on {W, Z, H}. Optionally freeze neutrinos.
- Writes a calibrated CSV with predictions and prints a summary.
"""

import argparse, sys, math, os
import numpy as np
import pandas as pd

def load_locked(path_locked: str) -> pd.DataFrame:
    L = pd.read_csv(path_locked)
    # Harmonize columns
    colmap = {}
    if "z_pred" in L.columns:
        colmap["z_pred"] = "z_pred"
    elif "z_predi" in L.columns:
        colmap["z_predi"] = "z_pred"
    else:
        raise ValueError(f"[ERR] locked CSV missing z_pred/z_predi; got: {list(L.columns)}")

    if "m_PDG_GeV" in L.columns:
        colmap["m_PDG_GeV"] = "m_PDG_GeV"
    elif "m_GeV" in L.columns:
        colmap["m_GeV"] = "m_PDG_GeV"
    else:
        raise ValueError(f"[ERR] locked CSV missing m_PDG_GeV/m_GeV; got: {list(L.columns)}")

    for k,v in colmap.items():
        if k != v:
            L[v] = L[k]

    need = ["species","ax","ay","z_pred","m_PDG_GeV"]
    for n in need:
        if n not in L.columns:
            raise ValueError(f"[ERR] locked CSV missing '{n}'. Got: {list(L.columns)}")

    # Sector: if missing, infer from species groups
    if "sector" not in L.columns:
        # very lightweight mapping
        up_set    = set(["u","c","t"])
        down_set  = set(["d","s","b"])
        lep_set   = set(["e","mu","tau"])
        neu_set   = set(["nu1","nu2","nu3","nu_e","nu_mu","nu_tau"])
        bos_set   = set(["W","Z","H","h","w","z"])
        sec = []
        for s in L["species"].astype(str):
            s0 = s.strip()
            if s0 in up_set: sec.append("up")
            elif s0 in down_set: sec.append("down")
            elif s0 in lep_set: sec.append("leptons")
            elif s0 in neu_set: sec.append("neutrinos")
            elif s0 in bos_set: sec.append("bosons")
            else: sec.append("unknown")
        L["sector"] = sec

    # Ensure numeric
    for c in ["ax","ay","z_pred","m_PDG_GeV"]:
        L[c] = pd.to_numeric(L[c], errors="coerce")
    return L

def load_ds_map(path_dsmap: str) -> pd.DataFrame:
    D = pd.read_csv(path_dsmap)
    # Expected cols: ax, ay, ds
    for n in ["ax","ay","ds"]:
        if n not in D.columns:
            raise ValueError(f"[ERR] dsmap CSV must contain ['ax','ay','ds']; got: {list(D.columns)}")
    for c in ["ax","ay","ds"]:
        D[c] = pd.to_numeric(D[c], errors="coerce")
    D = D.dropna(subset=["ax","ay","ds"]).reset_index(drop=True)
    if len(D)==0:
        raise ValueError("[ERR] dsmap has zero usable rows after cleaning.")
    return D

def load_sector_slopes(path_sectors: str) -> dict:
    S = pd.read_csv(path_sectors)
    # Expected columns include 'sector' and 'alpha_raw' (and optionally 'alpha_norm' for neutrinos)
    if "sector" not in S.columns:
        raise ValueError(f"[ERR] sector_slopes missing 'sector'; got: {list(S.columns)}")
    if "alpha_raw" not in S.columns:
        raise ValueError(f"[ERR] sector_slopes missing 'alpha_raw'; got: {list(S.columns)}")
    # Build mapping
    amap = {}
    for _,r in S.iterrows():
        sec = str(r["sector"]).strip().lower()
        a = r.get("alpha_raw", np.nan)
        if sec=="neutrinos":
            # If alpha_norm exists & finite, prefer it for neutrinos (your file has ~1.128786)
            a_norm = r.get("alpha_norm", np.nan)
            if pd.notna(a_norm):
                a = a_norm
        if pd.notna(a):
            amap[sec]=float(a)
    # Fallbacks for any missing:
    defaults = {
        "up": 4.36,
        "down": 4.46,
        "leptons": 4.43,
        "neutrinos": 1.13,   # from your file's alpha_norm
        "bosons": 3.45
    }
    for k,v in defaults.items():
        if k not in amap:
            amap[k]=v
    return amap

def knn_ds(ax, ay, D: pd.DataFrame, k=3, eps=1e-12):
    """Inverse-distance weighted k-NN ds at (ax,ay)."""
    # If exact match
    exact = D[(np.isclose(D["ax"], ax)) & (np.isclose(D["ay"], ay))]
    if len(exact)>0:
        return float(exact.iloc[0]["ds"])
    # else nearest k
    dx = D["ax"].values - ax
    dy = D["ay"].values - ay
    dist2 = dx*dx + dy*dy
    idx = np.argsort(dist2)[:max(1,k)]
    w = 1.0 / (np.sqrt(dist2[idx]) + eps)
    w = w / w.sum()
    return float(np.sum(w * D["ds"].values[idx]))

def predict_logm(alpha, z_pred, ds_eff, beta, gamma, lambda_ds):
    # Core model:
    # log m = alpha * (lambda_ds * ds_eff * z) + gamma * z + beta
    return alpha * (lambda_ds * ds_eff * z_pred) + gamma * z_pred + beta

def objective_WZH(logm_pred_series: pd.Series, L: pd.DataFrame, species_set=("W","Z","H")):
    # Mean absolute log error for these species (if present)
    rows = []
    for sp in species_set:
        m = L[L["species"].astype(str).str.lower()==sp.lower()]
        if len(m)==0: continue
        idx = m.index[0]
        if pd.isna(L.loc[idx,"m_PDG_GeV"]) or L.loc[idx,"m_PDG_GeV"]<=0:
            continue
        logm_pdg = math.log(L.loc[idx,"m_PDG_GeV"])
        rows.append(abs(logm_pred_series.loc[idx] - logm_pdg))
    if not rows:
        return np.inf
    return float(np.mean(rows))

def run_grid(L, D, alpha_map, beta_grid, gamma_grid, lambda_grid, k_nn=3, freeze_neutrinos=False):
    # Precompute ds_eff per row
    ds_eff = []
    for i,r in L.iterrows():
        ds_eff.append(knn_ds(float(r["ax"]), float(r["ay"]), D, k=k_nn))
    L = L.copy()
    L["ds_eff"] = ds_eff

    # species mask for neutrinos if we freeze
    neu_mask = L["sector"].astype(str).str.lower().eq("neutrinos")

    best = {"score": np.inf, "beta": None, "gamma": None, "lambda_ds": None}
    # For speed, vectorize all static bits
    z = L["z_pred"].values
    ds = L["ds_eff"].values
    # Gather alpha per row
    alpha_used = L["sector"].astype(str).str.lower().map(alpha_map).fillna(4.4).values

    for lam in lambda_grid:
        for beta in beta_grid:
            for gamma in gamma_grid:
                logm_pred = alpha_used * (lam * ds * z) + gamma * z + beta
                # freeze neutrinos: set their logm_pred to PDG
                if freeze_neutrinos:
                    idx = np.where(neu_mask.values)[0]
                    if idx.size>0:
                        # If PDG <=0, skip
                        pdg = L["m_PDG_GeV"].values[idx]
                        ok = pdg>0
                        logm_pred[idx[ok]] = np.log(pdg[ok])
                score = objective_WZH(pd.Series(logm_pred, index=L.index), L)
                if score < best["score"]:
                    best.update({"score":score, "beta":beta, "gamma":gamma, "lambda_ds":lam})
    # Final predictions with best params
    beta, gamma, lam = best["beta"], best["gamma"], best["lambda_ds"]
    logm_pred = alpha_used * (lam * ds * z) + gamma * z + beta
    if freeze_neutrinos:
        idx = np.where(neu_mask.values)[0]
        if idx.size>0:
            pdg = L["m_PDG_GeV"].values[idx]
            ok = pdg>0
            logm_pred[idx[ok]] = np.log(pdg[ok])
    L["alpha_used"] = alpha_used
    L["logm_pred"] = logm_pred
    L["m_pred_GeV"] = np.exp(logm_pred)
    return L, best

def summarize(L: pd.DataFrame):
    # Absolute log error summary (exclude non-positive PDG)
    M = L.copy()
    M = M[(M["m_PDG_GeV"]>0) & np.isfinite(M["m_PDG_GeV"])]
    M["abs_dlog"] = np.abs(np.log(M["m_PDG_GeV"]) - M["logm_pred"])
    med = float(M["abs_dlog"].median()) if len(M) else np.nan
    mean = float(M["abs_dlog"].mean()) if len(M) else np.nan
    # Relative mass error (%)
    M["rel_err"] = np.abs(M["m_pred_GeV"] - M["m_PDG_GeV"]) / M["m_PDG_GeV"]
    medp = float((100*M["rel_err"]).median()) if len(M) else np.nan
    meanp = float((100*M["rel_err"]).mean()) if len(M) else np.nan

    print("[SUMMARY]")
    print(f"  count = {len(M)}")
    print(f"  median |Δ log m| = {med:.4f}")
    print(f"  mean   |Δ log m| = {mean:.4f}")
    print(f"  median % mass error = {medp:.2f}%")
    print(f"  mean   % mass error = {meanp:.2f}%")

    # Show W/Z/H errors
    for sp in ["W","Z","H"]:
        m = M[M["species"].astype(str).str.lower()==sp.lower()]
        if len(m):
            row = m.iloc[0]
            print(f"  {sp}: pred={row['m_pred_GeV']:.6g}  PDG={row['m_PDG_GeV']:.6g}  |Δlog|={row['abs_dlog']:.4f}")

def main():
    ap = argparse.ArgumentParser(description="Grid-search calibration for beta,gamma(,lambda_ds) using W/Z/H.")
    ap.add_argument("--locked", required=True, help="CSV with species, ax, ay, z_pred/z_predi, m_GeV (PDG)")
    ap.add_argument("--dsmap", required=True, help="CSV with columns: ax, ay, ds")
    ap.add_argument("--sectorslopes", required=True, help="CSV with sector slopes (alpha_raw; alpha_norm for neutrinos optional)")
    ap.add_argument("--outcsv", required=True, help="Output CSV path")
    ap.add_argument("--outpng", default="", help="(Optional) write a PNG showing ds-field with species points")
    ap.add_argument("--freeze_neutrinos", action="store_true", help="Hold neutrino masses at PDG during fit")
    ap.add_argument("--nn_k", type=int, default=3, help="k-NN for ds interpolation (default 3)")

    # Grid params
    ap.add_argument("--beta_min", type=float, default=-7.0)
    ap.add_argument("--beta_max", type=float, default=-3.0)
    ap.add_argument("--beta_step", type=float, default=0.1)
    ap.add_argument("--gamma_min", type=float, default=-0.20)
    ap.add_argument("--gamma_max", type=float, default=0.10)
    ap.add_argument("--gamma_step", type=float, default=0.01)
    ap.add_argument("--lambda_min", type=float, default=0.02)
    ap.add_argument("--lambda_max", type=float, default=0.30)
    ap.add_argument("--lambda_step", type=float, default=0.02)

    args = ap.parse_args()
    print("[ARGS]", vars(args))

    L = load_locked(args.locked)
    D = load_ds_map(args.dsmap)
    alpha_map = load_sector_slopes(args.sectorslopes)

    beta_grid   = np.arange(args.beta_min, args.beta_max + 1e-12, args.beta_step)
    gamma_grid  = np.arange(args.gamma_min, args.gamma_max + 1e-12, args.gamma_step)
    lambda_grid = np.arange(args.lambda_min, args.lambda_max + 1e-12, args.lambda_step)

    print(f"[GRID] |beta|={len(beta_grid)}, |gamma|={len(gamma_grid)}, |lambda|={len(lambda_grid)}")

    Lout, best = run_grid(L, D, alpha_map, beta_grid, gamma_grid, lambda_grid,
                          k_nn=args.nn_k, freeze_neutrinos=args.freeze_neutrinos)

    print("[BEST]", best)
    summarize(Lout)

    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    # Order columns for readability
    keep = ["species","sector","ax","ay","z_pred","ds_eff","alpha_used",
            "m_PDG_GeV","m_pred_GeV","logm_pred"]
    for c in keep:
        if c not in Lout.columns:
            Lout[c] = np.nan
    Lout[keep].to_csv(args.outcsv, index=False)
    print("[WROTE]", args.outcsv)

    # Optional picture of ds field with species markers
    if args.outpng:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6,5), dpi=140)
            # draw ds map as scatter
            sc = ax.scatter(D["ax"], D["ay"], c=D["ds"], s=20, alpha=0.7)
            plt.colorbar(sc, ax=ax, label="d_s")
            # overlay species
            ax.scatter(Lout["ax"], Lout["ay"], edgecolor="k", facecolor="none", s=60)
            for _,r in Lout.iterrows():
                ax.text(r["ax"], r["ay"], str(r["species"]), fontsize=7, ha="center", va="center")
            ax.set_xlabel("a_x"); ax.set_ylabel("a_y")
            ax.set_title("d_s(a_x,a_y) with species")
            plt.tight_layout()
            plt.savefig(args.outpng)
            print("[PLOT]", args.outpng)
        except Exception as e:
            print("[WARN] failed to plot ds field:", e)

if __name__ == "__main__":
    main()