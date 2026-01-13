#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid-search over (gamma, lambda_ds) with *analytic beta* that best fits W/Z/H.
- log m_pred = alpha_sector * (lambda_ds * ds_eff * z_pred) + gamma * z_pred + beta*
- beta* = mean over {W,Z,H} of (log m_PDG - [alpha*(lambda*ds*z)+gamma*z])  (least squares)
"""

import argparse, os, math, numpy as np, pandas as pd

def load_locked(path):
    L = pd.read_csv(path)
    # harmonize column names
    if "z_pred" not in L.columns and "z_predi" in L.columns:
        L["z_pred"] = L["z_predi"]
    if "m_PDG_GeV" not in L.columns and "m_GeV" in L.columns:
        L["m_PDG_GeV"] = L["m_GeV"]
    need = ["species","ax","ay","z_pred","m_PDG_GeV"]
    for n in need:
        if n not in L.columns:
            raise ValueError(f"locked missing '{n}', got {list(L.columns)}")
    # sector inference if missing
    if "sector" not in L.columns:
        up,down,lep,neu,bos = set("u c t".split()), set("d s b".split()), set("e mu tau".split()), set("nu1 nu2 nu3 nu_e nu_mu nu_tau".split()), set("W Z H h w z".split())
        def sec_of(s):
            s=s.strip()
            if s in up: return "up"
            if s in down: return "down"
            if s in lep: return "leptons"
            if s in neu: return "neutrinos"
            if s in bos: return "bosons"
            return "unknown"
        L["sector"] = [sec_of(str(s)) for s in L["species"]]
    # numeric
    for c in ["ax","ay","z_pred","m_PDG_GeV"]:
        L[c] = pd.to_numeric(L[c], errors="coerce")
    return L

def load_ds_map(path):
    D = pd.read_csv(path)
    for n in ["ax","ay","ds"]:
        if n not in D.columns:
            raise ValueError(f"dsmap missing '{n}', got {list(D.columns)}")
        D[n] = pd.to_numeric(D[n], errors="coerce")
    D = D.dropna(subset=["ax","ay","ds"]).reset_index(drop=True)
    if len(D)==0: raise ValueError("dsmap empty after cleaning.")
    return D

def load_sector_slopes(path):
    S = pd.read_csv(path)
    if "sector" not in S.columns or "alpha_raw" not in S.columns:
        raise ValueError(f"sector_slopes needs columns ['sector','alpha_raw'...] got {list(S.columns)}")
    amap = {}
    for _,r in S.iterrows():
        sec = str(r["sector"]).strip().lower()
        a = r.get("alpha_raw", np.nan)
        if sec=="neutrinos" and pd.notna(r.get("alpha_norm", np.nan)):
            a = r["alpha_norm"]  # your file: ~1.128786
        if pd.notna(a): amap[sec]=float(a)
    # sensible defaults if any missing
    for k,v in {"up":4.36,"down":4.46,"leptons":4.43,"neutrinos":1.13,"bosons":3.45}.items():
        if k not in amap: amap[k]=v
    return amap

def knn_ds(ax, ay, D, k=3, eps=1e-12):
    # exact
    ex = D[(np.isclose(D["ax"],ax)) & (np.isclose(D["ay"],ay))]
    if len(ex): return float(ex.iloc[0]["ds"])
    # inverse-distance kNN
    dx = D["ax"].values-ax; dy = D["ay"].values-ay
    d2 = dx*dx+dy*dy
    idx = np.argsort(d2)[:max(1,k)]
    w = 1.0/(np.sqrt(d2[idx])+eps); w/=w.sum()
    return float((w*D["ds"].values[idx]).sum())

def fit_beta_star(L, base_log_no_beta):
    # base_log_no_beta[i] = alpha*(lambda*ds*z) + gamma*z   for each row
    # Fit on W/Z/H only:
    want = L["species"].astype(str).str.upper().isin(["W","Z","H"])
    sub = L[want & (L["m_PDG_GeV"]>0)].copy()
    if len(sub)==0: return 0.0
    resid = np.log(sub["m_PDG_GeV"].values) - base_log_no_beta[ sub.index ]
    # Least-squares beta* (also L1≈median is possible)
    return float(np.mean(resid))

def objective_abslog(L, logm_pred):
    # abs log error on W/Z/H
    want = L["species"].astype(str).str.upper().isin(["W","Z","H"])
    sub = L[want & (L["m_PDG_GeV"]>0)].copy()
    if len(sub)==0: return np.inf
    err = np.abs(np.log(sub["m_PDG_GeV"].values) - logm_pred[sub.index])
    return float(np.mean(err))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", required=True)
    ap.add_argument("--dsmap", required=True)
    ap.add_argument("--sectorslopes", required=True)
    ap.add_argument("--outcsv", required=True)
    ap.add_argument("--outpng", default="")
    ap.add_argument("--freeze_neutrinos", action="store_true")
    ap.add_argument("--nn_k", type=int, default=3)
    ap.add_argument("--gamma_min", type=float, default=-0.5)
    ap.add_argument("--gamma_max", type=float, default=0.5)
    ap.add_argument("--gamma_step", type=float, default=0.01)
    ap.add_argument("--lambda_min", type=float, default=0.02)
    ap.add_argument("--lambda_max", type=float, default=0.60)
    ap.add_argument("--lambda_step", type=float, default=0.02)
    args = ap.parse_args()
    print("[ARGS]", vars(args))

    L = load_locked(args.locked)
    D = load_ds_map(args.dsmap)
    alpha_map = load_sector_slopes(args.sectorslopes)

    # attach ds_eff and alphas
    L = L.copy()
    L["ds_eff"] = [knn_ds(float(r.ax), float(r.ay), D, k=args.nn_k) for _,r in L.iterrows()]
    L["alpha_used"] = L["sector"].astype(str).str.lower().map(alpha_map).fillna(4.4)

    z  = L["z_pred"].values
    ds = L["ds_eff"].values
    al = L["alpha_used"].values
    neu_mask = L["sector"].astype(str).str.lower().eq("neutrinos")

    gamma_grid  = np.arange(args.gamma_min, args.gamma_max+1e-12, args.gamma_step)
    lambda_grid = np.arange(args.lambda_min, args.lambda_max+1e-12, args.lambda_step)

    best = {"score":np.inf, "gamma":None, "lambda_ds":None, "beta":None}
    for lam in lambda_grid:
        for gamma in gamma_grid:
            base = al*(lam*ds*z) + gamma*z
            beta_star = fit_beta_star(L, base)
            logm_pred = base + beta_star
            if args.freeze_neutrinos:
                idx = np.where(neu_mask.values)[0]
                pdg = L["m_PDG_GeV"].values[idx]
                ok = pdg>0
                logm_pred[idx[ok]] = np.log(pdg[ok])
            score = objective_abslog(L, logm_pred)
            if score < best["score"]:
                best.update({"score":score,"gamma":gamma,"lambda_ds":lam,"beta":beta_star})

    print("[BEST]", best)

    # final table
    lam, gamma, beta = best["lambda_ds"], best["gamma"], best["beta"]
    base = al*(lam*ds*z) + gamma*z
    logm_pred = base + beta
    if args.freeze_neutrinos:
        idx = np.where(neu_mask.values)[0]
        pdg = L["m_PDG_GeV"].values[idx]
        ok = pdg>0
        logm_pred[idx[ok]] = np.log(pdg[ok])

    L["logm_pred"] = logm_pred
    L["m_pred_GeV"] = np.exp(logm_pred)

    # summary
    M = L[(L["m_PDG_GeV"]>0)].copy()
    M["abs_dlog"] = np.abs(np.log(M["m_PDG_GeV"]) - M["logm_pred"])
    M["rel_err"]  = np.abs(M["m_pred_GeV"]-M["m_PDG_GeV"])/M["m_PDG_GeV"]
    print("[SUMMARY]")
    print(f"  count = {len(M)}")
    print(f"  median |Δ log m| = {float(M['abs_dlog'].median()):.4f}")
    print(f"  mean   |Δ log m| = {float(M['abs_dlog'].mean()):.4f}")
    for sp in ["W","Z","H"]:
        r = M[M["species"].astype(str).str.upper()==sp].iloc[0]
        print(f"  {sp}: pred={r['m_pred_GeV']:.6g}, PDG={r['m_PDG_GeV']:.6g}, |Δlog|={r['abs_dlog']:.4f}")

    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    keep = ["species","sector","ax","ay","z_pred","ds_eff","alpha_used","m_PDG_GeV","m_pred_GeV","logm_pred"]
    for c in keep:
        if c not in L.columns: L[c]=np.nan
    L[keep].to_csv(args.outcsv, index=False)
    print("[WROTE]", args.outcsv)

    if args.outpng:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6,5), dpi=140)
            sc = ax.scatter(D["ax"], D["ay"], c=D["ds"], s=20, alpha=0.8)
            plt.colorbar(sc, ax=ax, label="d_s")
            ax.scatter(L["ax"], L["ay"], edgecolor="k", facecolor="none", s=60)
            for _,r in L.iterrows():
                ax.text(r["ax"], r["ay"], str(r["species"]), fontsize=7, ha="center", va="center")
            ax.set_xlabel("a_x"); ax.set_ylabel("a_y")
            ax.set_title("d_s(a_x,a_y) with species")
            plt.tight_layout(); plt.savefig(args.outpng)
            print("[PLOT]", args.outpng)
        except Exception as e:
            print("[WARN] plot failed:", e)

if __name__ == "__main__":
    main()