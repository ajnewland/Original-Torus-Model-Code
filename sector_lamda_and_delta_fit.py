#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate masses with a d_s(a_x,a_y) map:
1) Regress (beta, lambda_b, gamma) on W/Z/H.
2) For each fermion sector (up, down, leptons), fit (lambda_s, Delta_s) by least squares.
3) Optionally freeze neutrinos at PDG.

Input CSVs:
- locked: has species, ax, ay, z_pred (or z_predi), and m_PDG_GeV (or m_GeV)
- dsmap:  columns: ax, ay, ds
- sector slopes: sector, alpha_raw, (alpha_norm optional for neutrinos)

Author: you
"""

import argparse, sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def read_locked(path):
    L = pd.read_csv(path)
    if 'z_pred' not in L.columns and 'z_predi' in L.columns:
        L['z_pred'] = L['z_predi']
    req = ['species','ax','ay','z_pred']
    for c in req:
        if c not in L.columns:
            raise ValueError(f"[ERR] locked missing '{c}'")
    if 'm_PDG_GeV' not in L.columns:
        if 'm_GeV' in L.columns: L['m_PDG_GeV'] = L['m_GeV']
        else: raise ValueError("[ERR] locked needs m_PDG_GeV or m_GeV")
    # sector inference if needed
    if 'sector' not in L.columns:
        def sec(sp):
            sp=str(sp).lower()
            if sp in ('w','z','h'): return 'bosons'
            if sp.startswith('nu'): return 'neutrinos'
            if sp in ('u','c','t'): return 'up'
            if sp in ('d','s','b'): return 'down'
            if sp in ('e','mu','tau'): return 'leptons'
            return 'unknown'
        L['sector'] = L['species'].map(sec)
    # numerics
    for c in ['ax','ay','z_pred','m_PDG_GeV']:
        L[c] = pd.to_numeric(L[c], errors='coerce')
    L['logm_PDG'] = np.log(L['m_PDG_GeV'])
    L = L.dropna(subset=['species','sector','ax','ay','z_pred','m_PDG_GeV','logm_PDG']).copy()
    return L

def read_dsmap(path):
    D = pd.read_csv(path)
    for c in ['ax','ay','ds']:
        if c not in D.columns: raise ValueError("[ERR] dsmap must have ax, ay, ds")
    D[['ax','ay','ds']] = D[['ax','ay','ds']].apply(pd.to_numeric, errors='coerce')
    return D.dropna()

def read_slopes(path):
    S = pd.read_csv(path)
    for c in ['sector','alpha_raw']:
        if c not in S.columns: raise ValueError("[ERR] sector_slopes needs sector, alpha_raw")
    if 'alpha_norm' not in S.columns: S['alpha_norm']=np.nan
    a_raw = {str(r.sector).lower(): float(r.alpha_raw) for _,r in S.iterrows()}
    a_norm= {str(r.sector).lower(): (float(r.alpha_norm) if pd.notnull(r.alpha_norm) else np.nan)
             for _,r in S.iterrows()}
    return a_raw, a_norm

def build_knn(D, k):
    XY = D[['ax','ay']].values
    nbrs = NearestNeighbors(n_neighbors=min(k,len(XY))).fit(XY)
    return nbrs, XY, D['ds'].values

def ds_lookup(ax,ay,nbrs,XY,vals,k):
    dist, idx = nbrs.kneighbors([[ax,ay]], n_neighbors=min(k,len(XY)), return_distance=True)
    return float(np.mean(vals[idx[0]]))

def fit_beta_lambda_gamma_bosons(L, alpha_b, dfun):
    """Solve [beta, lambda_b, gamma] from W/Z/H exactly (linear solve)."""
    B = L[L['sector'].str.lower()=='bosons'].copy()
    if len(B) != 3:
        raise ValueError("[ERR] Need exactly W,Z,H in locked for boson fit.")
    rows=[]
    y=[]
    for _,r in B.iterrows():
        z = float(r['z_pred'])
        ds= dfun(float(r['ax']), float(r['ay']))
        X = alpha_b * ds * z
        # row is [1, X, z]
        rows.append([1.0, X, z])
        y.append(float(r['logm_PDG']))
    M = np.array(rows, dtype=float)
    y = np.array(y, dtype=float)
    # Solve M*[beta, lambda_b, gamma]^T = y
    sol = np.linalg.solve(M, y)
    beta, lam_b, gamma = map(float, sol)
    return beta, lam_b, gamma

def fit_sector_lambda_delta(Lsec, alpha_s, beta, lam_b, gamma, dfun, nonneg_lambda=True):
    """
    Solve (lambda_s, Delta_s) for a sector by least squares:
    logm = beta + gamma z + lambda_s*(alpha_s ds z) + Delta_s.
    """
    Xs=[]; ys=[]
    for _,r in Lsec.iterrows():
        z  = float(r['z_pred'])
        ds = dfun(float(r['ax']), float(r['ay']))
        Xs.append([alpha_s*ds*z, 1.0])  # [coef for lambda_s, coef for Delta_s]
        base = beta + gamma*z
        ys.append(float(r['logm_PDG']) - base)
    X = np.array(Xs, dtype=float)   # n×2
    y = np.array(ys, dtype=float)   # n
    # Least squares
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    lam_s, Delta_s = map(float, sol)
    if nonneg_lambda and lam_s < 0:
        # Refit with lambda_s=0 fixed => Delta_s = mean(y)
        lam_s = 0.0
        Delta_s = float(np.mean(y))
    return lam_s, Delta_s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", required=True)
    ap.add_argument("--dsmap", required=True)
    ap.add_argument("--sectorslopes", required=True)
    ap.add_argument("--outcsv", required=True)
    ap.add_argument("--outpng", required=True)
    ap.add_argument("--nn_k", type=int, default=3)
    ap.add_argument("--median_normalize_ds", action="store_true", help="Normalize ds by its median")
    ap.add_argument("--freeze_neutrinos", action="store_true")
    args = ap.parse_args()
    print("[ARGS]", vars(args))

    L  = read_locked(args.locked)
    D  = read_dsmap(args.dsmap)
    a_raw, a_norm = read_slopes(args.sectorslopes)

    # choose alpha per sector (use alpha_norm for neutrinos if provided)
    alpha = {}
    for sec in ['bosons','up','down','leptons','neutrinos']:
        sec_l = sec.lower()
        if sec_l=='neutrinos' and not np.isnan(a_norm.get(sec_l, np.nan)):
            alpha[sec_l] = float(a_norm[sec_l])
        else:
            alpha[sec_l] = float(a_raw.get(sec_l, np.nan))
        if np.isnan(alpha[sec_l]):
            alpha[sec_l] = 4.4 if sec_l!='bosons' else 3.4520009308762725

    # kNN d_s with optional median normalization
    nbrs, XY, vals = build_knn(D, k=max(1,args.nn_k))
    med = float(np.median(vals))
    def dfun(ax,ay):
        v = ds_lookup(ax,ay,nbrs,XY,vals,k=max(1,args.nn_k))
        return (v/med) if args.median_normalize_ds and med>0 else v

    # 1) Fit beta, lambda_b, gamma from W/Z/H
    beta, lambda_b, gamma = fit_beta_lambda_gamma_bosons(L, alpha['bosons'], dfun)
    print(f"[BOSONS] beta={beta:.6f}, lambda_b={lambda_b:.6f}, gamma={gamma:.6f}")

    # 2) Per-sector fits (lambda_s, Delta_s)
    sector_params = {'bosons': {'lambda': lambda_b, 'Delta': 0.0}}
    for sec in ['up','down','leptons']:
        Ls = L[L['sector'].str.lower()==sec].copy()
        lam_s, Delta_s = fit_sector_lambda_delta(Ls, alpha[sec], beta, lambda_b, gamma, dfun)
        sector_params[sec] = {'lambda': lam_s, 'Delta': Delta_s}
        print(f"[SECTOR {sec}] lambda_s={lam_s:.6f}, Delta_s={Delta_s:.6f}")
    # neutrinos: freeze or set trivial
    if args.freeze_neutrinos:
        sector_params['neutrinos'] = {'lambda': 0.0, 'Delta': 0.0}
    else:
        # very conservative default
        sector_params['neutrinos'] = {'lambda': 0.0, 'Delta': 0.0}

    # 3) Predict all
    rows=[]
    for _,r in L.iterrows():
        sp  = str(r['species'])
        sec = str(r['sector']).lower()
        z   = float(r['z_pred'])
        ax,ay = float(r['ax']), float(r['ay'])
        ds  = dfun(ax,ay)
        a   = alpha.get(sec, 4.4)
        lam = sector_params.get(sec, {'lambda':0.0})['lambda']
        Del = sector_params.get(sec, {'Delta':0.0})['Delta']
        logP= float(r['logm_PDG'])
        pdg = float(r['m_PDG_GeV'])

        logm = beta + gamma*z + lam*(a*ds*z) + Del
        mp   = float(np.exp(logm))
        if args.freeze_neutrinos and sec=='neutrinos':
            logm, mp = logP, pdg

        rows.append({
            'species': sp, 'sector': sec, 'ax': ax, 'ay': ay, 'z_pred': z,
            'ds_eff': ds, 'alpha_used': a,
            'beta': beta, 'gamma': gamma,
            'lambda_sector': lam, 'Delta_sector': Del,
            'm_PDG_GeV': pdg, 'm_pred_GeV': mp, 'logm_pred': logm,
            'abs_dlog': abs(logm-logP),
            'rel_err': (mp/pdg - 1.0) if pdg>0 else np.nan
        })
    OUT = pd.DataFrame(rows)

    # Summary
    med_dlog = float(OUT['abs_dlog'].median())
    mean_dlog= float(OUT['abs_dlog'].mean())
    finite   = OUT[np.isfinite(OUT['rel_err'])]
    med_pct  = float((finite['rel_err'].abs().median())*100) if not finite.empty else float('nan')
    mean_pct = float((finite['rel_err'].abs().mean())*100) if not finite.empty else float('nan')

    print("[SUMMARY]")
    print(f"  count = {len(OUT)}")
    print(f"  median |Δ log m| = {med_dlog:.4f}")
    print(f"  mean   |Δ log m| = {mean_dlog:.4f}")
    for sp in ['W','Z','H']:
        r0 = OUT[OUT['species'].str.lower()==sp.lower()]
        if not r0.empty:
            r0=r0.iloc[0]
            print(f"  {sp}: pred={r0['m_pred_GeV']:.6g}  PDG={r0['m_PDG_GeV']:.6g}  |Δlog|={r0['abs_dlog']:.4f}")
    print(f"  median % mass error = {med_pct:.2f}%")
    print(f"  mean   % mass error = {mean_pct:.2f}%")

    OUT.to_csv(args.outcsv, index=False)
    print(f"[WROTE] {args.outcsv}")

    # Plot d_s + species
    plt.figure(figsize=(8,6))
    sc = plt.scatter(D['ax'], D['ay'], c=D['ds'], cmap='viridis', s=50)
    plt.colorbar(sc, label='d_s (raw)')
    for _,r in L.iterrows():
        plt.text(r['ax'], r['ay'], str(r['species'])[0], ha='center', va='center',
                 fontsize=9, bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", alpha=0.6))
    plt.xlabel('a_x'); plt.ylabel('a_y'); plt.title('d_s(a_x,a_y) with species')
    plt.tight_layout(); plt.savefig(args.outpng, dpi=150)
    print(f"[PLOT] {args.outpng}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr); sys.exit(1)