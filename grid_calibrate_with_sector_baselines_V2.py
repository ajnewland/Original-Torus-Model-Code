#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
β pinned by W/Z/H; sector baselines Δ_s solved from fermion PDG (least squares, one scalar per sector).
d_s is kNN-smoothed and median-normalized. Supports z_pred or z_predi. Can freeze neutrinos.

Inputs:
  --locked, --dsmap, --sectorslopes
Outputs:
  --outcsv, --outpng
"""

import argparse, sys, math
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# ---------- IO helpers ----------
def read_locked(path):
    L = pd.read_csv(path)
    if 'z_pred' not in L.columns and 'z_predi' in L.columns:
        L['z_pred'] = L['z_predi']
    if 'z_pred' not in L.columns:
        raise ValueError("locked CSV must contain z_pred or z_predi")
    if 'm_PDG_GeV' not in L.columns:
        if 'm_GeV' in L.columns: L['m_PDG_GeV'] = L['m_GeV']
        else: raise ValueError("locked CSV must contain m_PDG_GeV or m_GeV")
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
    L['logm_PDG'] = np.log(L['m_PDG_GeV'].astype(float))
    for c in ['ax','ay','z_pred']: L[c]=pd.to_numeric(L[c], errors='coerce')
    L = L.dropna(subset=['species','sector','ax','ay','z_pred','m_PDG_GeV','logm_PDG']).copy()
    return L

def read_dsmap(path):
    D = pd.read_csv(path)
    for c in ['ax','ay','ds']:
        if c not in D.columns: raise ValueError("dsmap must have ax, ay, ds")
    D[['ax','ay','ds']]=D[['ax','ay','ds']].apply(pd.to_numeric, errors='coerce')
    return D.dropna()

def read_slopes(path):
    S = pd.read_csv(path)
    for c in ['sector','alpha_raw','beta_raw']:
        if c not in S.columns: raise ValueError("sector_slopes needs sector, alpha_raw, beta_raw")
    if 'alpha_norm' not in S.columns: S['alpha_norm']=np.nan
    a_raw = {str(r.sector).lower(): float(r.alpha_raw) for _,r in S.iterrows()}
    a_norm= {str(r.sector).lower(): (float(r.alpha_norm) if pd.notnull(r.alpha_norm) else np.nan)
             for _,r in S.iterrows()}
    return a_raw, a_norm

# ---------- d_s kNN ----------
def build_knn(D, k):
    XY = D[['ax','ay']].values
    nbrs = NearestNeighbors(n_neighbors=min(k,len(XY))).fit(XY)
    return nbrs, XY, D['ds'].values

def ds_lookup(ax,ay,nbrs,XY,vals,k):
    dist, idx = nbrs.kneighbors([[ax,ay]], n_neighbors=min(k,len(XY)), return_distance=True)
    return float(np.mean(vals[idx[0]]))

# ---------- core ----------
def fit_beta_bosons(L, alpha_s, lam, gamma, dfun):
    B = L[L['sector'].str.lower()=='bosons']
    if B.empty: raise ValueError("no bosons to pin β")
    terms = []
    logs  = []
    for _,r in B.iterrows():
        z   = float(r['z_pred'])
        ds  = dfun(float(r['ax']), float(r['ay']))
        a   = float(alpha_s['bosons'])
        terms.append(a*(lam*ds*z) + gamma*z)
        logs.append(float(r['logm_PDG']))
    return float(np.mean(np.array(logs) - np.array(terms)))

def solve_sector_offsets(L, alpha_s, beta_fit, lam, gamma, dfun, freeze_neutrinos):
    # Δ_s per sector (excluding bosons; neutrinos optionally skipped)
    residuals = {}
    counts    = {}
    for _,r in L.iterrows():
        sec = str(r['sector']).lower()
        if sec=='bosons':   continue
        if freeze_neutrinos and sec=='neutrinos': continue
        a   = float(alpha_s.get(sec, alpha_s.get('leptons',4.4)))
        z   = float(r['z_pred'])
        ds  = dfun(float(r['ax']), float(r['ay']))
        target = float(r['logm_PDG'])
        base   = beta_fit + a*(lam*ds*z) + gamma*z
        res    = target - base
        residuals.setdefault(sec, 0.0); counts.setdefault(sec, 0)
        residuals[sec] += res; counts[sec] += 1
    # mean residual per sector
    delta = {sec: (residuals[sec]/counts[sec]) for sec in residuals.keys() if counts[sec]>0}
    # ensure all sectors have a value (0 for bosons; 0 if truly absent)
    out = {'bosons': 0.0}
    for sec in ['up','down','leptons','neutrinos']:
        out[sec] = delta.get(sec, 0.0)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", required=True)
    ap.add_argument("--dsmap", required=True)
    ap.add_argument("--sectorslopes", required=True)
    ap.add_argument("--outcsv", required=True)
    ap.add_argument("--outpng", required=True)
    ap.add_argument("--lambda_ds", type=float, default=0.30)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--nn_k", type=int, default=3)
    ap.add_argument("--freeze_neutrinos", action="store_true")
    args = ap.parse_args()
    print("[ARGS]", vars(args))

    L = read_locked(args.locked)
    D = read_dsmap(args.dsmap)
    a_raw, a_norm = read_slopes(args.sectorslopes)

    # choose α_s (use alpha_norm for neutrinos if present, else alpha_raw)
    alpha_s = {}
    for sec, aval in a_raw.items():
        if sec=='neutrinos' and not np.isnan(a_norm.get(sec, np.nan)):
            alpha_s[sec] = float(a_norm[sec])
        else:
            alpha_s[sec] = float(aval)
    # sanity defaults
    for sec in ['bosons','up','down','leptons','neutrinos']:
        if sec not in alpha_s or np.isnan(alpha_s[sec]):
            alpha_s[sec] = 4.4 if sec!='bosons' else 3.45

    # kNN d_s; median-normalize to tame scale
    nbrs, XY, vals = build_knn(D, k=max(1,args.nn_k))
    med = float(np.median(vals))
    def dfun(ax,ay):
        v = ds_lookup(ax,ay,nbrs,XY,vals,k=max(1,args.nn_k))
        return v/med if med>0 else v

    # 1) pin β on W/Z/H
    beta_fit = fit_beta_bosons(L, alpha_s, args.lambda_ds, args.gamma, dfun)
    print(f"[FIT] beta (W/Z/H) = {beta_fit:.6f}")

    # 2) solve Δ_s from fermion PDG
    delta_s = solve_sector_offsets(L, alpha_s, beta_fit, args.lambda_ds, args.gamma, dfun, args.freeze_neutrinos)
    print("[Δ_s] sector baselines:", {k: round(v,6) for k,v in delta_s.items()})

    # 3) predict
    rows=[]
    for _,r in L.iterrows():
        sp  = str(r['species'])
        sec = str(r['sector']).lower()
        z   = float(r['z_pred'])
        ax,ay = float(r['ax']), float(r['ay'])
        pdg = float(r['m_PDG_GeV'])
        logp= float(r['logm_PDG'])
        ds  = dfun(ax,ay)
        a   = float(alpha_s.get(sec, alpha_s['leptons']))
        delta = float(delta_s.get(sec, 0.0))
        logm = beta_fit + delta + a*(args.lambda_ds*ds*z) + args.gamma*z
        mp   = float(np.exp(logm))
        if args.freeze_neutrinos and sec=='neutrinos':
            mp = pdg; logm = logp
        rows.append({
            'species': sp, 'sector': sec, 'ax': ax, 'ay': ay,
            'z_pred': z, 'ds_eff': ds, 'alpha_used': a,
            'delta_sector': delta, 'beta_fit': beta_fit,
            'lambda_ds': args.lambda_ds, 'gamma': args.gamma,
            'm_PDG_GeV': pdg, 'm_pred_GeV': mp, 'logm_pred': logm,
            'abs_dlog': abs(logm - logp),
            'rel_err': (mp/pdg - 1.0) if pdg>0 else np.nan
        })
    OUT = pd.DataFrame(rows)

    # summary
    med_dlog = float(OUT['abs_dlog'].median())
    mean_dlog= float(OUT['abs_dlog'].mean())
    finite   = OUT[np.isfinite(OUT['rel_err'])]
    med_pct  = float((finite['rel_err'].abs().median())*100) if not finite.empty else float('nan')
    mean_pct = float((finite['rel_err'].abs().mean())*100) if not finite.empty else float('nan')
    print("[SUMMARY]")
    print(f"  count = {len(OUT)}")
    print(f"  median |Δ log m| = {med_dlog:.4f}")
    print(f"  mean   |Δ log m| = {mean_dlog:.4f}")
    print(f"  median % mass error = {med_pct:.2f}%")
    print(f"  mean   % mass error = {mean_pct:.2f}%")
    for sp in ['W','Z','H']:
        r0 = OUT[OUT['species'].str.lower()==sp.lower()]
        if not r0.empty:
            r0=r0.iloc[0]
            print(f"  {sp}: pred={r0['m_pred_GeV']:.6g}  PDG={r0['m_PDG_GeV']:.6g}  |Δlog|={r0['abs_dlog']:.4f}")

    OUT.to_csv(args.outcsv, index=False)
    print(f"[WROTE] {args.outcsv}")

    # plot d_s field with species
    plt.figure(figsize=(8,6))
    sc = plt.scatter(D['ax'], D['ay'], c=D['ds']/med if med>0 else D['ds'], cmap='viridis', s=50)
    plt.colorbar(sc, label='d_s (median-normalized)')
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