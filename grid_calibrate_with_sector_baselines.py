#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibrate masses with a d_s(a_x,a_y) field and sector baselines Δ_s.

Model:
 log m_i = beta_fit + Δ_s + α_s * (λ * d_s(a_x,a_y) * z_i) + γ * z_i

- beta_fit is solved from W/Z/H by closed form (bosons only).
- Δ_s comes from sector_slopes.csv as: β_raw(s) - β_raw(bosons).
- α_s comes from sector_slopes.csv (α_raw by default; for neutrinos, α_norm if present).
- d_s is obtained by k-NN smoothing from ds_ax_ay_map.csv (or nearest).
- Supports both 'z_pred' and 'z_predi' column names in the locked CSV.
- Optionally freezes neutrino PDG masses.

Inputs expected:
 --locked        all_particles_locked.csv
                 columns: species, m_GeV, logm, z_target, ax, ay, z_predi (or z_pred), sector (can be missing)
 --dsmap         ds_ax_ay_map.csv (columns: ax, ay, ds)
 --sectorslopes  sector_slopes.csv (columns: sector, alpha_raw, beta_raw, [alpha_norm])
Outputs:
 --outcsv        written summary with predictions
 --outpng        ds field scatter with species labels
"""

import argparse
import math
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def read_locked(path_locked: str) -> pd.DataFrame:
   L = pd.read_csv(path_locked)
   # Normalize column names
   cols = {c: c.strip() for c in L.columns}
   L.rename(columns=cols, inplace=True)

   # Accept z_pred or z_predi
   if 'z_pred' not in L.columns and 'z_predi' in L.columns:
       L['z_pred'] = L['z_predi']
   if 'z_pred' not in L.columns:
       raise ValueError(f"[ERR] locked CSV missing 'z_pred' (or 'z_predi'). Columns={list(L.columns)}")

   # m_PDG_GeV column name (alias from m_GeV)
   if 'm_PDG_GeV' not in L.columns:
       if 'm_GeV' in L.columns:
           L['m_PDG_GeV'] = L['m_GeV']
       else:
           raise ValueError("[ERR] locked CSV must contain either 'm_PDG_GeV' or 'm_GeV'.")

   # Sector may be absent in some older exports; infer from species if needed
   if 'sector' not in L.columns:
       def infer_sector(sp):
           sp = str(sp).strip().lower()
           if sp in ('w', 'z', 'h'):
               return 'bosons'
           if sp.startswith('nu'):
               return 'neutrinos'
           if sp in ('u', 'c', 't'):
               return 'up'
           if sp in ('d', 's', 'b'):
               return 'down'
           if sp in ('e', 'mu', 'tau'):
               return 'leptons'
           return 'unknown'
       L['sector'] = L['species'].map(infer_sector)

   # Clean types
   for c in ['ax', 'ay', 'z_pred', 'm_PDG_GeV']:
       if c in L.columns:
           L[c] = pd.to_numeric(L[c], errors='coerce')

   # drop rows with missing essentials
   L = L.dropna(subset=['species', 'ax', 'ay', 'z_pred', 'm_PDG_GeV', 'sector']).copy()

   # natural log of PDG mass (for diagnostics)
   L['logm_PDG'] = np.log(L['m_PDG_GeV'])
   return L


def read_dsmap(path_ds: str) -> pd.DataFrame:
   D = pd.read_csv(path_ds)
   for c in ['ax', 'ay', 'ds']:
       if c not in D.columns:
           raise ValueError(f"[ERR] dsmap must contain columns ax, ay, ds. Got {list(D.columns)}")
   D[['ax','ay','ds']] = D[['ax','ay','ds']].apply(pd.to_numeric, errors='coerce')
   D = D.dropna(subset=['ax','ay','ds']).copy()
   return D


def build_knn_ds(dsmap: pd.DataFrame, k: int = 3):
   XY = dsmap[['ax','ay']].values
   nbrs = NearestNeighbors(n_neighbors=min(k, len(XY)), algorithm='auto').fit(XY)
   return nbrs, XY, dsmap['ds'].values


def lookup_ds(ax, ay, nbrs, XY, dsvals, k=3):
   # average ds of k nearest neighbors
   distances, indices = nbrs.kneighbors([[ax, ay]], n_neighbors=min(k, len(XY)), return_distance=True)
   idxs = indices[0]
   vals = dsvals[idxs]
   return float(np.mean(vals))


def read_sector_slopes(path_slopes: str):
   S = pd.read_csv(path_slopes)
   # normalize names
   rename = {c: c.strip() for c in S.columns}
   S.rename(columns=rename, inplace=True)

   req = ['sector', 'alpha_raw', 'beta_raw']
   for r in req:
       if r not in S.columns:
           raise ValueError(f"[ERR] sector_slopes missing column '{r}'. Columns={list(S.columns)}")

   # optional normalized alpha for neutrinos
   if 'alpha_norm' not in S.columns:
       S['alpha_norm'] = np.nan

   # build dictionaries
   a_raw = {}
   a_norm = {}
   b_raw = {}
   for _, row in S.iterrows():
       sec = str(row['sector']).strip().lower()
       a_raw[sec] = float(row['alpha_raw']) if pd.notnull(row['alpha_raw']) else np.nan
       b_raw[sec] = float(row['beta_raw']) if pd.notnull(row['beta_raw']) else np.nan
       if pd.notnull(row['alpha_norm']):
           a_norm[sec] = float(row['alpha_norm'])
       else:
           a_norm[sec] = np.nan

   # baseline β is bosons β_raw
   if 'bosons' not in b_raw or pd.isna(b_raw['bosons']):
       raise ValueError("[ERR] sector_slopes must provide bosons.beta_raw.")

   beta_boson_raw = b_raw['bosons']
   # Δ_s = β_raw(s) - β_raw(bosons)
   delta_s = {sec: (b_raw[sec] - beta_boson_raw) if pd.notnull(b_raw.get(sec, np.nan)) else 0.0
              for sec in b_raw.keys()}

   # choose α_s: use alpha_norm for neutrinos if present and reasonable, else alpha_raw
   alpha_s = {}
   for sec in a_raw.keys():
       if sec == 'neutrinos' and pd.notnull(a_norm.get(sec, np.nan)):
           alpha_s[sec] = a_norm[sec]
       else:
           alpha_s[sec] = a_raw[sec]

   return alpha_s, delta_s


def solve_beta_bosons(L: pd.DataFrame, alpha_s: dict, delta_s: dict, lambda_ds: float, gamma: float, ds_lookup):
   # Use only W/Z/H rows to solve beta in closed form
   B = L[L['sector'].str.lower() == 'bosons'].copy()
   if B.empty:
       raise ValueError("[ERR] no bosons in locked CSV to fit β.")
   # compute effective (α_s * λ * d_s * z + γ z + Δ_s)
   rhs_terms = []
   logs = []
   for _, r in B.iterrows():
       sec = 'bosons'
       z = float(r['z_pred'])
       ds_eff = ds_lookup(float(r['ax']), float(r['ay']))
       alpha = float(alpha_s.get(sec))
       delta = float(delta_s.get(sec, 0.0))
       # contribute
       term = delta + alpha * (lambda_ds * ds_eff * z) + gamma * z
       rhs_terms.append(term)
       logs.append(float(r['logm_PDG']))

   rhs_terms = np.array(rhs_terms)
   logs = np.array(logs)
   beta = float(np.mean(logs - rhs_terms))
   return beta


def main():
   ap = argparse.ArgumentParser(description="Calibrate masses with d_s map + sector baselines Δ_s.")
   ap.add_argument("--locked", required=True, help="all_particles_locked.csv")
   ap.add_argument("--dsmap", required=True, help="ds_ax_ay_map.csv")
   ap.add_argument("--sectorslopes", required=True, help="sector_slopes.csv")
   ap.add_argument("--outcsv", required=True, help="output CSV path")
   ap.add_argument("--outpng", required=True, help="output PNG plot path")
   ap.add_argument("--lambda_ds", type=float, default=0.30, help="λ scaling for d_s * z")
   ap.add_argument("--gamma", type=float, default=0.0, help="γ coefficient for z")
   ap.add_argument("--nn_k", type=int, default=3, help="k for kNN smoothing of d_s")
   ap.add_argument("--freeze_neutrinos", action="store_true", help="freeze neutrino PDG masses")
   args = ap.parse_args()

   print("[ARGS]", vars(args))

   # Load data
   L = read_locked(args.locked)
   D = read_dsmap(args.dsmap)
   alpha_s, delta_s = read_sector_slopes(args.sectorslopes)

   # kNN for d_s
   nbrs, XY, dsvals = build_knn_ds(D, k=max(1, args.nn_k))
   def ds_lookup(ax, ay):
       return lookup_ds(ax, ay, nbrs, XY, dsvals, k=max(1, args.nn_k))

   # Solve β on bosons (W/Z/H)
   beta_fit = solve_beta_bosons(
       L, alpha_s=alpha_s, delta_s=delta_s,
       lambda_ds=args.lambda_ds, gamma=args.gamma,
       ds_lookup=ds_lookup
   )
   print(f"[FIT] beta (from W/Z/H) = {beta_fit:.6f}")

   # Predict all
   out_rows = []
   for _, r in L.iterrows():
       sp = str(r['species']).strip()
       sec = str(r['sector']).strip().lower()
       z = float(r['z_pred'])
       ax = float(r['ax']); ay = float(r['ay'])
       m_pdg = float(r['m_PDG_GeV'])
       logm_pdg = float(r['logm_PDG'])

       ds_eff = ds_lookup(ax, ay)
       alpha = float(alpha_s.get(sec, np.nan))
       delta = float(delta_s.get(sec, 0.0))

       if np.isnan(alpha):
           # if unknown sector, fall back to leptons slope (conservative)
           alpha = float(alpha_s.get('leptons', 4.4))

       logm_pred = beta_fit + delta + alpha * (args.lambda_ds * ds_eff * z) + args.gamma * z
       m_pred = float(np.exp(logm_pred))

       if args.freeze_neutrinos and sec == 'neutrinos':
           # Use PDG neutrino mass (from input) if provided; otherwise keep predicted
           m_pred = m_pdg
           logm_pred = logm_pdg

       out_rows.append({
           'species': sp,
           'sector': sec,
           'ax': ax,
           'ay': ay,
           'z_pred': z,
           'ds_eff': ds_eff,
           'alpha_used': alpha,
           'delta_sector': delta,
           'beta_fit': beta_fit,
           'lambda_ds': args.lambda_ds,
           'gamma': args.gamma,
           'm_PDG_GeV': m_pdg,
           'm_pred_GeV': m_pred,
           'logm_pred': logm_pred,
           'abs_dlog': abs(logm_pred - logm_pdg),
           'rel_err': (m_pred / m_pdg - 1.0) if m_pdg > 0 else np.nan
       })

   OUT = pd.DataFrame(out_rows)

   # Diagnostics
   def fmt_pct(x):
       if pd.isna(x): return "nan"
       return f"{100*x:,.2f}%"

   median_dlog = float(OUT['abs_dlog'].median())
   mean_dlog   = float(OUT['abs_dlog'].mean())
   # Percentage mass error median/mean on finite entries
   finite = OUT[np.isfinite(OUT['rel_err'])]
   med_pct = float((finite['rel_err'].abs().median())*100.0) if not finite.empty else float('nan')
   mean_pct = float((finite['rel_err'].abs().mean())*100.0) if not finite.empty else float('nan')

   print("[SUMMARY]")
   print(f"  count = {len(OUT)}")
   print(f"  median |Δ log m| = {median_dlog:.4f}")
   print(f"  mean   |Δ log m| = {mean_dlog:.4f}")
   print(f"  median % mass error = {med_pct:.2f}%")
   print(f"  mean   % mass error = {mean_pct:.2f}%")

   # Show W/Z/H for sanity
   for sp in ['W','Z','H']:
       row = OUT[OUT['species'].str.lower()==sp.lower()]
       if not row.empty:
           r0 = row.iloc[0]
           print(f"  {sp}: pred={r0['m_pred_GeV']:.6g}, PDG={r0['m_PDG_GeV']:.6g}, |Δlog|={r0['abs_dlog']:.4f}")

   # Write CSV
   OUT.to_csv(args.outcsv, index=False)
   print(f"[WROTE] {args.outcsv}")

   # Plot ds field with species labels
   plt.figure(figsize=(8, 6))
   sc = plt.scatter(D['ax'], D['ay'], c=D['ds'], s=50, cmap='viridis')
   plt.colorbar(sc, label='d_s')
   for _, r in L.iterrows():
       plt.text(r['ax'], r['ay'], str(r['species'])[0], ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", alpha=0.6))
   plt.xlabel('a_x'); plt.ylabel('a_y')
   plt.title('d_s(a_x,a_y) with species')
   plt.tight_layout()
   plt.savefig(args.outpng, dpi=150)
   print(f"[PLOT] {args.outpng}")


if __name__ == "__main__":
   try:
       main()
   except Exception as e:
       print(f"[FATAL] {e}", file=sys.stderr)
       sys.exit(1)
