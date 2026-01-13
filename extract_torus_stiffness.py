#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_torus_stiffness.py
Compute local Hessians and "mode stiffness" proxies from torus CSVs.

Inputs:
  - latent z grid:  latent_z_merged2.csv   (columns ~ ax, ay, z)
  - locked minima:  all_particles_locked.csv (columns ~ species, ax, ay, z_pred|z)
  - sector slopes:  sector_slopes.csv (optional; columns ~ sector, alpha, beta)

Outputs:
  - CSV with per-species stiffness diagnostics:
      species, ax, ay,
      Huu, Huv, Hvv, kappa_min, kappa_max, lambda_vec, lambda_H, lambda_ax, lambda_rho,
      alpha_eff (if slopes given), m2_H_proxy, m2_ax_proxy, m2_vec_proxy, m2_rho_proxy
"""

import argparse
import os
import sys
import math
import json
import numpy as np
import pandas as pd

# ---------- Utilities ----------

def _norm_cols(df, wanted):
    """Try to normalize column names to a target set (case-insensitive, underscores ignored)."""
    def key(s): return ''.join(ch for ch in str(s).lower() if ch.isalnum())
    m = {key(c): c for c in df.columns}
    out = {}
    for w in wanted:
        k = key(w)
        if k in m:
            out[w] = m[k]
        else:
            # try some common aliases
            aliases = {
                'ax': ['a_x','alpha_x','alphax','x','u'],
                'ay': ['a_y','alpha_y','alphay','y','v'],
                'z':  ['z_pred','latent','zval'],
                'species': ['name','label'],
                'sector': ['family','band'],
                'alpha':  ['slope','k'],
                'beta':   ['intercept','b0'],
            }.get(w, [])
            hit = None
            for a in aliases:
                ka = key(a)
                if ka in m:
                    hit = m[ka]; break
            if hit is None:
                raise KeyError(f"Required column '{w}' not found (aliases tried: {aliases}). "
                               f"Available: {list(df.columns)}")
            out[w] = hit
    return out

def load_latent(latent_path):
    df = pd.read_csv(latent_path)
    cols = _norm_cols(df, ['ax','ay','z'])
    g = df[[cols['ax'], cols['ay'], cols['z']]].copy()
    g.columns = ['ax','ay','z']
    # sort & unique axes
    ax_vals = np.unique(g['ax'].values)
    ay_vals = np.unique(g['ay'].values)
    # attempt to reshape as a grid; if irregular we’ll just use neighbor stencils
    return g, ax_vals, ay_vals

def bilinear_interpolate(g, ax_vals, ay_vals, ax0, ay0):
    """Bilinear interpolation of z at (ax0,ay0) from scattered grid g assumed on rect grid."""
    # find surrounding grid nodes
    axu = ax_vals
    ayu = ay_vals
    if not (axu.min() <= ax0 <= axu.max() and ayu.min() <= ay0 <= ayu.max()):
        return np.nan, False
    ix = np.searchsorted(axu, ax0) - 1
    iy = np.searchsorted(ayu, ay0) - 1
    ix = max(0, min(ix, len(axu)-2))
    iy = max(0, min(iy, len(ayu)-2))
    x0,x1 = axu[ix], axu[ix+1]
    y0,y1 = ayu[iy], ayu[iy+1]
    # get z’s at corners
    # To be robust to duplicates, take mean over repeated points
    def z_at(x,y):
        sub = g[(np.isclose(g.ax,x))&(np.isclose(g.ay,y))]['z']
        if len(sub)==0: return np.nan
        return sub.mean()
    z00 = z_at(x0,y0); z10 = z_at(x1,y0); z01 = z_at(x0,y1); z11 = z_at(x1,y1)
    if any(np.isnan([z00,z10,z01,z11])):
        return np.nan, False
    # weights
    tx = 0. if x1==x0 else (ax0-x0)/(x1-x0)
    ty = 0. if y1==y0 else (ay0-y0)/(y1-y0)
    z0 = (1-tx)*(1-ty)*z00 + tx*(1-ty)*z10 + (1-tx)*ty*z01 + tx*ty*z11
    return float(z0), True

def local_quadratic_fit(g, ax0, ay0, stencil=5, ax_vals=None, ay_vals=None, tol=1e-6):
    """
    Fit f(u,v) ≈ c0 + c1 du + c2 dv + c3 du^2 + c4 du*dv + c5 dv^2
    then Hessian entries are: Huu=2*c3, Huv=c4, Hvv=2*c5
    We’ll gather a (stencil x stencil) window centered on nearest grid node.
    """
    if stencil % 2 == 0: stencil += 1
    # If rect grid known, snap to nearest indices & slice; else kNN fallback
    if ax_vals is not None and ay_vals is not None:
        ix = np.argmin(np.abs(ax_vals - ax0))
        iy = np.argmin(np.abs(ay_vals - ay0))
        half = stencil//2
        xs = ax_vals[max(0,ix-half): min(len(ax_vals), ix+half+1)]
        ys = ay_vals[max(0,iy-half): min(len(ay_vals), iy+half+1)]
        # collect points
        rows = []
        for x in xs:
            for y in ys:
                sub = g[(np.isclose(g.ax,x))&(np.isclose(g.ay,y))]
                if len(sub)==0: continue
                z = sub['z'].mean()
                rows.append((x,y,z))
        if len(rows) < 6:  # too few for quadratic
            return None, False
        P = np.array([[r[0], r[1], r[2]] for r in rows])
    else:
        # fallback: nearest N points in a radius
        arr = g[['ax','ay','z']].values
        # sort by distance
        d = (arr[:,0]-ax0)**2 + (arr[:,1]-ay0)**2
        idx = np.argsort(d)[:max(6, stencil*stencil)]
        P = arr[idx]

    # Build regression
    du = P[:,0] - ax0
    dv = P[:,1] - ay0
    A = np.column_stack([
        np.ones_like(du),
        du, dv,
        du*du, du*dv, dv*dv
    ])
    y = P[:,2]
    # least squares
    try:
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception:
        return None, False

    Huu = 2.0*c[3]
    Huv = c[4]
    Hvv = 2.0*c[5]
    H = np.array([[Huu, Huv],[Huv, Hvv]], dtype=float)
    # eigenvalues
    w, _ = np.linalg.eigh(H)
    kmin, kmax = float(w[0]), float(w[1])
    return dict(Huu=Huu, Huv=Huv, Hvv=Hvv, kmin=kmin, kmax=kmax), True

def dilaton_curvature(g, ax_vals, ay_vals, ax0, ay0, drho=0.01):
    """Estimate ∂^2 z / ∂ρ^2 along v = ρ u, at ρ0 = ay0/ax0, with u fixed ≈ ax0 via bilinear interpolation."""
    if ax0 == 0.0: return np.nan, False
    rho0 = ay0/ax0
    rhos = [rho0 - drho, rho0, rho0 + drho]
    zs, ok = [], True
    for r in rhos:
        ay = r*ax0
        z, hit = bilinear_interpolate(g, ax_vals, ay_vals, ax0, ay)
        if not hit: ok = False
        zs.append(z)
    if (not ok) or any(np.isnan(zs)): return np.nan, False
    # second derivative via central difference: f''(rho0) ~ (f(r+)-2f(0)+f(r-))/drho^2
    fpp = (zs[2] - 2*zs[1] + zs[0])/(drho**2)
    return float(fpp), True

def load_locked(locked_path):
    df = pd.read_csv(locked_path)
    # try to find species, ax, ay (and z if present)
    cols = _norm_cols(df, ['species','ax','ay'])
    L = df[[cols['species'], cols['ax'], cols['ay']]].copy()
    L.columns = ['species','ax','ay']
    # optional z value
    zcol = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in ('z','z_pred','latent'):
            zcol = c; break
    if zcol is not None:
        L['z'] = df[zcol].values
    else:
        L['z'] = np.nan
    return L

def load_slopes(slopes_path):
    if slopes_path is None or (not os.path.exists(slopes_path)):
        return None
    df = pd.read_csv(slopes_path)
    cols = _norm_cols(df, ['sector','alpha','beta'])
    S = df[[cols['sector'], cols['alpha'], cols['beta']]].copy()
    S.columns = ['sector','alpha','beta']
    # normalize sector names
    S['sector'] = S['sector'].astype(str).str.strip().str.lower()
    return {row['sector']: (float(row['alpha']), float(row['beta'])) for _,row in S.iterrows()}

def species_to_sector(name):
    n = str(name).strip().lower()
    if n in ('e','mu','tau','electron','muon','tauon'): return 'leptons'
    if n in ('u','c','t','up','charm','top'):          return 'up'
    if n in ('d','s','b','down','strange','bottom'):   return 'down'
    if n in ('h','higgs','w','z','boson'):             return 'bosons'
    if n.startswith('nu'):                              return 'neutrinos'
    return 'unknown'

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Extract torus-mode stiffness proxies from CSVs.")
    ap.add_argument('--latent', required=True, help="Path to latent_z_merged2.csv")
    ap.add_argument('--locked', required=True, help="Path to all_particles_locked.csv")
    ap.add_argument('--slopes', default=None, help="Optional path to sector_slopes.csv (alpha,beta per sector)")
    ap.add_argument('--outcsv', required=True, help="Output CSV path")
    ap.add_argument('--stencil', type=int, default=5, help="Odd stencil size for quadratic fit (default 5)")
    ap.add_argument('--rho_eps', type=float, default=0.01, help="Delta-rho for dilaton curvature (default 0.01)")
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    # Load
    if not args.quiet: print(f"[LOAD] latent: {args.latent}")
    g, ax_vals, ay_vals = load_latent(args.latent)
    if not args.quiet: print(f"[INFO] latent grid size: N={len(g)} ; unique ax={len(ax_vals)} ; unique ay={len(ay_vals)}")
    if not args.quiet: print(f"[LOAD] locked: {args.locked}")
    L = load_locked(args.locked)
    slopes = load_slopes(args.slopes)
    if slopes is None:
        if not args.quiet: print("[WARN] No sector slopes provided; mass proxies will be NaN.")
    else:
        if not args.quiet: print(f"[INFO] sector slopes: {json.dumps(slopes)}")

    rows = []
    for _,r in L.iterrows():
        species = r['species']
        ax0, ay0 = float(r['ax']), float(r['ay'])
        # Hessian via local quadratic fit
        fit, ok = local_quadratic_fit(g, ax0, ay0, stencil=args.stencil, ax_vals=ax_vals, ay_vals=ay_vals)
        if not ok:
            if not args.quiet: print(f"[WARN] quadratic fit failed at {species} ({ax0:.5f},{ay0:.5f})")
            fit = {'Huu':np.nan,'Huv':np.nan,'Hvv':np.nan,'kmin':np.nan,'kmax':np.nan}
        Huu, Huv, Hvv = fit['Huu'], fit['Huv'], fit['Hvv']
        kmin, kmax = fit['kmin'], fit['kmax']
        # proxies
        lambda_vec = (Huu + Hvv)  # trace
        lambda_H   = kmin         # min eigenvalue: “Higgs-like” curvature
        lambda_ax  = abs(Huv)     # mixed curvature magnitude: “axion-like” proxy
        # dilaton curvature along rho line
        l_rho, ok_rho = dilaton_curvature(g, ax_vals, ay_vals, ax0, ay0, drho=args.rho_eps)
        lambda_rho = l_rho if ok_rho else np.nan

        # sector slope and mass proxies
        sector = species_to_sector(species)
        alpha_eff = np.nan
        if slopes is not None and sector in slopes:
            alpha_eff = float(slopes[sector][0])
        def m2proxy(lmbd):
            if np.isnan(alpha_eff) or np.isnan(lmbd): return np.nan
            return (alpha_eff**2) * lmbd

        rows.append({
            'species': species,
            'ax': ax0, 'ay': ay0,
            'Huu': Huu, 'Huv': Huv, 'Hvv': Hvv,
            'kappa_min': kmin, 'kappa_max': kmax,
            'lambda_vec_trace': lambda_vec,
            'lambda_H_minEV': lambda_H,
            'lambda_ax_absHuv': lambda_ax,
            'lambda_rho_d2zdrho2': lambda_rho,
            'sector': sector,
            'alpha_eff': alpha_eff,
            'm2_H_proxy': m2proxy(lambda_H),
            'm2_ax_proxy': m2proxy(lambda_ax),
            'm2_vec_proxy': m2proxy(lambda_vec),
            'm2_rho_proxy': m2proxy(lambda_rho)
        })

    out = pd.DataFrame(rows)
    out.sort_values(['sector','species'], inplace=True, ignore_index=True)
    os.makedirs(os.path.dirname(args.outcsv), exist_ok=True)
    out.to_csv(args.outcsv, index=False)
    if not args.quiet:
        print(f"[DONE] Wrote: {args.outcsv}")
        # quick summary
        with np.printoptions(precision=6, suppress=True):
            print(out[['species','lambda_H_minEV','lambda_ax_absHuv','lambda_vec_trace','lambda_rho_d2zdrho2']].to_string(index=False))


if __name__ == "__main__":
    main()