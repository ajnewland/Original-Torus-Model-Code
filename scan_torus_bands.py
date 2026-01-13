#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
scan_torus_bands.py
Grid + adaptive torus scanner with band labeling and per-band engine stats.

Bands:
  Mode A: rectangles CSV with columns:
      ax_min,ax_max,ay_min,ay_max,label
  Mode B: seeds CSV with columns:
      label,species,ax,ay
    → nearest-neighbour (Voronoi) labeling by Euclidean distance in (ax,ay).

Outputs:
  - full_scan_samples.csv    (ax,ay,z,grad,DeltaF,w,C1,C2,band,level)
  - candidate_minima.csv     (raw minima, unfiltered, +band)
  - locked_minima.csv        (after separation +band)
  - band_averages.json       (per-band ⟨C1^2⟩, ⟨C2⟩, counts for samples & minima)

Usage example:
python scan_torus_bands.py \
  --latent geom_fit_many/latent_z_merged2.csv \
  --ax-min 2.45 --ax-max 2.70 --ax-steps 501 \
  --ay-min 0.70 --ay-max 0.95 --ay-steps 501 \
  --beta 0.5529 --gamma0 0.0 --gamma1 1.0 \
  --min-sep-ax 0.004 --min-sep-ay 0.004 --min-sep-z 1e-3 \
  --refine-rounds 2 --refine-halfwidth 0.01 --refine-steps 81 \
  --bands-rect "bands_rectangles.csv" \
  --outdir "Predicted Masses/torus_scan_bands"

OR with seeds:
  --bands-seed "bands_seeds.csv"
"""

import argparse, os, json, csv, math
import numpy as np
import pandas as pd

# ----------------------------- Fits -----------------------------

def fit_quadratic(ax, ay, z):
    X = np.column_stack([np.ones_like(ax), ax, ay, ax**2, ax*ay, ay**2])
    c, *_ = np.linalg.lstsq(X, z, rcond=None)
    def zhat(axv, ayv):
        axv, ayv = np.asarray(axv), np.asarray(ayv)
        return (c[0] + c[1]*axv + c[2]*ayv + c[3]*axv**2 + c[4]*axv*ayv + c[5]*ayv**2)
    return c, zhat

def finite_diff_grad(Z, dx, dy):
    Ny, Nx = Z.shape
    dZdx = np.zeros_like(Z); dZdy = np.zeros_like(Z)
    dZdx[:, 1:-1] = (Z[:, 2:] - Z[:, :-2])/(2*dx)
    dZdx[:, 0]    = (Z[:, 1] - Z[:, 0])/dx
    dZdx[:, -1]   = (Z[:, -1] - Z[:, -2])/dx
    dZdy[1:-1, :] = (Z[2:, :] - Z[:-2, :])/(2*dy)
    dZdy[0, :]    = (Z[1, :] - Z[0, :])/dy
    dZdy[-1, :]   = (Z[-1, :] - Z[-2, :])/dy
    return np.sqrt(dZdx**2 + dZdy**2)

def logistic_from_deltaF(deltaF, beta):
    w = 1.0/(1.0 + np.exp(-beta*deltaF))
    c1 = np.tanh(0.5*beta*deltaF)
    c2 = 4.0*w*(1.0 - w)
    return w, c1, c2

def local_minima(Z):
    Ny, Nx = Z.shape
    mins = []
    mask = ~np.eye(3, dtype=bool)
    for iy in range(1, Ny-1):
        for ix in range(1, Nx-1):
            v = Z[iy, ix]
            nb = Z[iy-1:iy+2, ix-1:ix+2]
            if np.all(v < nb[mask]):
                mins.append((iy, ix))
    return mins

def separation_filter(points, min_sep_ax, min_sep_ay, min_sep_z):
    pts = sorted(points, key=lambda d: d['z'])
    kept = []
    for p in pts:
        ok = True
        for q in kept:
            if (abs(p['ax']-q['ax']) < min_sep_ax and
                abs(p['ay']-q['ay']) < min_sep_ay and
                abs(p['z'] -q['z'])  < min_sep_z):
                ok = False; break
        if ok: kept.append(p)
    return kept

# ----------------------------- Bands -----------------------------

def load_bands_rectangles(path):
    rects = []
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        rects.append({
            'ax_min': float(r['ax_min']), 'ax_max': float(r['ax_max']),
            'ay_min': float(r['ay_min']), 'ay_max': float(r['ay_max']),
            'label': str(r['label'])
        })
    return rects

def load_bands_seeds(path):
    seeds = []
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        seeds.append({'label': str(r['label']),
                      'species': str(r.get('species','')),
                      'ax': float(r['ax']), 'ay': float(r['ay'])})
    return seeds

def label_point(ax, ay, rects=None, seeds=None):
    # Priority: rectangles, then seeds; if neither, return "unlabeled"
    if rects:
        for R in rects:
            if (R['ax_min'] <= ax <= R['ax_max']) and (R['ay_min'] <= ay <= R['ay_max']):
                return R['label']
    if seeds:
        best_label, best_d2 = None, float('inf')
        for s in seeds:
            d2 = (ax - s['ax'])**2 + (ay - s['ay'])**2
            if d2 < best_d2:
                best_d2 = d2; best_label = s['label']
        return best_label
    return 'unlabeled'

def aggregate_band_stats(rows, key='samples'):
    # rows: iterable of dicts with 'band','C1','C2'
    stats = {}
    for r in rows:
        b = r.get('band','unlabeled')
        s = stats.setdefault(b, {'count':0, 'sum_C2':0.0, 'sum_C1sq':0.0})
        s['count'] += 1
        c1 = float(r['C1']); c2 = float(r['C2'])
        s['sum_C2'] += c2
        s['sum_C1sq'] += (c1*c1)
    # finalize
    for b, s in stats.items():
        cnt = max(1, s['count'])
        s['avg_C2'] = s['sum_C2']/cnt
        s['avg_C1sq'] = s['sum_C1sq']/cnt
        s['kind'] = key
    return stats

# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latent', required=True)
    ap.add_argument('--ax-min', type=float, required=True)
    ap.add_argument('--ax-max', type=float, required=True)
    ap.add_argument('--ay-min', type=float, required=True)
    ap.add_argument('--ay-max', type=float, required=True)
    ap.add_argument('--ax-steps', type=int, default=401)
    ap.add_argument('--ay-steps', type=int, default=401)
    ap.add_argument('--beta', type=float, default=0.55)
    ap.add_argument('--gamma0', type=float, default=0.0)
    ap.add_argument('--gamma1', type=float, default=1.0)
    ap.add_argument('--min-sep-ax', type=float, default=0.004)   # ← fixed typo here
    ap.add_argument('--min-sep-ay', type=float, default=0.004)
    ap.add_argument('--min-sep-z',  type=float, default=1e-3)
    ap.add_argument('--refine-rounds', type=int, default=1)
    ap.add_argument('--refine-halfwidth', type=float, default=0.01)
    ap.add_argument('--refine-steps', type=int, default=81)
    ap.add_argument('--bands-rect', type=str, default=None)
    ap.add_argument('--bands-seed', type=str, default=None)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load latent CSV
    df = pd.read_csv(args.latent)
    for col in ('ax','ay','z'):
        if col not in df.columns:
            raise ValueError("latent CSV must have columns ax, ay, z")
    ax_raw = df['ax'].to_numpy(float)
    ay_raw = df['ay'].to_numpy(float)
    z_raw  = df['z'].to_numpy(float)
    has_AB = ('A' in df.columns and 'B' in df.columns)

    coeffs_z, zhat = fit_quadratic(ax_raw, ay_raw, z_raw)
    if has_AB:
        _, Ahat = fit_quadratic(ax_raw, ay_raw, df['A'].to_numpy(float))
        _, Bhat = fit_quadratic(ax_raw, ay_raw, df['B'].to_numpy(float))
    else:
        Ahat = Bhat = None

    # Load bands
    rects = load_bands_rectangles(args.bands_rect) if args.bands_rect else None
    seeds = load_bands_seeds(args.bands_seed) if args.bands_seed else None

    # Coarse grid
    ax_grid = np.linspace(args.ax_min, args.ax_max, args.ax_steps)
    ay_grid = np.linspace(args.ay_min, args.ay_max, args.ay_steps)
    dax = (args.ax_max - args.ax_min)/(args.ax_steps-1)
    day = (args.ay_max - args.ay_min)/(args.ay_steps-1)
    AX, AY = np.meshgrid(ax_grid, ay_grid)   # AY rows, AX cols
    Z  = zhat(AX, AY)
    DF = (Ahat(AX, AY) - Bhat(AX, AY)) if has_AB else (args.gamma0 + args.gamma1*Z)
    W, C1, C2 = logistic_from_deltaF(DF, args.beta)
    G  = finite_diff_grad(Z, dax, day)

    # Full samples with band labels
    samples = []
    for iy in range(AY.shape[0]):
        for ix in range(AX.shape[1]):
            axv, ayv = float(AX[iy, ix]), float(AY[iy, ix])
            band = label_point(axv, ayv, rects=rects, seeds=seeds)
            samples.append({
                'ax': axv, 'ay': ayv,
                'z': float(Z[iy, ix]), 'grad': float(G[iy, ix]),
                'DeltaF': float(DF[iy, ix]), 'w': float(W[iy, ix]),
                'C1': float(C1[iy, ix]), 'C2': float(C2[iy, ix]),
                'band': band, 'level': 'coarse'
            })

    # Write samples
    full_path = os.path.join(args.outdir, 'full_scan_samples.csv')
    pd.DataFrame(samples).to_csv(full_path, index=False)

    # Coarse minima
    mins_idx = local_minima(Z)
    candidates = []
    for (iy, ix) in mins_idx:
        axv, ayv = float(AX[iy, ix]), float(AY[iy, ix])
        band = label_point(axv, ayv, rects=rects, seeds=seeds)
        candidates.append({
            'ax': axv, 'ay': ayv, 'z': float(Z[iy, ix]),
            'grad': float(G[iy, ix]), 'DeltaF': float(DF[iy, ix]),
            'w': float(W[iy, ix]), 'C1': float(C1[iy, ix]), 'C2': float(C2[iy, ix]),
            'band': band, 'level': 'coarse'
        })

    # Adaptive refinement rounds
    for r in range(args.refine_rounds):
        new_cands = []
        for c in candidates:
            ax0, ay0 = c['ax'], c['ay']
            ax_lo = max(args.ax_min, ax0 - args.refine_halfwidth)
            ax_hi = min(args.ax_max, ax0 + args.refine_halfwidth)
            ay_lo = max(args.ay_min, ay0 - args.refine_halfwidth)
            ay_hi = min(args.ay_max, ay0 + args.refine_halfwidth)
            ax_ref = np.linspace(ax_lo, ax_hi, args.refine_steps)
            ay_ref = np.linspace(ay_lo, ay_hi, args.refine_steps)
            AXr, AYr = np.meshgrid(ax_ref, ay_ref)
            Zr  = zhat(AXr, AYr)
            DFr = (Ahat(AXr, AYr) - Bhat(AXr, AYr)) if has_AB else (args.gamma0 + args.gamma1*Zr)
            Wr, C1r, C2r = logistic_from_deltaF(DFr, args.beta)
            Gr  = finite_diff_grad(Zr, ax_ref[1]-ax_ref[0], ay_ref[1]-ay_ref[0])

            mins_r = local_minima(Zr)
            for (iry, irx) in mins_r:
                axv, ayv = float(AXr[iry, irx]), float(AYr[iry, irx])
                band = label_point(axv, ayv, rects=rects, seeds=seeds)
                new_cands.append({
                    'ax': axv, 'ay': ayv, 'z': float(Zr[iry, irx]),
                    'grad': float(Gr[iry, irx]), 'DeltaF': float(DFr[iry, irx]),
                    'w': float(Wr[iry, irx]), 'C1': float(C1r[iry, irx]), 'C2': float(C2r[iry, irx]),
                    'band': band, 'level': f'refine{r+1}'
                })
        candidates.extend(new_cands)

    # Save raw minima and locked
    cand_path = os.path.join(args.outdir, 'candidate_minima.csv')
    pd.DataFrame(candidates).sort_values('z').to_csv(cand_path, index=False)

    locked = separation_filter(
        candidates, args.min_sep_ax, args.min_sep_ay, args.min_sep_z
    )
    locked_path = os.path.join(args.outdir, 'locked_minima.csv')
    pd.DataFrame(locked).sort_values('z').to_csv(locked_path, index=False)

    # Band stats
    sample_stats = aggregate_band_stats(samples, key='samples')
    minima_stats = aggregate_band_stats(locked, key='minima')

    # Merge stats per band
    bands = sorted(set(list(sample_stats.keys()) + list(minima_stats.keys())))
    out_stats = {}
    for b in bands:
        out_stats[b] = {
            'samples': sample_stats.get(b, {'count':0,'avg_C2':0,'avg_C1sq':0}),
            'minima':  minima_stats.get(b, {'count':0,'avg_C2':0,'avg_C1sq':0})
        }

    stats_path = os.path.join(args.outdir, 'band_averages.json')
    with open(stats_path, 'w') as f:
        json.dump(out_stats, f, indent=2)

    print("[OK] Wrote:")
    print("  -", full_path)
    print("  -", cand_path)
    print("  -", locked_path)
    print("  -", stats_path)
    print(f"[INFO] Raw minima: {len(candidates)} | Locked minima: {len(locked)}")

if __name__ == '__main__':
    main()