#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def load_locked(path):
    L = pd.read_csv(path)
    # Tolerate z_pred or z_predi, and m_GeV or m_PDG_GeV
    if 'z_pred' not in L.columns and 'z_predi' in L.columns:
        L = L.rename(columns={'z_predi': 'z_pred'})
    if 'm_PDG_GeV' not in L.columns and 'm_GeV' in L.columns:
        L = L.rename(columns={'m_GeV': 'm_PDG_GeV'})
    # Try to infer sector if missing
    if 'sector' not in L.columns:
        def infer_sector(sp):
            if sp in ['H','W','Z']: return 'bosons'
            if sp in ['u','c','t']: return 'up'
            if sp in ['d','s','b']: return 'down'
            if sp in ['e','mu','tau']: return 'leptons'
            if sp.startswith('nu'): return 'neutrinos'
            return 'unknown'
        L['sector'] = L['species'].apply(infer_sector)
    needed = {'species','ax','ay','sector','z_pred'}
    missing = needed - set(L.columns)
    if missing:
        raise ValueError(f"Locked CSV missing columns: {missing}")
    return L

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dsmap', required=True, help='CSV with columns: ax, ay, ds')
    p.add_argument('--locked', required=True, help='all_particles_locked.csv')
    p.add_argument('--outpng', required=True)
    p.add_argument('--outcsv', required=True)
    p.add_argument('--nn_k', type=int, default=3, help='neighbors to average for ds at a species')
    args = p.parse_args()

    D = pd.read_csv(args.dsmap)
    if not {'ax','ay','ds'} <= set(D.columns):
        raise ValueError("dsmap must have columns ax, ay, ds")
    L = load_locked(args.locked)

    # Build NN for ds lookup
    tree = cKDTree(D[['ax','ay']].to_numpy())
    pts  = L[['ax','ay']].to_numpy()
    dists, idxs = tree.query(pts, k=min(args.nn_k, len(D)))
    # Average ds over k nearest neighbors (handles k=1 transparently)
    if np.ndim(idxs) == 1:
        ds_eff = D['ds'].to_numpy()[idxs]
    else:
        ds_eff = D['ds'].to_numpy()[idxs].mean(axis=1)

    L = L.copy()
    L['ds_eff'] = ds_eff

    # Robust baseline + raw asymmetry
    ds_med = float(np.median(D['ds'].to_numpy()))
    D['T_raw'] = D['ds'] - ds_med

    # Choose sign so that bosons tend to have T>0
    bos = L[L['sector']=='bosons']['ds_eff'].to_numpy()
    sign = 1.0 if (bos - ds_med).mean() >= 0 else -1.0
    D['T'] = sign * D['T_raw']
    L['T_eff'] = sign * (L['ds_eff'] - ds_med)

    # Export per-species table
    outcols = ['species','sector','ax','ay','z_pred','ds_eff','T_eff']
    L[outcols].to_csv(args.outcsv, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(9,6.8), constrained_layout=True)
    sc = ax.scatter(D['ax'], D['ay'], c=D['T'], s=36)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(r'$T(a_x,a_y)$ (torsion asymmetry)')

    # Contour of zero line (the geometric divider sheet)
    try:
        # Build a coarse grid for contours
        gx = np.linspace(D['ax'].min(), D['ax'].max(), 200)
        gy = np.linspace(D['ay'].min(), D['ay'].max(), 200)
        GX, GY = np.meshgrid(gx, gy)
        G = np.c_[GX.ravel(), GY.ravel()]
        gd, gi = cKDTree(D[['ax','ay']].to_numpy()).query(G, k=6)
        GT = np.take(D['T'].to_numpy(), gi, mode='wrap')
        GT = GT.mean(axis=1) if GT.ndim==2 else GT
        CS = ax.contour(GX, GY, GT.reshape(GX.shape), levels=[0.0], linewidths=2)
        CS.collections[0].set_label('T=0 divider')
    except Exception:
        pass

    # Overlay species (use faint halo for readability)
    for _, r in L.iterrows():
        ax.scatter([r['ax']], [r['ay']], s=80, facecolors='none', edgecolors='k', linewidths=1.2, alpha=0.5)
        ax.text(r['ax'], r['ay'], r['species'], ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='circle,pad=0.15', fc='white', ec='0.7', alpha=0.7))

    ax.set_xlabel(r'$a_x$')
    ax.set_ylabel(r'$a_y$')
    ax.set_title(r'$T(a_x,a_y) = \mathrm{sign}_\mathrm{boson}\,[\,d_s(a_x,a_y)-\mathrm{median}(d_s)\,]$')
    ax.legend(loc='lower right', frameon=False)
    fig.savefig(args.outpng, dpi=160)
    print(f"[WROTE] {args.outcsv}")
    print(f"[PLOT ] {args.outpng}")
    print(f"[INFO ] median(d_s)={ds_med:.6f}, sign={sign:+.0f} (bosons -> T>0)")

if __name__ == '__main__':
    main()