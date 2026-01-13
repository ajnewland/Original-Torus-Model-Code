#!/usr/bin/env python3
import argparse, math, os
import numpy as np
import pandas as pd

def load_slopes(path):
    df = pd.read_csv(path)
    # normalize column names
    cols = {c.lower(): c for c in df.columns}
    # expected: sector, alpha_raw or alpha, alpha_norm (for neutrinos)
    sec_col = cols.get('sector', None)
    if sec_col is None:
        raise ValueError(f"sector column missing in {path}")
    def pick_alpha(row):
        # prefer alpha_norm for neutrinos if present & finite
        a_norm = row.get('alpha_norm', np.nan) if 'alpha_norm' in df.columns else np.nan
        a_raw  = row.get('alpha_raw',  np.nan) if 'alpha_raw'  in df.columns else np.nan
        a_basic= row.get('alpha',      np.nan) if 'alpha'      in df.columns else np.nan
        sector = str(row[sec_col]).strip().lower()
        if sector == 'neutrinos' and np.isfinite(a_norm):
            return a_norm
        # prefer raw if available; else alpha
        if np.isfinite(a_raw):  return a_raw
        if np.isfinite(a_basic):return a_basic
        return np.nan
    df['_alpha_used'] = df.apply(pick_alpha, axis=1)
    slope_map = { str(r[sec_col]).strip().lower(): float(r['_alpha_used']) for _,r in df.iterrows() if np.isfinite(r['_alpha_used']) }
    return slope_map

def nearest_ds_for(ax, ay, ds_df):
    # brute-force nearest neighbor (15 species × N_map is fine)
    dx = ds_df['ax'].to_numpy() - ax
    dy = ds_df['ay'].to_numpy() - ay
    i  = np.argmin(dx*dx + dy*dy)
    return float(ds_df.iloc[i]['ds']), float(ds_df.iloc[i]['ax']), float(ds_df.iloc[i]['ay'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--locked', required=True, help='all_particles_locked.csv (your columns)')
    ap.add_argument('--dsmap',  required=True, help='ax,ay,ds map CSV')
    ap.add_argument('--sectorslopes', required=True, help='sector_slopes.csv')
    ap.add_argument('--outcsv', required=True, help='output CSV path')
    ap.add_argument('--outpng', default='', help='optional PNG of ds field with species overlay')
    ap.add_argument('--lambda_ds', type=float, default=0.10, help='strength of ds correction (default 0.10)')
    ap.add_argument('--ds_lo', type=float, default=0.8, help='min clamp for dshat (default 0.8)')
    ap.add_argument('--ds_hi', type=float, default=1.2, help='max clamp for dshat (default 1.2)')
    ap.add_argument('--gamma', type=float, default=0.0, help='tiny global bias term (default 0.0)')
    args = ap.parse_args()

    # 1) load
    L = pd.read_csv(args.locked)
    for need in ['species','ax','ay','z_pred']:
        if need not in L.columns:
            raise ValueError(f"[ERR] locked CSV missing required column '{need}'. Got: {list(L.columns)}")
    # m_PDG_GeV may be called m_GeV in your file
    if 'm_PDG_GeV' not in L.columns and 'm_GeV' in L.columns:
        L['m_PDG_GeV'] = L['m_GeV']
    if 'm_PDG_GeV' not in L.columns:
        raise ValueError("[ERR] locked CSV must contain 'm_PDG_GeV' or 'm_GeV'.")
    # sector may be missing; we can derive from species if present in your version
    if 'sector' not in L.columns:
        # very light heuristic; adjust if you prefer to pass sector explicitly
        def infer_sector(s):
            s=str(s).lower()
            if s in ['w','z','h','higgs']: return 'bosons'
            if s in ['nu','nu1','nu2','nu3','nu_e','nu_mu','nu_tau']: return 'neutrinos'
            if s in ['e','mu','tau']: return 'leptons'
            if s in ['u','c','t']: return 'up'
            if s in ['d','s','b']: return 'down'
            return 'unknown'
        L['sector'] = L['species'].map(infer_sector)
    L['sector'] = L['sector'].astype(str).str.strip().str.lower()

    dsmap = pd.read_csv(args.dsmap)
    for need in ['ax','ay','ds']:
        if need not in dsmap.columns:
            raise ValueError(f"[ERR] dsmap CSV must contain columns ax, ay, ds. Got: {list(dsmap.columns)}")
    slopes = load_slopes(args.sectorslopes)
    # Check slopes present for sectors we need
    missing = sorted(set(L['sector']) - set(slopes.keys()))
    if missing:
        raise ValueError(f"[ERR] missing slopes for sectors: {missing}. Found: {slopes}")

    # 2) attach local ds by nearest neighbor
    dsvals, axn, ayn = [], [], []
    for _,r in L.iterrows():
        ds, ax_nn, ay_nn = nearest_ds_for(float(r['ax']), float(r['ay']), dsmap)
        dsvals.append(ds); axn.append(ax_nn); ayn.append(ay_nn)
    L['ds_raw'] = dsvals
    L['ds_ax_nn'] = axn
    L['ds_ay_nn'] = ayn

    # 3) normalized & clamped dshat
    ds_mean = float(np.mean(L['ds_raw'].replace([np.inf,-np.inf], np.nan).dropna()))
    if not np.isfinite(ds_mean) or ds_mean == 0:
        ds_mean = 1.0
    L['dshat'] = 1.0 + args.lambda_ds * ((L['ds_raw'] - ds_mean) / ds_mean)
    L['dshat'] = L['dshat'].clip(lower=args.ds_lo, upper=args.ds_hi)

    # 4) per-sector alpha
    L['alpha_used'] = L['sector'].map(slopes).astype(float)

    # 5) fit beta on W/Z/H only
    def ln_safe(x):
        x = np.asarray(x, float)
        x = np.where(x>0, x, np.nan)
        return np.log(x)

    mask_boson = L['species'].str.lower().isin(['w','z','h','higgs'])
    # predicted without beta:
    base_pred = L['alpha_used'] * L['dshat'] * L['z_pred'] + args.gamma
    # target logs:
    target_log = ln_safe(L['m_PDG_GeV'])

    if mask_boson.sum() >= 2 and np.isfinite(target_log[mask_boson]).all():
        beta = np.nanmean(target_log[mask_boson] - base_pred[mask_boson])
    else:
        # fallback: center on all finite targets
        beta = np.nanmean(target_log - base_pred)

    # 6) final predictions
    L['beta_used'] = beta
    L['logm_pred'] = base_pred + beta
    L['m_pred_GeV'] = np.exp(L['logm_pred'])

    # metrics
    L['abs_dlog'] = np.abs(L['logm_pred'] - target_log)

    # 7) write CSV
    outcols = ['species','sector','ax','ay','z_pred','m_PDG_GeV',
               'ds_raw','dshat','alpha_used','beta_used','gamma',
               'logm_pred','m_pred_GeV','abs_dlog','ds_ax_nn','ds_ay_nn']
    # Ensure gamma column present
    L['gamma'] = args.gamma
    L[outcols].to_csv(args.outcsv, index=False)
    print(f"[OK] wrote {args.outcsv}")

    # 8) optional plot of ds field + species (no heavy deps)
    if args.outpng:
        try:
            import matplotlib.pyplot as plt
            fig, axp = plt.subplots(figsize=(7,5))
            sc = axp.scatter(dsmap['ax'], dsmap['ay'], c=dsmap['ds'], s=12, alpha=0.6)
            plt.colorbar(sc, ax=axp, label='d_s (raw)')
            axp.scatter(L['ax'], L['ay'], c='k', s=30, marker='x')
            for _,r in L.iterrows():
                axp.text(r['ax'], r['ay'], str(r['species']), fontsize=8)
            axp.set_xlabel('a_x'); axp.set_ylabel('a_y'); axp.set_title('d_s map + species')
            fig.tight_layout()
            fig.savefig(args.outpng, dpi=150)
            plt.close(fig)
            print(f"[OK] plot {args.outpng}")
        except Exception as e:
            print(f"[WARN] plot failed: {e}")

if __name__ == '__main__':
    main()