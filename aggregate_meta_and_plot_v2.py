#!/usr/bin/env python3
import json, re, math, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def find_one(patterns, root):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(root, pat)))
        if hits:
            return hits[0]
    return None

def fit_xi_from_corr(csv_path):
    df = pd.read_csv(csv_path)
    # Accept flexible column names
    cols = {c.lower():c for c in df.columns}
    rcol = cols.get('r') or cols.get('radius') or list(df.columns)[0]
    ccol = cols.get('c') or cols.get('corr') or list(df.columns)[1]
    r = df[rcol].to_numpy()
    C = df[ccol].to_numpy()
    if not np.isfinite(C).any():
        return np.nan, np.nan
    C0 = C[0] if np.isfinite(C[0]) else np.nanmax(C)
    if C0 <= 0 or not np.isfinite(C0):
        return np.nan, np.nan
    mask = (C > 0) & np.isfinite(C) & np.isfinite(r)
    r, C = r[mask], C[mask]
    if r.size < 5: return np.nan, np.nan
    # Use mid-range where C/C0 in [0.1, 0.9]
    ratio = C / C0
    m = (ratio >= 0.1) & (ratio <= 0.9)
    if m.sum() < 5:
        # fallback: use central 60% in r
        lo, hi = np.percentile(r, [20,80])
        m = (r>=lo) & (r<=hi)
    if m.sum() < 5:
        return np.nan, np.nan
    R = r[m]; Y = np.log(C[m])
    # robust fit
    try:
        slope, intercept = np.polyfit(R, Y, 1)
        xi = -1.0/slope if slope < 0 else np.nan
        # R^2 in log-space
        yhat = slope*R + intercept
        ss_res = np.sum((Y-yhat)**2)
        ss_tot = np.sum((Y - Y.mean())**2)
        r2 = 1.0 - ss_res/ss_tot if ss_tot>0 else np.nan
        return float(xi), float(r2)
    except Exception:
        return np.nan, np.nan

def ds_from_heat_trace(csv_path):
    df = pd.read_csv(csv_path)
    cols = {c.lower():c for c in df.columns}
    scol = cols.get('s') or list(df.columns)[0]
    kcol = cols.get('k') or cols.get('heat') or list(df.columns)[1]
    s = df[scol].to_numpy()
    K = df[kcol].to_numpy()
    m = np.isfinite(s) & np.isfinite(K) & (s>0) & (K>0)
    s, K = s[m], K[m]
    if s.size < 6:
        return np.nan, np.nan, np.nan, np.nan
    # spectral dimension: d_s(s) = -2 d ln K / d ln s
    lnS = np.log(s); lnK = np.log(K)
    # numerical derivative on a monotone grid
    order = np.argsort(lnS); lnS, lnK = lnS[order], lnK[order]
    dlnS = np.diff(lnS); dlnK = np.diff(lnK)
    dS = -2.0 * (dlnK / dlnS)
    S_mid = 0.5*(lnS[1:]+lnS[:-1])
    # take the mid 30%–70% of lnS as a crude plateau window
    lo, hi = np.percentile(S_mid, [30,70])
    win = (S_mid>=lo) & (S_mid<=hi)
    if win.sum() < 5:
        # fallback: middle 50%
        lo, hi = np.percentile(S_mid, [25,75])
        win = (S_mid>=lo) & (S_mid<=hi)
    if win.sum() < 5:
        return float(np.nan), float(np.nan), float(np.nan), float(np.nan)
    ds_vals = dS[win]
    return float(np.nanmean(ds_vals)), float(np.nanmedian(ds_vals)), float(np.nanmax(dS)), float(np.nanstd(ds_vals))

def parse_meta_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        try:
            j = json.load(f)
        except Exception:
            return {}
    out = {}
    # Flexible keys
    for k in ['N','alpha','operator','smooth_px','sigma_graph','rot_deg','windowsize']:
        for cand in [k, k.upper(), k.lower()]:
            if cand in j:
                out[k] = j[cand]
                break
    return out

def guess_from_folder(stem):
    # pull numbers like N256, smooth_px=80, sigma_graph=450, rot_deg=5, alpha=4.9
    out = {}
    m = re.search(r'N(\d+)', stem, re.I);           out['N'] = int(m.group(1)) if m else np.nan
    m = re.search(r'smooth[_-]?px[_=]?(\d+(\.\d+)?)', stem, re.I); out['smooth_px']= safe_float(m.group(1)) if m else np.nan
    m = re.search(r'sigma[_-]?graph[_=]?(\d+(\.\d+)?)', stem, re.I);out['sigma_graph']= safe_float(m.group(1)) if m else np.nan
    m = re.search(r'rot[_-]?deg[_=]?(\d+(\.\d+)?)', stem, re.I);     out['rot_deg']= safe_float(m.group(1)) if m else np.nan
    m = re.search(r'alpha[_=]?(\d+(\.\d+)?)', stem, re.I);           out['alpha']= safe_float(m.group(1)) if m else np.nan
    return out

def aggregate(base_dir, outdir):
    rows = []
    run_dirs = [p for p in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(p)]
    for rd in sorted(run_dirs):
        stem = os.path.basename(rd)
        meta = parse_meta_json(find_one(['meta_summary.json','*meta_summary*.json'], rd))
        guess = guess_from_folder(stem)
        N          = meta.get('N', guess.get('N'))
        alpha      = meta.get('alpha', guess.get('alpha'))
        operator   = meta.get('operator', 'spectral')
        smooth_px  = meta.get('smooth_px', guess.get('smooth_px'))
        sigma_g    = meta.get('sigma_graph', guess.get('sigma_graph'))
        rot_deg    = meta.get('rot_deg', guess.get('rot_deg'))
        windowsize = meta.get('windowsize', np.nan)

        corr_path  = find_one(['corr_vs_distance*.csv'], rd)
        heat_path  = find_one(['*heat_trace*.csv'], rd)

        xi, xi_r2 = (np.nan, np.nan)
        if corr_path:
            xi, xi_r2 = fit_xi_from_corr(corr_path)

        ds_mean=ds_median=ds_max=ds_spread=np.nan
        if heat_path:
            ds_mean, ds_median, ds_max, ds_spread = ds_from_heat_trace(heat_path)

        rows.append(dict(run=stem, path=rd, N=N, alpha=alpha, operator=operator,
                         smooth_px=smooth_px, sigma_graph=sigma_g, rot_deg=rot_deg,
                         windowsize=windowsize, xi_pixels=xi, xi_logfit_R2=xi_r2,
                         ds_plateau_mean=ds_mean, ds_plateau_spread=ds_spread,
                         ds_max=ds_max, ds_median=ds_median))

    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_out = os.path.join(outdir, 'grid_meta_summary_all.csv')
    df.to_csv(csv_out, index=False)
    print(f'[OK] Wrote {csv_out} with {len(df)} rows')

    # Column completeness
    print('\n=== Column completeness ===')
    for col in ['N','alpha','operator','smooth_px','sigma_graph','rot_deg',
                'windowsize','xi_pixels','xi_logfit_R2',
                'ds_plateau_mean','ds_plateau_spread','ds_max','ds_median']:
        miss = df[col].isna().mean()*100 if col in df else 100.0
        if df[col].dtype.kind in 'iufc' if col in df else False:
            vals = df[col].dropna()
            mean = vals.mean() if len(vals) else float('nan')
            std  = vals.std() if len(vals) else float('nan')
            mn   = vals.min() if len(vals) else float('nan')
            mx   = vals.max() if len(vals) else float('nan')
            print(f'{col:18s} missing={miss:5.1f}%   mean={mean:g}  std={std:g}  min={mn:g}  max={mx:g}')
        else:
            print(f'{col:18s} missing={miss:5.1f}%   n/a')

    # Plots (only if we have data)
    def has(col): return col in df and df[col].notna().any()

    if has('ds_plateau_mean') and has('N'):
        plt.figure(figsize=(7,5))
        plt.scatter(df['N'], df['ds_plateau_mean'], s=30)
        plt.axhline(4.0, ls='--', color='k', label='Einstein limit ($d_s=4$)')
        plt.xlabel('N (number of tori)'); plt.ylabel(r'$d_s$ plateau mean')
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'ds_plateau_mean_vs_N.png'), dpi=160)

    if has('xi_pixels') and has('N') and has('smooth_px'):
        plt.figure(figsize=(7,5))
        sc = plt.scatter(df['N'], df['xi_pixels'], c=df['smooth_px'], cmap='viridis')
        cbar = plt.colorbar(sc); cbar.set_label('smooth_px')
        plt.xlabel('N (number of tori)'); plt.ylabel(r'$\xi$ (pixels)')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'xi_vs_N.png'), dpi=160)

        plt.figure(figsize=(7,5))
        sc2 = plt.scatter(df['smooth_px'], df['xi_pixels'], c=df['N'], cmap='plasma')
        cbar2 = plt.colorbar(sc2); cbar2.set_label('N')
        plt.xlabel('smooth_px'); plt.ylabel(r'$\xi$ (pixels)')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'xi_vs_smooth_px.png'), dpi=160)

    # Tiny convergence hints
    print('\n=== Convergence diagnostics ===')
    if has('xi_pixels'):
        xv = df['xi_pixels'].dropna().values
        if len(xv)>3:
            print(f'xi median={np.median(xv):.2f}, IQR={np.percentile(xv,75)-np.percentile(xv,25):.2f}')
        else:
            print('xi fit: insufficient data')
    else:
        print('xi fit: insufficient data')

    if has('ds_plateau_mean'):
        dv = df['ds_plateau_mean'].dropna().values
        if len(dv)>3:
            print(f'ds mean={np.mean(dv):.3f}, median={np.median(dv):.3f}, spread~{np.std(dv):.3f}')
        else:
            print('ds fit: insufficient data')
    else:
        print('ds fit: insufficient data')

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Aggregate meta-geometry runs and plot summaries (robust filenames).")
    ap.add_argument('--base_dir', required=True, help='Folder that contains many run subfolders')
    ap.add_argument('--outdir',    required=True, help='Where to write CSV and plots')
    args = ap.parse_args()
    aggregate(args.base_dir, args.outdir)