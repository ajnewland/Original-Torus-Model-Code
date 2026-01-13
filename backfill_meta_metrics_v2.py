# backfill_meta_metrics_v2.py
import argparse, json, os, glob, math
from pathlib import Path
import numpy as np
import pandas as pd

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def robust_linfit(x, y):
    # simple least-squares with nan guard
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan
    A = np.vstack([x[m], np.ones(m.sum())]).T
    coef, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    slope, intercept = coef
    yhat = A @ coef
    ss_res = np.sum((y[m]-yhat)**2)
    ss_tot = np.sum((y[m]-y[m].mean())**2)
    R2 = 1.0 - ss_res/max(ss_tot, 1e-30)
    return slope, R2

def estimate_xi_pixels(corr_df, method="log_tail"):
    # expects columns: r (pixels), C(r)
    r = corr_df.get("r", corr_df.iloc[:,0]).to_numpy()
    C = corr_df.get("C", corr_df.iloc[:,1]).to_numpy()
    m = np.isfinite(r) & np.isfinite(C) & (r>0) & (C>0)
    r, C = r[m], C[m]
    if len(r) < 10:
        return np.nan, np.nan
    # take last third as "tail" and fit log C ~ -r/xi
    k0 = max(10, len(r)//3)
    r_tail = r[-k0:]
    C_tail = C[-k0:]
    slope, R2 = robust_linfit(r_tail, -np.log(C_tail))
    if not np.isfinite(slope) or slope <= 0:
        return np.nan, R2
    xi = 1.0/slope
    return float(xi), float(R2)

def spectral_dimension_from_heat_trace(ht_df, tau_min=1e-3, tau_max=1e+1, windowsize=32):
    # expects columns: tau, K  (heat trace vs diffusion time)
    tau = ht_df.get("tau", ht_df.iloc[:,0]).to_numpy()
    K   = ht_df.get("K",   ht_df.iloc[:,1]).to_numpy()
    m = np.isfinite(tau) & np.isfinite(K) & (tau>0) & (K>0)
    tau, K = tau[m], K[m]
    if len(tau) < windowsize+2:
        return dict(ds_plateau_mean=np.nan, ds_plateau_spread=np.nan,
                    ds_max=np.nan, ds_median=np.nan)
    # restrict tau window
    m2 = (tau>=tau_min) & (tau<=tau_max)
    tau, K = tau[m2], K[m2]
    if len(tau) < windowsize+2:
        return dict(ds_plateau_mean=np.nan, ds_plateau_spread=np.nan,
                    ds_max=np.nan, ds_median=np.nan)
    # d_s(tau) = -2 d ln K / d ln tau (use finite difference in sliding window)
    ln_tau = np.log(tau)
    ln_K   = np.log(K)
    ds_vals = []
    half = windowsize//2
    for i in range(half, len(tau)-half):
        x = ln_tau[i-half:i+half+1]
        y = ln_K[i-half:i+half+1]
        if len(x) < 3:
            continue
        slope, _R2 = robust_linfit(x, y)
        if np.isfinite(slope):
            ds_vals.append(float(-2.0*slope))
    if len(ds_vals) == 0:
        return dict(ds_plateau_mean=np.nan, ds_plateau_spread=np.nan,
                    ds_max=np.nan, ds_median=np.nan)
    ds_vals = np.array(ds_vals)
    return dict(
        ds_plateau_mean   = float(np.mean(ds_vals)),
        ds_plateau_spread = float(np.std(ds_vals)),
        ds_max            = float(np.max(ds_vals)),
        ds_median         = float(np.median(ds_vals))
    )

def backfill_one_run(run_dir, windowsize, tau_min, tau_max, verbose=False):
    run_dir = Path(run_dir)
    jpaths = list(run_dir.glob("meta_summary.json"))
    if not jpaths:
        return False, "no_json"
    jpath = jpaths[0]
    meta = load_json(jpath)

    # Locate inputs
    corr_paths = sorted(glob.glob(str(run_dir / "corr_vs_distance*.csv")))
    heat_paths = sorted(glob.glob(str(run_dir / "meta_heat_trace*.csv")))
    changed = False

    # Recompute xi
    xi, xiR2 = (np.nan, np.nan)
    if corr_paths:
        try:
            corr_df = pd.read_csv(corr_paths[0])
            xi, xiR2 = estimate_xi_pixels(corr_df)
            meta["xi_pixels"] = None if not np.isfinite(xi) else float(xi)
            meta["xi_logfit_R2"] = None if not np.isfinite(xiR2) else float(xiR2)
            changed = True
        except Exception as e:
            if verbose: print(f"[{run_dir.name}] xi fail: {e}")

    # Recompute ds plateau
    if heat_paths:
        try:
            ht_df = pd.read_csv(heat_paths[0])
            ds_stats = spectral_dimension_from_heat_trace(
                ht_df, tau_min=tau_min, tau_max=tau_max, windowsize=windowsize
            )
            meta.update(ds_stats)
            meta["windowsize"] = int(windowsize)
            changed = True
        except Exception as e:
            if verbose: print(f"[{run_dir.name}] ds fail: {e}")

    if changed:
        save_json(jpath, meta)
        return True, "updated"
    else:
        return False, "no_inputs"

def main():
    ap = argparse.ArgumentParser(description="Force-recompute xi and ds metrics into meta_summary.json files.")
    ap.add_argument("--roots", nargs="+", required=True, help="One or more root folders containing many run_* subfolders.")
    ap.add_argument("--windowsize", type=int, default=32, help="Sliding window size for d_s(τ) fit (default 32).")
    ap.add_argument("--ds_tau_min", type=float, default=1e-3, help="Min τ for spectral-dimension fit.")
    ap.add_argument("--ds_tau_max", type=float, default=1e+1, help="Max τ for spectral-dimension fit.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    total = 0
    patched = 0
    skipped = 0
    for root in args.roots:
        for run_dir in Path(root).glob("run_*"):
            total += 1
            ok, msg = backfill_one_run(run_dir, args.windowsize, args.ds_tau_min, args.ds_tau_max, args.verbose)
            if ok:
                patched += 1
            else:
                skipped += 1
    print(f"[DONE] scanned {total} run dirs, updated {patched}, skipped {skipped}")

if __name__ == "__main__":
    main()