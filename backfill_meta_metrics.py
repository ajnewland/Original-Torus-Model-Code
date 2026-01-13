# backfill_meta_metrics.py
# Recompute xi and spectral-dimension plateau from existing CSVs and
# update meta_summary.json in each run directory.

import os, json, glob
import numpy as np
import pandas as pd

def robust_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, sep=';')
        except Exception:
            return None

def estimate_xi_from_corr(df):
    """
    df columns expected: 'r' (pixel distance), 'corr' (normalized C(r), C(0)=1)
    We try two estimates:
      (1) e-folding: first r where corr <= 1/e
      (2) linear fit on ln corr vs r over the range corr in [0.2, 0.8]
    Return the best available (prefer fit if stable).
    """
    if df is None or {'r','corr'} - set(df.columns):
        return None, None
    dd = df.dropna().sort_values('r')
    if dd.empty: return None, None

    # (1) e-folding estimate
    e_fold = None
    try:
        below = dd[dd['corr'] <= np.exp(-1)]
        if not below.empty:
            e_fold = below.iloc[0]['r']
    except Exception:
        pass

    # (2) log-linear fit in a moderate-corr window
    xi_fit, R2 = None, None
    try:
        mask = (dd['corr'] > 0) & (dd['corr'] <= 0.8) & (dd['corr'] >= 0.2)
        ww = dd[mask]
        if len(ww) >= 5:
            x = ww['r'].values
            y = np.log(ww['corr'].values)
            A = np.vstack([x, np.ones_like(x)]).T
            sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            slope, intercept = sol
            # y ~ intercept + slope * r  => corr ~ exp(intercept) * exp(slope r)
            if slope < 0:
                xi_fit = -1.0/slope
            # R^2 of log-fit
            yhat = intercept + slope*x
            ss_res = np.sum((y - yhat)**2)
            ss_tot = np.sum((y - y.mean())**2) + 1e-30
            R2 = 1.0 - ss_res/ss_tot
    except Exception:
        pass

    # choose fit if sensible; otherwise e-fold
    if xi_fit is not None and np.isfinite(xi_fit) and (R2 is not None) and R2 > 0.05:
        return float(xi_fit), float(R2)
    if e_fold is not None and np.isfinite(e_fold):
        return float(e_fold), None
    return None, None

def spectral_dimension_from_heat(df):
    """
    df should contain columns:
      - 't' (diffusion time)
      - one of: 'P_t' (heat trace) or 'heat_trace'
      - optional: 'ds' (precomputed spectral dimension)
    We compute ds(t) = -2 dlog(P)/dlog(t) with a robust finite difference.
    Then identify a 'plateau' region by looking for a contiguous band where
    |d(ds)/dlog t| is small and ds is within a broad physical range (0..8).
    Return: plateau_mean, plateau_spread, ds_max, ds_median
    """
    if df is None:
        return None
    df = df.copy()
    cols = set(c.lower() for c in df.columns)
    tcol = next((c for c in df.columns if c.lower()=='t'), None)
    if tcol is None: return None

    # pick heat trace column
    H = None
    for cname in ['P_t','heat_trace','p_t','P','trace']:
        if cname in df.columns:
            H = cname
            break
    if H is None:
        # sometimes column name is 'P' but stored as string; try last numeric
        nums = [c for c in df.columns if c not in [tcol] and np.issubdtype(df[c].dtype, np.number)]
        if nums:
            H = nums[0]
        else:
            return None

    t = df[tcol].to_numpy(dtype=float)
    P = df[H].to_numpy(dtype=float)
    good = np.isfinite(t) & np.isfinite(P) & (t>0) & (P>0)
    t, P = t[good], P[good]
    if t.size < 8: return None

    # sort and log
    order = np.argsort(t)
    t, P = t[order], P[order]
    logt = np.log(t)
    logP = np.log(P)

    # smooth a bit (3-pt moving average) to reduce noise
    def movavg(a, k=3):
        k = max(1, int(k))
        if k==1: return a
        pad = k//2
        ap = np.pad(a, (pad,pad), mode='edge')
        ker = np.ones(k)/k
        return np.convolve(ap, ker, mode='valid')
    logt_s = movavg(logt, 3)
    logP_s = movavg(logP, 3)

    # numeric derivative
    dlogP = np.gradient(logP_s, logt_s)
    ds = -2.0 * dlogP

    # restrict to a safe window (avoid very small/large t)
    # use middle 60% quantile of t
    q1, q2 = np.quantile(logt_s, [0.2, 0.8])
    mask_mid = (logt_s >= q1) & (logt_s <= q2)
    ds_mid = ds[mask_mid]
    logt_mid = logt_s[mask_mid]
    if ds_mid.size < 10:  # not enough samples
        return None

    # reject outliers and unphysical spikes
    phys = (ds_mid >= 0.0) & (ds_mid <= 8.0)
    ds_mid = ds_mid[phys]
    logt_mid = logt_mid[phys]
    if ds_mid.size < 10: return None

    # plateau = where derivative of ds w.r.t log t is small
    dds = np.gradient(ds_mid, logt_mid)
    small_slope = np.abs(dds) <= np.quantile(np.abs(dds), 0.4)  # most gentle 60%
    band = ds_mid[small_slope]
    if band.size < 5:  # fallback: just use ds_mid
        band = ds_mid

    plateau_mean = float(np.mean(band))
    plateau_spread = float(np.std(band))
    return {
        "ds_plateau_mean": plateau_mean,
        "ds_plateau_spread": plateau_spread,
        "ds_max": float(np.max(ds_mid)),
        "ds_median": float(np.median(ds_mid)),
    }

def update_one_run(run_dir):
    meta_path = os.path.join(run_dir, "meta_summary.json")
    corr_path = os.path.join(run_dir, "corr_vs_distance.csv")
    heat_path = os.path.join(run_dir, "meta_heat_trace.csv")

    # load/create meta
    meta = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    changed = False

    # xi
    if meta.get("xi_pixels") is None:
        corr = robust_read_csv(corr_path)
        xi, R2 = estimate_xi_from_corr(corr)
        if xi is not None:
            meta["xi_pixels"] = float(xi)
            if R2 is not None: meta["xi_logfit_R2"] = float(R2)
            changed = True

    # spectral dimension
    need_ds = any(meta.get(k) is None for k in
                  ["ds_plateau_mean","ds_plateau_spread","ds_max","ds_median"])
    if need_ds:
        heat = robust_read_csv(heat_path)
        ds_info = spectral_dimension_from_heat(heat)
        if ds_info is not None:
            meta.update(ds_info)
            changed = True

    if changed:
        # make sure required header info exists (safeguard)
        meta.setdefault("N", None)
        meta.setdefault("alpha", None)
        meta.setdefault("operator", None)
        meta.setdefault("smooth_px", None)
        meta.setdefault("sigma_graph", None)
        meta.setdefault("rot_deg", None)
        meta.setdefault("windowsize", None)

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    return changed

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Back-fill xi and ds metrics into meta_summary.json for meta-geometry runs.")
    ap.add_argument("--roots", nargs="+", required=True,
                    help="One or more root directories containing many run folders.")
    args = ap.parse_args()

    total, changed = 0, 0
    for root in args.roots:
        # accept both flat and nested structures
        for run_dir in glob.glob(os.path.join(root, "*")):
            if not os.path.isdir(run_dir): continue
            # must contain either corr_vs_distance.csv or meta_heat_trace.csv
            if not (os.path.isfile(os.path.join(run_dir, "corr_vs_distance.csv")) or
                    os.path.isfile(os.path.join(run_dir, "meta_heat_trace.csv"))):
                continue
            total += 1
            if update_one_run(run_dir):
                changed += 1
    print(f"[DONE] scanned {total} run dirs, updated {changed}")

if __name__ == "__main__":
    main()