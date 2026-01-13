#!/usr/bin/env python3
import argparse, json, os, glob
import numpy as np

def _load_tau_H(csv_path):
    taus, Hs = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                t = float(parts[0]); h = float(parts[1])
            except:
                continue
            if np.isfinite(t) and np.isfinite(h) and t>0 and h>0:
                taus.append(t); Hs.append(h)
    if not taus:
        return None, None
    taus = np.asarray(taus); Hs = np.asarray(Hs)

    # Sort & uniq on tau
    idx = np.argsort(taus)
    taus = taus[idx]; Hs = Hs[idx]
    # remove duplicate taus (keep first)
    m = np.ones_like(taus, dtype=bool)
    m[1:] = taus[1:] > taus[:-1]
    return taus[m], Hs[m]

def _rolling_slope(x, y, w):
    """rolling linear slope dy/dx with window w (odd int >=5)"""
    n = len(x)
    slopes = np.full(n, np.nan, dtype=float)
    half = w // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + (w - half))
        xx = x[lo:hi]; yy = y[lo:hi]
        if len(xx) < 3:
            continue
        A = np.vstack([xx, np.ones_like(xx)]).T
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        slopes[i] = coef[0]
    return slopes

def _iqr_mask(z, k=3.5):
    """keep points within k * IQR from Q1..Q3."""
    q1, q3 = np.nanpercentile(z, [25, 75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return np.isfinite(z)
    lo = q1 - k*iqr
    hi = q3 + k*iqr
    return (z >= lo) & (z <= hi) & np.isfinite(z)

def _plateau_from_slopes(slopes):
    """d_s = -2*slope; plateau = interquartile band; return stats + score."""
    ds = -2.0 * slopes
    ds = ds[np.isfinite(ds)]
    if ds.size < 8:
        return None
    # robust de-spike in ds
    keep = _iqr_mask(ds, k=3.5)
    ds = ds[keep]
    if ds.size < 8:
        return None

    q25, q50, q75 = np.nanpercentile(ds, [25, 50, 75])
    plateau = ds[(ds >= q25) & (ds <= q75)]
    if plateau.size < 6:
        plateau = ds

    mean = float(np.nanmean(plateau))
    spread = float(np.nanstd(plateau))
    dmax = float(np.nanmax(ds))
    dmed = float(q50)

    # stability score: lower spread is better; also penalize wild max
    score = spread + 0.01 * max(0.0, dmax - mean)
    return dict(ds_plateau_mean=mean,
                ds_plateau_spread=spread,
                ds_max=dmax,
                ds_median=dmed,
                ds_npts=int(ds.size),
                ds_score=float(score))

def robust_ds_from_heattrace(ht_path,
                             tau_bands=((1e-3,1.0),(1e-2,1.0),(1e-2,3.0),(3e-3,3.0)),
                             windows=(9,11,13,17)):
    taus, Hs = _load_tau_H(ht_path)
    if taus is None:
        return None, "no_data"

    # log transform; reject H outliers first
    x_all = np.log(taus)
    y_all = np.log(Hs)
    keep = _iqr_mask(y_all, k=4.0)
    x_all = x_all[keep]; y_all = y_all[keep]
    if x_all.size < 16:
        return None, "too_few_points_after_preclean"

    best = None
    best_cfg = None

    for (tmin, tmax) in tau_bands:
        # mask tau band in original (not log)
        mask_tau = (np.exp(x_all) >= tmin) & (np.exp(x_all) <= tmax)
        if mask_tau.sum() < 16:
            continue
        x = x_all[mask_tau]; y = y_all[mask_tau]

        # gentle smoothing in y via moving average over 3 points in log space
        if x.size >= 5:
            y_sm = y.copy()
            for i in range(1, len(y)-1):
                y_sm[i] = (y[i-1] + y[i] + y[i+1]) / 3.0
            y = y_sm

        for w in windows:
            if w % 2 == 0:  # ensure odd
                w += 1
            if len(x) < max(w, 9):
                continue
            slopes = _rolling_slope(x, y, w)
            # reject slope outliers
            keep_s = _iqr_mask(slopes[np.isfinite(slopes)], k=4.0)
            # rebuild cleaned slopes array with NaNs preserved
            ss = slopes.copy()
            finite_idx = np.where(np.isfinite(slopes))[0]
            mask_local = np.full_like(slopes, False, dtype=bool)
            mask_local[finite_idx[keep_s]] = True
            ss[~mask_local] = np.nan

            stats = _plateau_from_slopes(ss)
            if stats is None:
                continue

            # prefer lower score (spread), but also prefer more points
            score = stats["ds_score"] - 1e-4 * stats["ds_npts"]
            if (best is None) or (score < best["ds_score"] - 1e-6):
                best = stats
                best_cfg = dict(ds_window=int(w), ds_tau_min=float(tmin), ds_tau_max=float(tmax))

    if best is None:
        return None, "no_stable_plateau"

    best.update(best_cfg)
    return best, None

def update_meta_json(run_dir, ds_dict=None, reason=None):
    meta_path = os.path.join(run_dir, "meta_summary.json")
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except:
        meta = {}
    if ds_dict:
        meta.update(ds_dict)
        meta["_ds_backfilled_from"] = "heattrace_v2"
        meta.pop("_ds_fail_reason", None)
    elif reason:
        meta["_ds_fail_reason"] = reason
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

def main():
    ap = argparse.ArgumentParser(description="Robustly recompute ds_* from meta_heat_trace*.csv and patch meta_summary.json")
    ap.add_argument("--roots", nargs="+", required=True, help="root folders containing many run dirs")
    args = ap.parse_args()

    total = updated = failed = missing_ht = 0
    for root in args.roots:
        if not os.path.isdir(root):
            print(f"[WARN] root missing: {root}")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "meta_summary.json" not in filenames:
                continue
            total += 1
            # find a heat-trace
            cands = []
            for pat in ("meta_heat_trace.csv", "meta_heat_trace*.csv"):
                cands.extend(glob.glob(os.path.join(dirpath, pat)))
            cands = [c for c in cands if os.path.isfile(c)]
            if not cands:
                missing_ht += 1
                update_meta_json(dirpath, ds_dict=None, reason="no_heat_trace_found")
                continue
            ds, reason = robust_ds_from_heattrace(cands[0])
            if ds is None:
                failed += 1
                update_meta_json(dirpath, ds_dict=None, reason=reason or "unknown_failure")
            else:
                updated += 1
                update_meta_json(dirpath, ds_dict=ds, reason=None)

    print(f"[DONE] runs: {total} | updated: {updated} | failed: {failed} | no_heat_trace: {missing_ht}")

if __name__ == "__main__":
    main()