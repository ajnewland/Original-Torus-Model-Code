#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
search_signflip_hot_latent.py
Scan (scale, tilt_ax, tilt_ay) deformations of a latent surface z(ax, ay) and
stop when we detect:
  (A) a sign flip: sign(z_hot) != sign(z_orig) for at least one point,
and optionally
  (B) a curvature flip near the zero-crossing: a local quadratic fit shows both
      Hessian eigenvalues > 0 (new convex basin) where previously not convex.

Input latent must contain: ax, ay, z  (z_pred is ignored if present).
Hot deformation: z_hot = scale*z + tilt_ax*(ax-ax0) + tilt_ay*(ay-ay0)

Outputs (on first hit):
  - <outdir>/latent_hot_hit.csv      (the deformed latent at the hit)
  - <outdir>/hit_summary.csv         (one-line summary)
  - <outdir>/scan_log.csv            (rolling log of tried params & counts)
If the scan completes with no sign flips, only scan_log.csv is written.

Tested with Python ≥ 3.9. Uses only numpy/pandas.
"""

import argparse, os, sys, math, json
import numpy as np
import pandas as pd

def load_latent(path):
    df = pd.read_csv(path)
    # Normalize column names just in case
    df = df.rename(columns={c: c.strip() for c in df.columns})
    need = {"ax","ay","z"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"Input CSV must contain columns: ax, ay, z. Found: {list(df.columns)}")
    # Coerce numeric
    for c in ["ax","ay","z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ax","ay","z"]).copy()
    return df

def deform(df, scale, tilt_ax, tilt_ay, ax0, ay0):
    # z_hot = scale*z + tilt_ax*(ax-ax0) + tilt_ay*(ay-ay0)
    z_hot = scale * df["z"].values + tilt_ax * (df["ax"].values - ax0) + tilt_ay * (df["ay"].values - ay0)
    out = df.copy()
    out["z_hot"] = z_hot
    return out

def in_box(df, box):
    if box is None:
        return np.ones(len(df), dtype=bool)
    ax_min, ax_max, ay_min, ay_max = box
    return (df["ax"].values >= ax_min) & (df["ax"].values <= ax_max) & \
           (df["ay"].values >= ay_min) & (df["ay"].values <= ay_max)

def sign(x):
    return np.where(x>0, 1, np.where(x<0, -1, 0))

def knn_quadratic_hessian(ax, ay, z, idx, k=12):
    """
    Fit z ≈ a0 + a1*dx + a2*dy + a3*dx^2 + a4*dx*dy + a5*dy^2
    around point idx using k nearest neighbours, return Hessian eigenvalues.
    """
    x0, y0 = ax[idx], ay[idx]
    # distances
    d2 = (ax - x0)**2 + (ay - y0)**2
    # exclude itself (still okay if included; but we’ll keep it)
    order = np.argsort(d2)
    nbrs = order[:max(k,6)]
    dx = ax[nbrs] - x0
    dy = ay[nbrs] - y0
    M  = np.column_stack([np.ones_like(dx), dx, dy, dx*dx, dx*dy, dy*dy])
    try:
        coef, *_ = np.linalg.lstsq(M, z[nbrs], rcond=None)
        # Hessian for this quadratic is:
        # H = [[d2z/dx2, d2z/dxdy],
        #      [d2z/dydx, d2z/dy2]] = [[2*a3, a4], [a4, 2*a5]]
        a3, a4, a5 = coef[3], coef[4], coef[5]
        H = np.array([[2*a3, a4],[a4, 2*a5]], dtype=float)
        eig = np.linalg.eigvalsh(H)
        return eig, H
    except np.linalg.LinAlgError:
        return None, None

def classify_curvature(eigs):
    if eigs is None: return "unknown"
    lam1, lam2 = eigs[0], eigs[1]
    if lam1 > 0 and lam2 > 0: return "convex"
    if lam1 < 0 and lam2 < 0: return "concave"
    return "saddle"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", required=True, help="Path to base latent CSV (ax, ay, z).")
    ap.add_argument("--outdir",  required=True, help="Output directory.")
    # Scan ranges
    ap.add_argument("--scale_min",  type=float, default=1.02)
    ap.add_argument("--scale_max",  type=float, default=1.30)
    ap.add_argument("--scale_steps",type=int,   default=15)
    ap.add_argument("--tilt_ax_min",type=float, default=-0.06)
    ap.add_argument("--tilt_ax_max",type=float, default=+0.06)
    ap.add_argument("--tilt_ax_steps",type=int, default=13)
    ap.add_argument("--tilt_ay_min",type=float, default=-0.06)
    ap.add_argument("--tilt_ay_max",type=float, default=+0.06)
    ap.add_argument("--tilt_ay_steps",type=int, default=13)
    # Center for tilt (defaults to mid-range of the data)
    ap.add_argument("--ax0", type=float, default=None)
    ap.add_argument("--ay0", type=float, default=None)
    # Optional AOI box where we demand sign flips (dark-band style)
    ap.add_argument("--box", type=float, nargs=4, default=None,
                    help="ax_min ax_max ay_min ay_max (restrict sign-flip check to this box)")
    # Stop conditions
    ap.add_argument("--min_flips", type=int, default=1,
                    help="Stop when at least this many sign flips occur (in AOI if --box set).")
    ap.add_argument("--epsilon_zero", type=float, default=0.03,
                    help="Near-zero |z_hot| threshold to probe curvature flip.")
    ap.add_argument("--curvature_k", type=int, default=12,
                    help="kNN size for local quadratic fit.")
    ap.add_argument("--require_convex", action="store_true",
                    help="If set, only stop when we also detect a convex basin (both Hessian eigs > 0) near a zero crossing.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df0 = load_latent(args.latent)
    ax_vals = df0["ax"].values
    ay_vals = df0["ay"].values
    z0      = df0["z"].values

    # Default centers
    ax0 = args.ax0 if args.ax0 is not None else 0.5*(ax_vals.min()+ax_vals.max())
    ay0 = args.ay0 if args.ay0 is not None else 0.5*(ay_vals.min()+ay_vals.max())

    # Parameter grids
    scales   = np.linspace(args.scale_min,   args.scale_max,   args.scale_steps)
    tilts_ax = np.linspace(args.tilt_ax_min, args.tilt_ax_max, args.tilt_ax_steps)
    tilts_ay = np.linspace(args.tilt_ay_min, args.tilt_ay_max, args.tilt_ay_steps)

    box_mask_all = in_box(df0, args.box)

    # Rolling log
    log_rows = []
    def flush_log():
        if not log_rows: return
        pd.DataFrame(log_rows).to_csv(os.path.join(args.outdir,"scan_log.csv"), index=False)

    base_sign = sign(z0)

    hit_saved = False
    for s in scales:
        for ta in tilts_ax:
            for tb in tilts_ay:
                df_hot = deform(df0, s, ta, tb, ax0, ay0)
                z_hot = df_hot["z_hot"].values
                hot_sign = sign(z_hot)

                # AOI filter if requested
                inA = box_mask_all
                flips_mask = (hot_sign != base_sign) & inA
                n_flips = int(flips_mask.sum())

                # Curvature check near zero crossing (only where flip occurs and |z_hot| small)
                n_convex_hits = 0
                if n_flips > 0 and args.require_convex:
                    idxs = np.where(flips_mask & (np.abs(z_hot) <= args.epsilon_zero))[0]
                    for idx in idxs:
                        eigs_hot, _ = knn_quadratic_hessian(ax_vals, ay_vals, z_hot, idx, k=args.curvature_k)
                        eigs_base, _= knn_quadratic_hessian(ax_vals, ay_vals, z0,    idx, k=args.curvature_k)
                        curv_hot  = classify_curvature(eigs_hot)
                        curv_base = classify_curvature(eigs_base)
                        if curv_hot == "convex" and curv_base != "convex":
                            n_convex_hits += 1

                # Log this trial
                row = dict(scale=s, tilt_ax=ta, tilt_ay=tb,
                           n_flips=n_flips, n_convex_hits=n_convex_hits)
                log_rows.append(row)

                # Stop condition
                stop = (n_flips >= args.min_flips)
                if args.require_convex:
                    stop = stop and (n_convex_hits >= 1)

                if stop:
                    # Save hot latent and a one-line summary
                    out_lat = os.path.join(args.outdir, "latent_hot_hit.csv")
                    df_hot.to_csv(out_lat, index=False)
                    out_sum = os.path.join(args.outdir, "hit_summary.csv")
                    pd.DataFrame([{
                        "scale": s, "tilt_ax": ta, "tilt_ay": tb,
                        "n_flips": n_flips, "n_convex_hits": n_convex_hits,
                        "ax0": ax0, "ay0": ay0,
                        "box": json.dumps(args.box) if args.box else ""
                    }]).to_csv(out_sum, index=False)
                    flush_log()
                    print("=== HIT FOUND ===")
                    print(f"scale={s:.6f}  tilt_ax={ta:.6f}  tilt_ay={tb:.6f}")
                    print(f"n_flips={n_flips}  n_convex_hits={n_convex_hits}")
                    print(f"[WROTE] {out_lat}")
                    print(f"[WROTE] {out_sum}")
                    hit_saved = True
                    return

    # No hit — write log and exit
    flush_log()
    if not hit_saved:
        print("[INFO] Scan complete. No sign flips met the stop criteria.")
        print(f"[WROTE] {os.path.join(args.outdir,'scan_log.csv')}")
        if args.require_convex:
            print("Hint: relax --require_convex or widen ranges/steps; you can also enlarge --epsilon_zero or increase --curvature_k.")
        else:
            print("Hint: widen --scale/--tilt ranges or decrease --min_flips.")
    return

if __name__ == "__main__":
    main()