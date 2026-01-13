#!/usr/bin/env python3

# Finds flat-band (plateau) regions on torus grids (ax, ay, z or z_pred).

# CMD usage example (Windows):

#   python "C:\...\plateau_scan.py" --grids "C:\...\boson_refined_W\grid_ax_ay_z.csv" ^

#       "C:\...\boson_refined_Z\grid_ax_ay_z.csv" "C:\...\boson_refined_H\grid_ax_ay_z.csv" ^

#       --outdir "C:\...\Predicted Masses\plateau_scan" --auto

#

# Outputs:

#   outdir/plateau_summary.csv

#   outdir/<gridname>_plateau_points.csv

#   outdir/<gridname>_plateau_map.png



import os, sys, argparse, math

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt



def load_grid(path):

    df = pd.read_csv(path)

    # Accept either z or z_pred

    zcol = 'z' if 'z' in df.columns else ('z_pred' if 'z_pred' in df.columns else None)

    if zcol is None or ('ax' not in df.columns) or ('ay' not in df.columns):

        raise ValueError(f"[{os.path.basename(path)}] CSV must contain columns ax, ay, and z or z_pred.")

    ax_vals = np.sort(df['ax'].unique())

    ay_vals = np.sort(df['ay'].unique())

    ax_index = {v:i for i,v in enumerate(ax_vals)}

    ay_index = {v:i for i,v in enumerate(ay_vals)}

    Z = np.full((ay_vals.size, ax_vals.size), np.nan, dtype=float)

    for _, row in df.iterrows():

        i = ay_index[row['ay']]

        j = ax_index[row['ax']]

        Z[i,j] = float(row[zcol])

    if np.isnan(Z).any():

        raise ValueError(f"[{os.path.basename(path)}] Grid has missing cells; ensure full rectilinear grid.")

    return ax_vals, ay_vals, Z, zcol



def finite_diffs(ax, ay, Z):

    # spacing (assume nearly uniform)

    if ax.size < 2 or ay.size < 2:

        raise ValueError("Grid too small for derivatives.")

    dx = float(np.median(np.diff(ax)))

    dy = float(np.median(np.diff(ay)))

    # gradients

    Zy, Zx = np.gradient(Z, dy, dx, edge_order=2)  # note: axis 0 is y, axis 1 is x

    # Hessian pieces

    Zxy_y, Zxx = np.gradient(Zx, dy, dx, edge_order=2)

    Zyy, Zxy_x = np.gradient(Zy, dy, dx, edge_order=2)

    Zxy = 0.5*(Zxy_y + Zxy_x)

    return Zx, Zy, Zxx, Zxy, Zyy



def principal_curvatures(Zxx, Zxy, Zyy):

    # eigenvalues of [[Zxx, Zxy],[Zxy, Zyy]]

    # closed form: λ = (tr ± sqrt(tr^2 - 4 det))/2

    tr = Zxx + Zyy

    det = Zxx*Zyy - Zxy*Zxy

    disc = np.clip(tr*tr - 4.0*det, 0.0, None)

    sqrt_disc = np.sqrt(disc)

    lam1 = 0.5*(tr + sqrt_disc)

    lam2 = 0.5*(tr - sqrt_disc)

    kmax = np.maximum(np.abs(lam1), np.abs(lam2))

    return lam1, lam2, kmax, tr, det



def label_components(mask):

    # simple 4-connected component labeling

    h, w = mask.shape

    labels = np.zeros_like(mask, dtype=np.int32)

    current = 0

    for i in range(h):

        for j in range(w):

            if mask[i,j] and labels[i,j] == 0:

                current += 1

                # BFS

                q = [(i,j)]

                labels[i,j] = current

                while q:

                    y,x = q.pop()

                    for ny, nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):

                        if 0<=ny<h and 0<=nx<w and mask[ny,nx] and labels[ny,nx]==0:

                            labels[ny,nx]=current

                            q.append((ny,nx))

    return labels, current



def summarize_component(label_id, labels, ax, ay, Z, grad_mag, kmax):

    ys, xs = np.where(labels==label_id)

    if ys.size == 0:

        return None

    ax_pts = ax[xs]

    ay_pts = ay[ys]

    z_pts  = Z[ys, xs]

    g_pts  = grad_mag[ys, xs]

    k_pts  = kmax[ys, xs]

    summary = {

        "n_pixels": int(ys.size),

        "ax_min": float(ax_pts.min()), "ax_max": float(ax_pts.max()),

        "ay_min": float(ay_pts.min()), "ay_max": float(ay_pts.max()),

        "ax_med": float(np.median(ax_pts)), "ay_med": float(np.median(ay_pts)),

        "z_med": float(np.median(z_pts)), "z_range": float(z_pts.max()-z_pts.min()),

        "grad_med": float(np.median(g_pts)), "curv_med": float(np.median(k_pts)),

    }

    return summary, ys, xs



def analyze_grid(path, outdir, grad_thresh=None, curv_thresh=None, auto=False, boundary_trim=1, gridname_hint=None):

    ax, ay, Z, zcol = load_grid(path)

    Zx, Zy, Zxx, Zxy, Zyy = finite_diffs(ax, ay, Z)

    grad_mag = np.hypot(Zx, Zy)

    lam1, lam2, kmax, tr, det = principal_curvatures(Zxx, Zxy, Zyy)



    # classify extrema (positive/negative definite Hessian)

    eps_det = 1e-12

    is_min = (det > eps_det) & (tr > 0)

    is_max = (det > eps_det) & (tr < 0)



    # thresholds

    if auto or (grad_thresh is None or curv_thresh is None):

        # adaptive: use low percentiles of observed magnitudes

        gth = np.percentile(grad_mag, 12.0)

        kth = np.percentile(kmax,     25.0)

        if grad_thresh is None: grad_thresh = float(gth)

        if curv_thresh is None: curv_thresh = float(kth)



    # plateau mask: small gradient & small curvature; remove boundaries + true extrema

    mask = (grad_mag <= grad_thresh) & (kmax <= curv_thresh)

    h, w = mask.shape

    if boundary_trim > 0:

        mask[:boundary_trim,:] = False

        mask[-boundary_trim:,:]= False

        mask[:,:boundary_trim] = False

        mask[:,-boundary_trim:]= False

    mask = mask & (~is_min) & (~is_max)



    labels, ncomp = label_components(mask)



    # Prepare outputs

    base = os.path.splitext(os.path.basename(path))[0]

    gridname = gridname_hint if gridname_hint else base

    pts_csv = os.path.join(outdir, f"{gridname}_plateau_points.csv")

    png     = os.path.join(outdir, f"{gridname}_plateau_map.png")



    # Save points CSV

    rows = []

    for lbl in range(1, ncomp+1):

        ys, xs = np.where(labels==lbl)

        for y,x in zip(ys, xs):

            rows.append({

                "ax": ax[x], "ay": ay[y], "z": Z[y,x],

                "grad": grad_mag[y,x], "kmax": kmax[y,x],

                "label": lbl

            })

    df_pts = pd.DataFrame(rows)

    df_pts.to_csv(pts_csv, index=False)



    # Build summary entries

    comps = []

    for lbl in range(1, ncomp+1):

        summ = summarize_component(lbl, labels, ax, ay, Z, grad_mag, kmax)

        if summ is None: continue

        info, ys, xs = summ

        info.update({

            "grid": gridname,

            "file": path,

            "grad_thresh": grad_thresh,

            "curv_thresh": curv_thresh,

            "n_ax": int(len(ax)), "n_ay": int(len(ay)),

        })

        comps.append(info)



    # PNG overlay

    fig, axp = plt.subplots(figsize=(6,5), dpi=150)

    im = axp.imshow(Z, origin='lower',

                    extent=[ax.min(), ax.max(), ay.min(), ay.max()],

                    aspect='auto')

    plt.colorbar(im, ax=axp, label='z')

    # outline plateau pixels

    yy, xx = np.where(mask)

    axp.scatter(ax[xx], ay[yy], s=4, marker='s', linewidths=0, alpha=0.7, label='plateau')

    axp.set_xlabel('a_x'); axp.set_ylabel('a_y')

    axp.set_title(f"{gridname}: flat-band candidates\n"

                  f"grad≤{grad_thresh:.3g}, curv≤{curv_thresh:.3g}, comps={len(comps)}")

    axp.legend(loc='best', fontsize=8)

    fig.tight_layout()

    fig.savefig(png)

    plt.close(fig)



    return comps, pts_csv, png, grad_thresh, curv_thresh



def main():

    ap = argparse.ArgumentParser(description="Detect flat-band (plateau) modes on torus grids.")

    ap.add_argument("--grids", nargs="+", required=True, help="One or more grid_ax_ay_z.csv files.")

    ap.add_argument("--outdir", required=True)

    ap.add_argument("--grad_thresh", type=float, default=None, help="Absolute threshold on |∇z|.")

    ap.add_argument("--curv_thresh", type=float, default=None, help="Absolute threshold on max|principal curvature|.")

    ap.add_argument("--auto", action="store_true", help="Use adaptive percentile thresholds (recommended).")

    ap.add_argument("--boundary_trim", type=int, default=1, help="Trim N cells off each boundary before detection.")

    args = ap.parse_args()



    os.makedirs(args.outdir, exist_ok=True)

    all_rows = []

    for g in args.grids:

        gridname_hint = None

        # Try to pick parent dir name as tag (e.g., boson_refined_W)

        try:

            gridname_hint = os.path.basename(os.path.dirname(g))

        except Exception:

            pass

        try:

            comps, pts_csv, png, gth, kth = analyze_grid(

                g, args.outdir, grad_thresh=args.grad_thresh, curv_thresh=args.curv_thresh,

                auto=args.auto, boundary_trim=args.boundary_trim, gridname_hint=gridname_hint

            )

            if not comps:

                all_rows.append({

                    "grid": gridname_hint or os.path.basename(g),

                    "file": g, "n_components": 0, "grad_thresh": gth, "curv_thresh": kth

                })

            else:

                for c in comps:

                    row = {

                        "grid": c["grid"], "file": c["file"], "n_components": len(comps),

                        "ax_med": c["ax_med"], "ay_med": c["ay_med"],

                        "ax_min": c["ax_min"], "ax_max": c["ax_max"],

                        "ay_min": c["ay_min"], "ay_max": c["ay_max"],

                        "z_med": c["z_med"], "z_range": c["z_range"],

                        "grad_med": c["grad_med"], "curv_med": c["curv_med"],

                        "n_pixels": c["n_pixels"],

                        "grad_thresh": c["grad_thresh"], "curv_thresh": c["curv_thresh"],

                        "n_ax": c["n_ax"], "n_ay": c["n_ay"],

                    }

                    all_rows.append(row)

        except Exception as e:

            print(f"[ERROR] {g}: {e}", file=sys.stderr)



    summary_path = os.path.join(args.outdir, "plateau_summary.csv")

    pd.DataFrame(all_rows).to_csv(summary_path, index=False)

    print(f"[DONE] Summary -> {summary_path}")

    print(f"[NOTE] Per-grid plateau points and PNG overlays were saved next to the summary.")



if __name__ == "__main__":

    main()