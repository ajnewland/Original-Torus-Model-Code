#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os, math
import numpy as np
import pandas as pd

PAIR_PRESETS = {
    # sensible defaults for this dataset
    "default": [
        ("scale","tilt_ax"),
        ("scale","tilt_ay"),
        ("tilt_ax","tilt_ay"),
        ("anis_ax","anis_ay"),
        ("shear","rot_deg"),
        ("bias","scale")
    ]
}

def load_csv(path):
    df = pd.read_csv(path)
    req = {'scale','tilt_ax','tilt_ay','shear','rot_deg','bias',
           'anis_ax','anis_ay','min_abs_z','n_neg','n_pos','flip'}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Input missing columns: {sorted(missing)}")
    return df

def choose_mask(df, use_flip=True, eps=0.0):
    """
    If use_flip: mask rows with flip == 1.
    Else:        mask rows with min_abs_z <= eps.
    """
    if use_flip:
        m = (df['flip'] == 1)
        note = "flip==1"
    else:
        m = (df['min_abs_z'] <= eps)
        note = f"min_abs_z<= {eps:g}"
    return m.values, note

def bin_grid(x, y, nbx, nby, xrng=None, yrng=None):
    """
    Returns:
      H (nby, nbx) as boolean occupancy
      xedges (nbx+1,), yedges (nby+1,)
      ix, iy indices (for the points)
    """
    if xrng is None: xrng = (np.nanmin(x), np.nanmax(x))
    if yrng is None: yrng = (np.nanmin(y), np.nanmax(y))
    # add tiny pad to include max
    xrng = (xrng[0], xrng[1] + 1e-12)
    yrng = (yrng[0], yrng[1] + 1e-12)

    H, xedges, yedges = np.histogram2d(y, x, bins=[nby, nbx], range=[yrng, xrng])
    H = (H > 0)

    # compute per-point bin indices
    ix = np.floor((x - xrng[0]) / (xrng[1]-xrng[0]) * nbx).astype(int)
    iy = np.floor((y - yrng[0]) / (yrng[1]-yrng[0]) * nby).astype(int)
    ix = np.clip(ix, 0, nbx-1)
    iy = np.clip(iy, 0, nby-1)

    return H, xedges, yedges, ix, iy

def connected_components_bool(H):
    """
    4-connected components on boolean grid H (nby, nbx).
    Returns comp_id grid (int, -1 for background) and list of components,
    each with list of (iy,ix) cells.
    """
    nby, nbx = H.shape
    comp = -np.ones_like(H, dtype=int)
    comps = []
    cid = 0

    for iy in range(nby):
        for ix in range(nbx):
            if not H[iy, ix] or comp[iy, ix] != -1:
                continue
            # BFS
            q = [(iy, ix)]
            comp[iy, ix] = cid
            cells = [(iy, ix)]
            while q:
                cy, cx = q.pop()
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = cy+dy, cx+dx
                    if 0 <= ny < nby and 0 <= nx < nbx:
                        if H[ny, nx] and comp[ny, nx] == -1:
                            comp[ny, nx] = cid
                            q.append((ny, nx))
                            cells.append((ny, nx))
            comps.append(cells)
            cid += 1
    return comp, comps

def summarize_components(pairname, H, xedges, yedges, comp_grid, comps, ix, iy, x, y):
    """
    Build a tidy DataFrame of plateau components with geometric stats.
    area_cells = number of occupied bins in the component
    x/y range (bin) and physical bbox (in coordinate units)
    centroid (in coordinate units)
    """
    rows = []
    nby, nbx = H.shape

    # pre-compute per-bin center coordinates
    xcent = 0.5*(xedges[:-1] + xedges[1:])
    ycent = 0.5*(yedges[:-1] + yedges[1:])

    # map each point to a component id
    p_comp = comp_grid[iy, ix]

    # list of points per component (for n_points)
    # use integer dict
    pts_in_comp = {}
    for cid in range(len(comps)):
        pts_in_comp[cid] = 0
    for c in p_comp:
        if c >= 0:
            pts_in_comp[c] += 1

    for cid, cells in enumerate(comps):
        ys = [c[0] for c in cells]
        xs = [c[1] for c in cells]
        y_min, y_max = min(ys), max(ys)
        x_min, x_max = min(xs), max(xs)
        area_cells = len(cells)

        # centroid in bin space -> convert to coord via centers
        cx = np.mean([xcent[ix_] for ix_ in xs])
        cy = np.mean([ycent[iy_] for iy_ in ys])

        # also gather bounding box in coordinates
        # xedges is oriented for x coordinate, yedges for y coordinate
        x0 = xedges[x_min]
        x1 = xedges[x_max+1]
        y0 = yedges[y_min]
        y1 = yedges[y_max+1]

        rows.append({
            "pair": pairname,
            "comp_id": cid,
            "area_cells": area_cells,
            "bin_x_min": x_min, "bin_x_max": x_max,
            "bin_y_min": y_min, "bin_y_max": y_max,
            "x_min": x0, "x_max": x1,
            "y_min": y0, "y_max": y1,
            "x_centroid": cx, "y_centroid": cy,
            "n_points": int(pts_in_comp[cid]),
        })

    df = pd.DataFrame(rows)
    return df

def min_pairwise_centroid_dist(df_comp):
    """Add nearest-neighbor centroid distance per component (within same pair)."""
    out = df_comp.copy()
    out["dmin_centroid"] = np.nan
    # group by pair (each pair separately)
    for pair, g in out.groupby("pair"):
        xy = g[["x_centroid","y_centroid"]].to_numpy()
        dmin = np.full(len(g), np.nan)
        for i in range(len(g)):
            dx = xy[:,0] - xy[i,0]
            dy = xy[:,1] - xy[i,1]
            d = np.hypot(dx, dy)
            d[i] = np.inf
            dmin[i] = np.min(d) if np.isfinite(d).any() else np.nan
        out.loc[g.index, "dmin_centroid"] = dmin
    return out

def run_pair(df, colx, coly, mask, nbx, nby, outdir, tag):
    pairname = f"{colx}_vs_{coly}"
    os.makedirs(outdir, exist_ok=True)

    x = df.loc[mask, colx].to_numpy()
    y = df.loc[mask, coly].to_numpy()
    if x.size == 0:
        return None, None, pairname

    H, xedges, yedges, ix, iy = bin_grid(x, y, nbx, nby)
    if not H.any():
        return None, None, pairname

    comp_grid, comps = connected_components_bool(H)
    df_comp = summarize_components(pairname, H, xedges, yedges, comp_grid, comps, ix, iy, x, y)

    # write per-pair components
    comp_csv = os.path.join(outdir, f"{tag}_{pairname}_components.csv")
    df_comp_sorted = df_comp.sort_values(["area_cells","n_points"], ascending=False)
    df_comp_sorted.to_csv(comp_csv, index=False)

    # per-pair summary
    summary = {
        "pair": pairname,
        "n_components": int(len(comps)),
        "n_points_total": int(x.size),
        "area_cells_med": float(df_comp["area_cells"].median()) if len(df_comp) else 0.0,
        "area_cells_mean": float(df_comp["area_cells"].mean()) if len(df_comp) else 0.0,
        "n_points_med": float(df_comp["n_points"].median()) if len(df_comp) else 0.0,
        "n_points_mean": float(df_comp["n_points"].mean()) if len(df_comp) else 0.0,
    }
    return df_comp_sorted, summary, pairname

def parse_pairs_arg(pairs_arg, df_cols):
    if pairs_arg.lower() in PAIR_PRESETS:
        pairs = PAIR_PRESETS[pairs_arg.lower()]
    else:
        # user gave comma-separated list like: scale:tilt_ax,anis_ax:anis_ay
        pairs = []
        for tok in pairs_arg.split(","):
            tok = tok.strip()
            if not tok: continue
            if ":" not in tok:
                raise ValueError(f"Bad --pairs token: '{tok}'. Use 'x:y'.")
            a,b = [t.strip() for t in tok.split(":",1)]
            pairs.append((a,b))
    # validate
    for a,b in pairs:
        if a not in df_cols or b not in df_cols:
            raise ValueError(f"Pair uses unknown columns: {a},{b}")
    return pairs

def main():
    ap = argparse.ArgumentParser(description="Analyze zero-margin plateaus (sign flips or near zero) on 2D parameter pairs.")
    ap.add_argument("--csv", required=True, help="Path to zero_margin_coarse.csv")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--pairs", default="default",
                    help="Either 'default' or custom comma list like 'scale:tilt_ax,anis_ax:anis_ay'")
    ap.add_argument("--bins", type=int, default=120, help="Number of bins per axis (square grid)")
    ap.add_argument("--use_flip", action="store_true", help="Use flip==1 rows (else use min_abs_z<=eps).")
    ap.add_argument("--eps", type=float, default=0.0, help="Threshold for min_abs_z when --use_flip is not set.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_csv(args.csv)
    mask, note = choose_mask(df, use_flip=args.use_flip, eps=args.eps)
    pairs = parse_pairs_arg(args.pairs, df.columns)

    all_comp = []
    summaries = []
    tag = "flipmask" if args.use_flip else f"eps{args.eps:g}"
    for (cx, cy) in pairs:
        comp, summ, pname = run_pair(df, cx, cy, mask, args.bins, args.bins, args.outdir, tag)
        if comp is None:
            continue
        all_comp.append(comp)
        summaries.append(summ)

    if not summaries:
        print("[INFO] No components found with given mask/pairs.")
        return

    df_comp_all = pd.concat(all_comp, ignore_index=True)
    df_comp_all = min_pairwise_centroid_dist(df_comp_all)
    df_comp_all.to_csv(os.path.join(args.outdir, f"{tag}_ALL_components.csv"), index=False)

    df_sum = pd.DataFrame(summaries)
    # add robust gap metrics per pair
    gap_rows = []
    for pair, g in df_comp_all.groupby("pair"):
        if len(g):
            gap_rows.append({
                "pair": pair,
                "dmin_centroid_med": float(g["dmin_centroid"].median()),
                "dmin_centroid_mean": float(g["dmin_centroid"].mean()),
                "dmin_centroid_min": float(g["dmin_centroid"].min()),
                "dmin_centroid_max": float(g["dmin_centroid"].max()),
                "area_cells_med": float(g["area_cells"].median()),
                "area_cells_mean": float(g["area_cells"].mean()),
                "n_points_med": float(g["n_points"].median()),
                "n_points_mean": float(g["n_points"].mean()),
            })
    df_gap = pd.DataFrame(gap_rows)
    df_sum = df_sum.merge(df_gap, on="pair", how="left")
    df_sum.to_csv(os.path.join(args.outdir, f"{tag}_summary.csv"), index=False)

    print("=== analyze_zero_margin_v2: DONE ===")
    print(f"[MASK] {note}")
    print(f"[OUT]  {args.outdir}")
    print(f"[ALL COMPONENTS] {tag}_ALL_components.csv")
    print(f"[SUMMARY]        {tag}_summary.csv")

if __name__ == "__main__":
    main()