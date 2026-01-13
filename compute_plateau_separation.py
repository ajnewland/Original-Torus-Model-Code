# compute_plateau_separation.py
# CMD-friendly tool to quantify plateau separation gaps from (ax, ay) plateau points.
# No external deps beyond numpy/pandas. Works whether or not a component/label column exists.

import argparse, os, sys, math
import numpy as np
import pandas as pd
from collections import deque, defaultdict

def load_points(path):
    df = pd.read_csv(path)
    # Normalize column names
    cols = {c.strip().lower(): c for c in df.columns}
    # Required: ax, ay (case/space tolerant)
    ax_col = None
    ay_col = None
    for k,v in cols.items():
        if k in ("ax","a_x"):
            ax_col = v
        if k in ("ay","a_y"):
            ay_col = v
    if ax_col is None or ay_col is None:
        raise ValueError(f"{path}: CSV must contain columns 'ax' and 'ay' (found: {list(df.columns)})")

    pts = df[[ax_col, ay_col]].to_numpy(dtype=float)
    # Optional known component/label
    comp_col = None
    for k,v in cols.items():
        if k in ("comp","comp_id","component","label","region_id","cluster"):
            comp_col = v
            break
    labels = None
    if comp_col is not None:
        try:
            labels = df[comp_col].to_numpy()
        except Exception:
            labels = None
    return pts, labels, df

def estimate_grid_step(vals):
    """Estimate dominant grid spacing along one axis from unique sorted differences."""
    u = np.unique(np.round(vals, 12))
    if u.size < 2:
        return np.nan
    diffs = np.diff(u)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return np.nan
    # robust "mode": take median of the lower quartile to avoid occasional large gaps
    q = np.quantile(diffs, 0.35)
    return float(q)

def cluster_by_connectivity(pts, eps):
    """Simple grid-neighborhood BFS clustering using a spatial hash."""
    # Spatial hash: map rounded coords to index list
    r = max(8, int(abs(math.log10(eps))+3))  # rounding precision
    key = lambda x,y: (round(x, r), round(y, r))
    buckets = defaultdict(list)
    for i,(x,y) in enumerate(pts):
        buckets[key(x,y)].append(i)

    # Candidate neighbor offsets within a small stencil around eps
    # Use 3x3 stencil in key space
    def neighbor_indices(x, y):
        kx, ky = round(x, r), round(y, r)
        out = []
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                out.extend(buckets.get((kx+dx*round(eps, r), ky+dy*round(eps, r)), []))
        return out

    visited = np.zeros(len(pts), dtype=bool)
    comp_id = -1
    labels = np.full(len(pts), -1, dtype=int)
    for i in range(len(pts)):
        if visited[i]: continue
        comp_id += 1
        q = deque([i])
        visited[i] = True
        labels[i] = comp_id
        while q:
            u = q.popleft()
            x0,y0 = pts[u]
            # check all points within eps via buckets + fallback dense local check
            # We'll scan a small set of candidates by brute force threshold
            # (safe: your patches are modest in size).
            # Faster: kd-tree; but we stick to stdlib.
            for v in range(len(pts)):
                if visited[v]: continue
                x1,y1 = pts[v]
                if (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) <= eps*eps*1.05:
                    visited[v] = True
                    labels[v] = comp_id
                    q.append(v)
    return labels

def cluster_points(pts, existing_labels):
    if existing_labels is not None:
        # Normalize labels to consecutive ints
        _, inv = np.unique(existing_labels, return_inverse=True)
        return inv
    # infer grid step from data
    step_x = estimate_grid_step(pts[:,0])
    step_y = estimate_grid_step(pts[:,1])
    # Fallback: use overall nearest-neighbor heuristic if steps not found
    if not np.isfinite(step_x) or not np.isfinite(step_y):
        # crude NN spacing estimate
        if len(pts) < 3:
            eps = 1e-3
        else:
            # sample small subset
            samp = pts[np.random.choice(len(pts), min(500, len(pts)), replace=False)]
            dmins = []
            for i in range(len(samp)):
                d = np.sqrt(np.sum((samp - samp[i])**2, axis=1))
                d = np.partition(d, 1)[1]  # nearest neighbor (exclude self=0)
                dmins.append(d)
            eps = float(np.median(dmins)) * 1.2
    else:
        eps = float(min(step_x, step_y)) * 1.5
    labels = cluster_by_connectivity(pts, eps)
    return labels

def per_component_stats(pts, labels):
    comps = []
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        sub = pts[idx]
        cx, cy = np.mean(sub[:,0]), np.mean(sub[:,1])
        # radius estimate: RMS distance to centroid
        d2 = (sub[:,0]-cx)**2 + (sub[:,1]-cy)**2
        radius = float(np.sqrt(np.mean(d2))) if len(sub)>1 else 0.0
        comps.append((cid, len(sub), cx, cy, radius))
    comp_df = pd.DataFrame(comps, columns=["comp_id","n_points","ax_centroid","ay_centroid","radius_rms"])
    # nearest-neighbor distance between centroids
    C = comp_df[["ax_centroid","ay_centroid"]].to_numpy()
    nn = []
    for i in range(len(C)):
        if len(C)==1:
            dmin = np.nan
        else:
            d = np.sqrt(np.sum((C - C[i])**2, axis=1))
            dmin = float(np.partition(d, 1)[1])  # nearest other centroid
        nn.append(dmin)
    comp_df["dmin_to_other_centroid"] = nn
    return comp_df

def outcome_from_stats(comp_df):
    # Robust plateau size proxy = median radius; robustness vs spacing = median dmin
    radius_med = float(np.nanmedian(comp_df["radius_rms"])) if len(comp_df) else np.nan
    dmin_med   = float(np.nanmedian(comp_df["dmin_to_other_centroid"])) if len(comp_df) else np.nan
    # Rule of thumb: isolated if separation >> size
    if not np.isfinite(radius_med) or not np.isfinite(dmin_med):
        return "insufficient-data"
    ratio = dmin_med / max(radius_med, 1e-12)
    if ratio >= 5.0:
        return "isolated-attractor"
    elif ratio >= 2.5:
        return "marginal"
    else:
        return "diffuse-tiling"

def summarize_region(path, outdir):
    region = os.path.splitext(os.path.basename(path))[0]
    try:
        pts, labels_in, _ = load_points(path)
    except Exception as e:
        print(f"[ERROR] {region}: {e}")
        return None, None
    labels = cluster_points(pts, labels_in)
    comp_df = per_component_stats(pts, labels)
    os.makedirs(outdir, exist_ok=True)
    comp_out = os.path.join(outdir, f"{region}_components.csv")
    comp_df.to_csv(comp_out, index=False)
    # Summary
    out = {
        "region": region,
        "n_components": int(len(comp_df)),
        "n_points": int(len(pts)),
        "radius_med": float(np.nanmedian(comp_df["radius_rms"])) if len(comp_df) else np.nan,
        "radius_mean": float(np.nanmean(comp_df["radius_rms"])) if len(comp_df) else np.nan,
        "dmin_med": float(np.nanmedian(comp_df["dmin_to_other_centroid"])) if len(comp_df) else np.nan,
        "dmin_mean": float(np.nanmean(comp_df["dmin_to_other_centroid"])) if len(comp_df) else np.nan,
        "dmin_min": float(np.nanmin(comp_df["dmin_to_other_centroid"])) if len(comp_df) else np.nan,
        "dmin_max": float(np.nanmax(comp_df["dmin_to_other_centroid"])) if len(comp_df) else np.nan,
    }
    out["outcome"] = outcome_from_stats(comp_df)
    return out, comp_out

def main():
    ap = argparse.ArgumentParser(description="Compute plateau separation gaps from plateau point CSVs.")
    ap.add_argument("--csv", nargs="+", required=True,
                    help="One or more CSVs (plateau point sets). Each must have ax, ay columns.")
    ap.add_argument("--outdir", required=True, help="Output directory for summaries and per-component CSVs.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = []
    comp_paths = []
    for p in args.csv:
        out, comp_path = summarize_region(p, args.outdir)
        if out is not None:
            rows.append(out)
            comp_paths.append(comp_path)

    if not rows:
        print("[ERROR] No valid inputs processed.")
        sys.exit(2)

    summary = pd.DataFrame(rows, columns=[
        "region","n_components","n_points",
        "radius_med","radius_mean",
        "dmin_med","dmin_mean","dmin_min","dmin_max",
        "outcome"
    ])
    summary_path = os.path.join(args.outdir, "plateau_separation_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("=== Plateau separation summary ===")
    print(summary.to_string(index=False))
    print(f"\n[WROTE] {summary_path}")
    for cp in comp_paths:
        if cp is not None:
            print(f"[WROTE] {cp}")

if __name__ == "__main__":
    main()