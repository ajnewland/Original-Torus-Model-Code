# diagnose_seesaw.py
# Analyse cycle-shift sin^2(theta_W) data for a geometric "see-saw":
# - Cluster sin^2 into two bands (~0.231 light, ~0.249 heavy)
# - Report cluster stats, membership by shift (dx), and correlation with condG
# - If two CSVs are given (e.g., ay=0.78 vs ay=0.86), report membership swaps

import argparse
from pathlib import Path
import csv
import math
import numpy as np

def load_cycles_csv(path):
    """Load CSV with columns at least: dx,dy,sin2_mean,condG (names case-insensitive)."""
    cols = None
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        rdr = csv.reader(f)
        # try to detect header
        header = next(rdr)
        # normalize header names
        hdr = [h.strip().lower() for h in header]
        # find column indices
        def idx(name):
            try:
                return hdr.index(name)
            except ValueError:
                return None
        ix_dx    = idx("dx")
        ix_dy    = idx("dy")
        ix_sin2  = idx("sin2_mean")
        ix_condg = idx("condg")
        if ix_dx is None or ix_sin2 is None:
            raise RuntimeError(f"{path}: missing required columns (need at least dx,sin2_mean).")
        # if header looked like data (no 'dx' etc), treat header as first row and reset reader
        if not any(h in ("dx","dy","sin2_mean","condg") for h in hdr):
            # no header; re-read including first line as data
            f.seek(0)
            rdr = csv.reader(f)
            ix_dx = 0; ix_dy = 1; ix_sin2 = 2; ix_condg = 3  # assume the order pasted earlier
        else:
            pass

        # iterate rows
        for r in rdr:
            if not r or all((x.strip()=="" for x in r)):
                continue
            try:
                dx = int(float(r[ix_dx]))
                dy = int(float(r[ix_dy])) if ix_dy is not None else 0
                s2 = float(r[ix_sin2])
                cg = float(r[ix_condg]) if ix_condg is not None else float("nan")
            except Exception:
                # skip malformed rows
                continue
            rows.append((dx, dy, s2, cg))
    if not rows:
        raise RuntimeError(f"{path}: no data rows parsed.")
    # sort by dx for readability
    rows.sort(key=lambda t: (t[1], t[0]))
    return rows

def two_cluster_1d(values, max_iter=50):
    """
    Minimal k=2 clustering on 1D values (no sklearn).
    Returns labels (0/1) and centers sorted so that c0 < c1.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise RuntimeError("No finite values to cluster.")
    # initialize centers as min and max
    c0, c1 = float(np.min(v)), float(np.max(v))
    if abs(c1 - c0) < 1e-12:
        # everything the same
        labels = np.zeros_like(values, dtype=int)
        return labels, (c0, c1)
    for _ in range(max_iter):
        # assign
        labels = np.array([0 if abs(x - c0) <= abs(x - c1) else 1 for x in values], dtype=int)
        # recompute
        v0 = [x for x,l in zip(values,labels) if l==0 and np.isfinite(x)]
        v1 = [x for x,l in zip(values,labels) if l==1 and np.isfinite(x)]
        if len(v0)==0 or len(v1)==0:
            # fall back: split at midpoint
            mid = 0.5*(c0 + c1)
            labels = np.array([0 if x<=mid else 1 for x in values], dtype=int)
            v0 = [x for x in values if np.isfinite(x) and x<=mid]
            v1 = [x for x in values if np.isfinite(x) and x> mid]
        nc0 = float(np.mean(v0)) if len(v0)>0 else c0
        nc1 = float(np.mean(v1)) if len(v1)>0 else c1
        if abs(nc0-c0)<1e-12 and abs(nc1-c1)<1e-12:
            break
        c0, c1 = nc0, nc1
    # ensure c0<c1 ordering
    if c0 > c1:
        labels = 1-labels
        c0, c1 = c1, c0
    return labels, (c0, c1)

def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    xm = x[m] - x[m].mean()
    ym = y[m] - y[m].mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    if denom == 0:
        return float("nan")
    return float((xm*ym).sum()/denom)

def describe_band(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"), 0, float("nan"), float("nan"))
    return (float(v.mean()), float(v.std(ddof=0)), int(v.size), float(np.min(v)), float(np.max(v)))

def run_one(label, rows):
    dx = [r[0] for r in rows]
    s2 = [r[2] for r in rows]
    cg = [r[3] for r in rows]
    # cluster into two bands
    labs, centers = two_cluster_1d(s2)
    c0, c1 = centers
    band0 = [s for s,l in zip(s2,labs) if l==0]
    band1 = [s for s,l in zip(s2,labs) if l==1]
    dx0   = [d for d,l in zip(dx,labs) if l==0]
    dx1   = [d for d,l in zip(dx,labs) if l==1]
    m0,s0,n0,lo0,hi0 = describe_band(band0)
    m1,s1,n1,lo1,hi1 = describe_band(band1)
    # correlation sin2 vs condG (expect NEGATIVE if lower sin2 ↔ higher condG)
    rho = corr(s2, cg)

    print(f"\n--- {label} ---")
    print(f"N rows = {len(rows)}")
    print(f"Clusters (k=2): c_low≈{c0:.6f} (n={n0}), c_high≈{c1:.6f} (n={n1})")
    print(f"  low-band stats:  mean={m0:.6f}  std={s0:.6f}  min={lo0:.6f}  max={hi0:.6f}")
    print(f"  high-band stats: mean={m1:.6f}  std={s1:.6f}  min={lo1:.6f}  max={hi1:.6f}")
    print(f"sin² vs condG correlation r = {rho:.3f}  (expect r<0 if see-saw tilt present)")
    print(f"dx in low-band:  {dx0}")
    print(f"dx in high-band: {dx1}")
    return {
        "dx": dx, "sin2": s2, "condG": cg,
        "labels": labs, "centers": (c0,c1),
        "dx_low": dx0, "dx_high": dx1
    }

def swap_report(A, B):
    # assumes both have same dx ordering
    dxA, labA = A["dx"], A["labels"]
    dxB, labB = B["dx"], B["labels"]
    # map dx->label
    mA = {d:int(l) for d,l in zip(dxA, labA)}
    mB = {d:int(l) for d,l in zip(dxB, labB)}
    common = sorted(set(mA.keys()) & set(mB.keys()))
    flips = [d for d in common if mA[d] != mB[d]]
    print("\n=== Swap / see-saw report (A→B) ===")
    print(f"Common shifts (dx): {len(common)}")
    print(f"Flips in band membership: {len(flips)} / {len(common)}  ({100.0*len(flips)/len(common):.1f}%)")
    if len(flips):
        print(f"dx that flipped: {flips}")
    else:
        print("No flips detected.")
    # also print which band is the ~0.231 one each side
    cA = A["centers"]; cB = B["centers"]
    print(f"A centers: low={cA[0]:.6f}, high={cA[1]:.6f}")
    print(f"B centers: low={cB[0]:.6f}, high={cB[1]:.6f}")
    # interpret
    def near(x,t=0.231, eps=0.005): return abs(x-t)<eps
    interp = []
    for tag, c in (("A.low",cA[0]),("A.high",cA[1]),("B.low",cB[0]),("B.high",cB[1])):
        if near(c): interp.append(f"{tag}≈0.231")
        elif near(c, t=0.249, eps=0.005): interp.append(f"{tag}≈0.249")
    if interp:
        print("Centers near targets: " + ", ".join(interp))

def main():
    ap = argparse.ArgumentParser(description="Diagnose cycle see-saw from cycle-shift CSVs.")
    ap.add_argument("--csvA", required=True, help="CSV with dx,dy,sin2_mean,condG")
    ap.add_argument("--csvB", help="Optional second CSV for swap comparison")
    ap.add_argument("--labelA", default="Run A")
    ap.add_argument("--labelB", default="Run B")
    args = ap.parse_args()

    A = run_one(args.labelA, load_cycles_csv(args.csvA))
    if args.csvB:
        B = run_one(args.labelB, load_cycles_csv(args.csvB))
        swap_report(A, B)

if __name__ == "__main__":
    main()