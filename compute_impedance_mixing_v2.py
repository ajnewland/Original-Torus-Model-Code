# compute_impedance_mixing_v2.py
import argparse, os, sys
import numpy as np
import pandas as pd

def load_grid(path):
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    colmap = {c.lower(): c for c in df.columns}
    if 'ax' not in cols or 'ay' not in cols:
        raise ValueError(f"CSV must contain 'ax' and 'ay'. Found: {list(df.columns)}")
    zcol = 'z' if 'z' in cols else ('z_pred' if 'z_pred' in cols else None)
    if zcol is None:
        raise ValueError(f"CSV must contain 'z' or 'z_pred'. Found: {list(df.columns)}")

    axc, ayc, zc = colmap['ax'], colmap['ay'], colmap[zcol]
    df = df[[axc, ayc, zc]].copy()
    df = df.rename(columns={axc:'ax', ayc:'ay', zc:'z'})
    # pivot to a regular grid
    ax_vals = np.sort(df['ax'].unique())
    ay_vals = np.sort(df['ay'].unique())
    grid = np.full((len(ay_vals), len(ax_vals)), np.nan, dtype=float)
    ax_index = {v:i for i,v in enumerate(ax_vals)}
    ay_index = {v:i for i,v in enumerate(ay_vals)}
    for _, row in df.iterrows():
        i = ay_index[row['ay']]
        j = ax_index[row['ax']]
        grid[i, j] = row['z']
    if np.isnan(grid).any():
        raise ValueError("Grid has NaNs after pivot (incomplete regular grid). "
                         "Make sure the refined sweep CSV is a full rectilinear grid.")
    return ax_vals, ay_vals, grid

def nearest_index(vals, x):
    # returns closest index in sorted 1D array 'vals'
    i = np.searchsorted(vals, x)
    if i == 0: return 0
    if i >= len(vals): return len(vals)-1
    # choose nearer of i-1 and i
    if abs(vals[i]-x) < abs(vals[i-1]-x):
        return i
    return i-1

def centered_diff_1d(arr, h, idx, stencil):
    """
    centered first derivative at index idx using odd stencil (3,5,7,...)
    on a uniformly-spaced axis with spacing h. Returns derivative and ok-flag.
    """
    s = stencil
    r = (s-1)//2
    n = len(arr)
    if idx - r < 0 or idx + r >= n:
        return np.nan, False
    # build symmetric weights for first derivative (finite difference)
    # using simple central difference of order s:
    # we can use the standard coefficients from polynomial fits.
    # For robustness, derive coefficients via Vandermonde least squares locally:
    xs = np.arange(-r, r+1, dtype=float) * h
    V = np.vstack([xs**k for k in range(s)]).T  # shape (s, s)
    # derivative target = d/dx at 0 -> vector d = [0!,1!,2!,...] with 1 at k=1 and 0 else (scaled)
    d = np.zeros(s); d[1] = 1.0
    # Solve least-squares weights w so that sum w_i * f(x_i) ≈ f'(0)
    # (We are fitting a polynomial and taking its derivative at 0)
    w, *_ = np.linalg.lstsq(V, d, rcond=None)
    window = arr[idx-r:idx+r+1]
    return float(np.dot(w, window)), True

def centered_diff_2d(Z, ax_vals, ay_vals, i0, j0, stencil):
    """
    First and second partials at (i0,j0) using separable 1D stencils.
    Returns Zx, Zy, Zxx, Zxy, Zyy and ok-flag.
    """
    s = stencil
    r = (s-1)//2
    ny, nx = Z.shape
    if i0 - r < 0 or i0 + r >= ny or j0 - r < 0 or j0 + r >= nx:
        return (np.nan,)*5, False

    dax = float(ax_vals[1] - ax_vals[0])
    day = float(ay_vals[1] - ay_vals[0])

    # 1D derivative weights (first & second) via least squares as above
    xs = np.arange(-r, r+1, dtype=float)
    V = np.vstack([xs**k for k in range(s)]).T
    d1 = np.zeros(s); d1[1] = 1.0       # for first derivative
    d2 = np.zeros(s); d2[2] = 2.0       # for second derivative (since d^2/dx^2 x^2 = 2)
    w1, *_ = np.linalg.lstsq(V, d1, rcond=None)
    w2, *_ = np.linalg.lstsq(V, d2, rcond=None)

    # Zx: apply along x at fixed y
    row = Z[i0, j0-r:j0+r+1]
    Zx = float(np.dot(w1, row)) / dax

    # Zy: apply along y at fixed x
    col = Z[i0-r:i0+r+1, j0]
    Zy = float(np.dot(w1, col)) / day

    # Zxx: second derivative along x
    Zxx = float(np.dot(w2, row)) / (dax**2)

    # Zyy: second derivative along y
    Zyy = float(np.dot(w2, col)) / (day**2)

    # Zxy: first in x then in y (or vice versa); use separable stencil
    # Finite-difference: compute Zx along x for each y in window, then diff in y at center
    Zx_strip = np.zeros(s, dtype=float)
    for u in range(-r, r+1):
        row_u = Z[i0+u, j0-r:j0+r+1]
        Zx_strip[u+r] = float(np.dot(w1, row_u)) / dax
    Zxy = float(np.dot(w1, Zx_strip)) / day

    return Zx, Zy, Zxx, Zxy, Zyy, True

def principal_curvatures(Zx, Zy, Zxx, Zxy, Zyy):
    """
    Shape-operator eigenvalues (proportional to principal curvatures) in Monge patch.
    Up to a common factor, use the Hessian entries.
    Return absolute eigenvalues (magnitudes).
    """
    H = np.array([[Zxx, Zxy],
                  [Zxy, Zyy]], dtype=float)
    w = np.linalg.eigvals(H)
    k1, k2 = np.sort(np.abs(w))  # ascending
    return float(k1), float(k2)

def local_impedance(ax_vals, ay_vals, Z, Ly_over_Lx, ax0, ay0, stencil=5, use_grad_metric=False):
    j0 = nearest_index(ax_vals, ax0)
    i0 = nearest_index(ay_vals, ay0)
    Zx, Zy, Zxx, Zxy, Zyy, ok = centered_diff_2d(Z, ax_vals, ay_vals, i0, j0, stencil)
    if not ok:
        return {
            "ok": False, "reason": "center too close to boundary for chosen stencil",
            "ax0": ax0, "ay0": ay0
        }
    k1, k2 = principal_curvatures(Zx, Zy, Zxx, Zxy, Zyy)

    # direction weights (impedances)
    if use_grad_metric:
        gx = np.sqrt(1.0 + Zx*Zx)
        gy = np.sqrt(1.0 + Zy*Zy)
    else:
        # global cycle ratio proxy for metric anisotropy
        # normalize so that gy/gx = Ly/Lx
        gx = 1.0
        gy = float(Ly_over_Lx)

    Ix = gx * k1
    Iy = gy * k2
    s_here = float(Iy / (Ix + Iy)) if (Ix + Iy) != 0 else np.nan

    # Neighborhood stats
    r = (stencil-1)//2
    s_vals = []
    for di in range(-r, r+1):
        for dj in range(-r, r+1):
            ii = i0 + di
            jj = j0 + dj
            Zx_, Zy_, Zxx_, Zxy_, Zyy_, ok_ = centered_diff_2d(Z, ax_vals, ay_vals, ii, jj, stencil)
            if not ok_:
                continue
            k1_, k2_ = principal_curvatures(Zx_, Zy_, Zxx_, Zxy_, Zyy_)
            if use_grad_metric:
                gx_ = np.sqrt(1.0 + Zx_*Zx_)
                gy_ = np.sqrt(1.0 + Zy_*Zy_)
            else:
                gx_, gy_ = 1.0, float(Ly_over_Lx)
            Ix_ = gx_ * k1_
            Iy_ = gy_ * k2_
            s_loc = float(Iy_ / (Ix_ + Iy_)) if (Ix_ + Iy_) != 0 else np.nan
            if np.isfinite(s_loc):
                s_vals.append(s_loc)

    s_vals = np.array(s_vals, dtype=float)
    if s_vals.size == 0:
        s_med, s_mad = np.nan, np.nan
    else:
        s_med = float(np.median(s_vals))
        s_mad = float(np.median(np.abs(s_vals - s_med)))

    out = {
        "ok": True,
        "ax0": float(ax0), "ay0": float(ay0),
        "Zx": float(Zx), "Zy": float(Zy),
        "Zxx": float(Zxx), "Zxy": float(Zxy), "Zyy": float(Zyy),
        "k1": float(k1), "k2": float(k2),
        "Ly_over_Lx": float(Ly_over_Lx),
        "s_star": s_here,
        "s_med": s_med,
        "s_MAD": s_mad,
        "stencil": int(stencil),
        "use_grad_metric": bool(use_grad_metric)
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="refined grid CSV (must be full ax×ay grid)")
    ap.add_argument("--Ly_over_Lx", type=float, default=1.0, help="global cycle-length ratio Ly/Lx")
    ap.add_argument("--ax0", type=float, required=True)
    ap.add_argument("--ay0", type=float, required=True)
    ap.add_argument("--stencil", type=int, default=5, help="odd stencil size: 3,5,7,...")
    ap.add_argument("--use_grad_metric", action="store_true",
                    help="use local sqrt(1+Zx^2), sqrt(1+Zy^2) instead of global Ly/Lx")
    args = ap.parse_args()

    ax_vals, ay_vals, Z = load_grid(args.grid)
    out = local_impedance(ax_vals, ay_vals, Z, args.Ly_over_Lx, args.ax0, args.ay0,
                          stencil=args.stencil, use_grad_metric=args.use_grad_metric)

    if not out["ok"]:
        print(f"[ERROR] {out['reason']}")
        sys.exit(2)

    print("=== Impedance mixing (v2) at chosen point ===")
    print(f"ax0,ay0  = {out['ax0']:.6f}, {out['ay0']:.6f}")
    print(f"Zx,Zy    = {out['Zx']:+.6e}, {out['Zy']:+.6e}")
    print(f"Zxx,Zxy,Zyy = {out['Zxx']:+.6e}, {out['Zxy']:+.6e}, {out['Zyy']:+.6e}")
    print(f"|k1|,|k2| = {out['k1']:.6e}, {out['k2']:.6e}")
    print(f"Ly/Lx     = {out['Ly_over_Lx']}")
    print(f"s_star    = {out['s_star']:.6f}")
    print(f"s_med     = {out['s_med']:.6f}")
    print(f"s_MAD     = {out['s_MAD']:.6e}")
    print(f"stencil   = {out['stencil']} (use_grad_metric={out['use_grad_metric']})")

if __name__ == "__main__":
    main()