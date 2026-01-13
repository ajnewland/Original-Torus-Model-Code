# ew_autosweep_invariant.py  (defensive version)
# Robust auto-sweep in (ax, ay) using the basis-invariant sin^2θ_W from G = C (A+εI)^(-1) C^T
# Fixes:
#  - Guarantee 2D shapes for X and G
#  - Defensive reshape of G to 2x2
#  - Validate/repair C rows; skip setting if still degenerate
#  - Clear diagnostics in console + CSV

import argparse, math
from pathlib import Path
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ---------- IO helpers ----------
def write_text(path, text):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

def savetxt_utf8(path, arr2d, header):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(p, arr2d, delimiter=",", fmt="%.10g",
               header=header, comments="", encoding="utf-8")

# ---------- Mesh (periodic rectangular triangulation) ----------
def build_periodic_grid(Lx, Ly, jitter, ax=1.0, ay=1.0, seed=0):
    rng = np.random.default_rng(seed)
    def vid(ix, iy): return (iy % Ly) * Lx + (ix % Lx)

    X = np.linspace(0.0, ax, Lx, endpoint=False)
    Y = np.linspace(0.0, ay, Ly, endpoint=False)
    xx, yy = np.meshgrid(X, Y, indexing="xy")
    P = np.column_stack([xx.ravel(), yy.ravel()])

    if jitter > 0:
        dx = (ax / Lx) * jitter
        dy = (ay / Ly) * jitter
        P += rng.uniform(-1.0, 1.0, size=P.shape) * np.array([dx, dy])

    faces = []
    for iy in range(Ly):
        for ix in range(Lx):
            i  = vid(ix,   iy  )
            j  = vid(ix+1, iy  )
            k  = vid(ix,   iy+1)
            l  = vid(ix+1, iy+1)
            faces.append([i, j, l])
            faces.append([i, l, k])
    F = np.array(faces, dtype=int)

    undirected = {}
    E_pairs = []
    face_edges = np.zeros((F.shape[0], 3), dtype=int)
    face_signs = np.zeros((F.shape[0], 3), dtype=int)

    def edge_key(a, b):
        return ((a, b), +1) if a < b else ((b, a), -1)

    for fi, (a, b, c) in enumerate(F):
        for k, (u, v) in enumerate([(a, b), (b, c), (c, a)]):
            key, sgn = edge_key(u, v)
            if key not in undirected:
                undirected[key] = len(E_pairs)
                E_pairs.append(key)
            ei = undirected[key]
            face_edges[fi, k] = ei
            face_signs[fi, k] = sgn

    E = np.array(E_pairs, dtype=int)
    V = P.shape[0]; Ecount = E.shape[0]; Fcount = F.shape[0]

    # d0 (E x V)
    rows, cols, data = [], [], []
    for ei, (i, j) in enumerate(E):
        rows += [ei, ei]; cols += [i, j]; data += [-1.0, +1.0]
    d0 = sp.csr_matrix((data, (rows, cols)), shape=(Ecount, V))

    # d1 (F x E)
    rows, cols, data = [], [], []
    for fi in range(Fcount):
        for k in range(3):
            rows.append(fi); cols.append(face_edges[fi, k]); data.append(float(face_signs[fi, k]))
    d1 = sp.csr_matrix((data, (rows, cols)), shape=(Fcount, Ecount))

    return P, E, F, d0, d1, face_edges, face_signs

# ---------- DEC star1 (cotan) ----------
def cot_angle(A, B, C):
    v1, v2 = B - A, C - A
    dot = float(np.dot(v1, v2))
    nrm = math.sqrt(max(np.dot(v1, v1)*np.dot(v2, v2), 1e-30))
    c = max(min(dot/nrm, 1.0), -1.0)
    s = math.sqrt(max(1.0 - c*c, 0.0))
    return 0.0 if s < 1e-14 else c/s

def star1_cotan(P, E, F, face_edges):
    Ecount = E.shape[0]; Fcount = F.shape[0]
    edge_faces = [[] for _ in range(Ecount)]
    for fi in range(Fcount):
        for k in range(3):
            edge_faces[face_edges[fi, k]].append(fi)

    face_pts = P[F]  # (F,3,2)
    cot_opp = np.zeros((Fcount, 3))
    for fi in range(Fcount):
        A, B, C = face_pts[fi]
        cotA = cot_angle(A, B, C)
        cotB = cot_angle(B, C, A)
        cotC = cot_angle(C, A, B)
        cot_opp[fi, 0] = cotC
        cot_opp[fi, 1] = cotA
        cot_opp[fi, 2] = cotB

    star_vals = np.zeros(Ecount)
    for ei, (i, j) in enumerate(E):
        le = npl.norm(P[j] - P[i])
        csum = 0.0
        for fi in edge_faces[ei]:
            a, b, c = F[fi]
            for k, (u, v) in enumerate([(a, b), (b, c), (c, a)]):
                if (u == i and v == j) or (u == j and v == i):
                    csum += cot_opp[fi, k]
        star_vals[ei] = max(0.5 * csum * le, 1e-12)
    return sp.diags(star_vals, 0, format="csc")

# ---------- Period matrix C (2 x E) ----------
def period_matrix(P, E, loosen=1.0):
    """Two indicator rows for x- and y-wrapping edges. 'loosen'>1 relaxes the wrap threshold slightly."""
    xs, ys = P[:, 0], P[:, 1]
    hx = 0.5 * (xs.max() - xs.min() + 1e-12) / loosen
    hy = 0.5 * (ys.max() - ys.min() + 1e-12) / loosen
    rows, cols, data = [], [], []
    for ei, (i, j) in enumerate(E):
        if abs(xs[j] - xs[i]) > hx:
            rows.append(0); cols.append(ei); data.append(1.0)
    for ei, (i, j) in enumerate(E):
        if abs(ys[j] - ys[i]) > hy:
            rows.append(1); cols.append(ei); data.append(1.0)
    return sp.csr_matrix((data, (rows, cols)), shape=(2, E.shape[0]))

def ensure_valid_C(C, P, E):
    """Make sure both rows of C have at least one nonzero; retry with looser test if needed."""
    ok = True
    for r in range(2):
        if C[r].nnz == 0:
            ok = False
    if ok: return C, True
    # retry with looser wrap detection
    C2 = period_matrix(P, E, loosen=1.2)
    ok2 = (C2[0].nnz > 0) and (C2[1].nnz > 0)
    return (C2 if ok2 else C2), ok2

# ---------- Invariant sin^2 from G ----------
def sin2_from_G(star1, C, ridge=1e-10):
    """sin^2 = g_min / (g_min + g_max) from G = C (A+ridge I)^(-1) C^T, with robust shaping."""
    A = star1 + ridge * sp.eye(star1.shape[0], format="csc")

    # Solve A X = C^T (2 RHS). Force dense 2D result (E,2).
    CT = C.T.tocsc()
    X = spla.spsolve(A, CT)              # may return (E,2) ndarray
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape((-1, 1))
    if X.shape[1] == 1:  # if somehow one column snuck in, pad zeros to keep (E,2)
        X = np.column_stack([X, np.zeros_like(X)])

    # G = C @ X  -> ensure 2x2 ndarray
    G = C @ X
    if sp.issparse(G):
        G = G.toarray()
    else:
        G = np.asarray(G)
    G = G.reshape(2, 2)  # defensive

    # Symmetrize & eigenvalues
    Gs = 0.5 * (G + G.T)
    w = npl.eigvalsh(Gs)
    w = np.maximum(w, 1e-18)
    gmin, gmax = float(w[0]), float(w[1])
    sin2 = gmin / (gmin + gmax)
    condG = gmax / gmin
    return sin2, condG, Gs

# ---------- One setting ----------
def one_setting_eval(Lx, Ly, jitter, ax, ay, seed, ridge=1e-10):
    P, E, F, d0, d1, face_edges, face_signs = build_periodic_grid(Lx, Ly, jitter, ax=ax, ay=ay, seed=seed)
    star1 = star1_cotan(P, E, F, face_edges)
    C = period_matrix(P, E)
    C, ok = ensure_valid_C(C, P, E)
    if not ok:
        # Degenerate periods — return NaNs with a clear flag
        return dict(ax=ax, ay=ay, seed=seed, sin2=np.nan, condG=np.nan, Gxx=np.nan, Gyy=np.nan, Gxy=np.nan, note="degenerate_C")

    try:
        sin2, condG, G = sin2_from_G(star1, C, ridge=ridge)
        return dict(ax=ax, ay=ay, seed=seed,
                    sin2=float(sin2), condG=float(condG),
                    Gxx=float(G[0, 0]), Gyy=float(G[1, 1]), Gxy=float(G[0, 1]),
                    note="")
    except Exception as e:
        # Any unexpected collapse: report and continue
        return dict(ax=ax, ay=ay, seed=seed,
                    sin2=np.nan, condG=np.nan, Gxx=np.nan, Gyy=np.nan, Gxy=np.nan,
                    note=f"fail:{type(e).__name__}")

def aggregate_stats(rows):
    # keep only finite rows
    vals = [r for r in rows if np.isfinite(r["sin2"])]
    if not vals:
        return dict(sin2_mean=np.nan, sin2_std=np.nan,
                    condG_mean=np.nan, condG_std=np.nan,
                    Gxx_mean=np.nan, Gyy_mean=np.nan, Gxy_mean=np.nan,
                    n_ok=0, n_total=len(rows), n_bad=len(rows))
    arr = {k: np.array([r[k] for r in vals], float) for k in ["sin2", "condG", "Gxx", "Gyy", "Gxy"]}
    out = {}
    for k in arr:
        out[k+"_mean"] = float(np.mean(arr[k]))
        out[k+"_std"]  = float(np.std(arr[k]))
    out["n_ok"] = len(vals); out["n_total"] = len(rows); out["n_bad"] = len(rows) - len(vals)
    return out

# ---------- Sweep ----------
def autosweep(Lx, Ly, jitter, n, ax_min, ax_max, ax_steps,
              ay_min, ay_max, ay_steps, target, tol, ridge, outdir,
              seed0, save_per_setting=False, stop_on_hit=True, std_cap=0.02):

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    ax_vals = np.linspace(ax_min, ax_max, ax_steps)
    ay_vals = np.linspace(ay_min, ay_max, ay_steps)

    grid_records = []
    best = None; best_err = 1e9

    for ia, ax in enumerate(ax_vals):
        for jb, ay in enumerate(ay_vals):
            rows = []
            for k in range(n):
                seed = seed0 + 97*k + 7919*ia + 104729*jb
                rows.append(one_setting_eval(Lx, Ly, jitter, ax, ay, seed, ridge=ridge))
            agg = aggregate_stats(rows)

            rec = [ax, ay,
                   agg["sin2_mean"], agg["sin2_std"],
                   agg["condG_mean"], agg["condG_std"],
                   agg["Gxx_mean"], agg["Gyy_mean"], agg["Gxy_mean"],
                   agg.get("n_ok", 0), agg.get("n_total", n), agg.get("n_bad", 0)]
            grid_records.append(rec)

            if np.isfinite(agg["sin2_mean"]):
                err = abs(agg["sin2_mean"] - target)
                if err < best_err:
                    best_err = err
                    best = dict(ax=ax, ay=ay, **agg)

                print(f"(ax,ay)=({ax:.3f},{ay:.3f})  sin2={agg['sin2_mean']:.3f} ± {agg['sin2_std']:.3f}  "
                      f"condG={agg['condG_mean']:.2e}  ok={agg['n_ok']}/{agg['n_total']}")
                if stop_on_hit and (err <= tol) and (agg["sin2_std"] <= max(std_cap, 0.5*tol)) and agg["n_ok"] >= max(4, n//2):
                    msg = (f"HIT: (ax,ay)=({ax:.6f},{ay:.6f})  sin2_mean={agg['sin2_mean']:.6f}  "
                           f"std={agg['sin2_std']:.6f}  condG_mean={agg['condG_mean']:.6f}  "
                           f"ok={agg['n_ok']}/{agg['n_total']}")
                    print(msg)
                    write_text(outdir / "best.txt", msg + "\n")
                    _finalize(ax_vals, ay_vals, grid_records, outdir)
                    return
            else:
                print(f"(ax,ay)=({ax:.3f},{ay:.3f})  sin2=NaN  (degenerate periods or failure in some meshes)")

            # optional per-setting CSV
            if save_per_setting:
                hdr = ["ax","ay","seed","sin2","condG","Gxx","Gyy","Gxy","note"]
                arr = np.array([[r["ax"], r["ay"], r["seed"], r["sin2"], r["condG"], r["Gxx"], r["Gyy"], r["Gxy"]] for r in rows], float)
                # 'note' is not numeric; write a parallel txt
                savetxt_utf8(outdir / f"per_setting_ax{ax:.3f}_ay{ay:.3f}.csv", arr, header=",".join(hdr[:-1]))
                notes = "\n".join([r.get("note","") for r in rows])
                write_text(outdir / f"per_setting_ax{ax:.3f}_ay{ay:.3f}_notes.txt", notes)

    # done without a tolerance hit
    _finalize(ax_vals, ay_vals, grid_records, outdir)
    if best is not None:
        msg = (f"BEST: (ax,ay)=({best['ax']:.6f},{best['ay']:.6f})  "
               f"sin2_mean={best['sin2_mean']:.6f}  std={best['sin2_std']:.6f}  "
               f"condG_mean={best['condG_mean']:.6f}  ok={best['n_ok']}/{best['n_total']}")
        print(msg); write_text(outdir / "best.txt", msg + "\n")

def _finalize(ax_vals, ay_vals, grid_records, outdir):
    hdr = "ax,ay,sin2_mean,sin2_std,condG_mean,condG_std,Gxx_mean,Gyy_mean,Gxy_mean,n_ok,n_total,n_bad"
    savetxt_utf8(Path(outdir) / "grid.csv", np.array(grid_records, float), header=hdr)
    make_heatmap(ax_vals, ay_vals, grid_records, Path(outdir) / "heatmap.png")

# ---------- Plot ----------
def make_heatmap(ax_vals, ay_vals, grid_records, outpng):
    A = len(ax_vals); B = len(ay_vals)
    # Fill with NaN-safe; map by loop order (ax major, ay minor)
    M = np.full((A, B), np.nan)
    k = 0
    for ia in range(A):
        for jb in range(B):
            M[ia, jb] = grid_records[k][2]  # sin2_mean
            k += 1
    plt.figure(figsize=(6.0, 4.6), dpi=140)
    extent = [ay_vals[0], ay_vals[-1], ax_vals[0], ax_vals[-1]]
    im = plt.imshow(M, origin="lower", extent=extent, aspect="auto")
    cbar = plt.colorbar(im, label="mean sin^2(theta_W)")
    plt.xlabel("ay"); plt.ylabel("ax")
    plt.tight_layout()
    Path(outpng).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpng); plt.close()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Auto-sweep (ax, ay) to hit target sin^2(theta_W) from invariant G with robust shaping")
    ap.add_argument("--Lx", type=int, default=20)
    ap.add_argument("--Ly", type=int, default=20)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=16, help="meshes per (ax,ay)")
    ap.add_argument("--ax_min", type=float, default=1.30)
    ap.add_argument("--ax_max", type=float, default=1.55)
    ap.add_argument("--ax_steps", type=int, default=6)
    ap.add_argument("--ay_min", type=float, default=0.95)
    ap.add_argument("--ay_max", type=float, default=1.10)
    ap.add_argument("--ay_steps", type=int, default=4)
    ap.add_argument("--target", type=float, default=0.231)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--ridge", type=float, default=1e-10)
    ap.add_argument("--out", type=str, default="out_autosweep")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--save_per_setting", action="store_true")
    ap.add_argument("--no_early_stop", action="store_true")
    args = ap.parse_args()

    autosweep(Lx=args.Lx, Ly=args.Ly, jitter=args.jitter,
              n=args.n, ax_min=args.ax_min, ax_max=args.ax_max, ax_steps=args.ax_steps,
              ay_min=args.ay_min, ay_max=args.ay_max, ay_steps=args.ay_steps,
              target=args.target, tol=args.tol, ridge=args.ridge, outdir=args.out,
              seed0=args.seed, save_per_setting=args.save_per_setting,
              stop_on_hit=(not args.no_early_stop), std_cap=0.02)

if __name__ == "__main__":
    main()