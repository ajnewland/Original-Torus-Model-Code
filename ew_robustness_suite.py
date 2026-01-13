# ew_robustness_suite.py
# Robustness suite for sin^2(theta_W) derived from the invariant
#   G = (C A^{-1} C^T)^{-1}, where A = ⋆1 (Hodge star on edges) and
#   C encodes two fundamental cycles on the torus.
#
# Sub-tests:
#   1) Ridge sweep       : A -> A + ρ I, ρ ∈ [1e-14, 1e-9]
#   2) Cycle swap        : shift the two fundamental loops (x and y)
#   3) Hodge star swap   : cotan  <-> circumcentric
#   4) Size scaling      : (L,L) with L ∈ {16, 20, 24, 28, 32}
#
# Output: CSVs + quick PNGs per sub-test in --out.
#
# Usage examples (Windows-friendly, no trailing backslashes):
#   python ew_robustness_suite.py --run ridge --L 20 --jitter 0.02 --n 64 --out out_robust
#   python ew_robustness_suite.py --run cycles --L 20 --jitter 0.02 --n 64 --out out_robust
#   python ew_robustness_suite.py --run stars  --L 20 --jitter 0.02 --n 64 --out out_robust
#   python ew_robustness_suite.py --run size   --jitter 0.02 --n 64 --out out_robust
#   python ew_robustness_suite.py --run all    --jitter 0.02 --n 64 --out out_robust
#
# All math/IO in one file. No external deps beyond NumPy/SciPy/Matplotlib.

import argparse, math
from pathlib import Path
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ---------------- I/O helpers ----------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def write_csv(path, header, rows_2d_float):
    arr = np.array(rows_2d_float, float)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(Path(path), arr, delimiter=",", fmt="%.10g",
               header=",".join(header), comments="", encoding="utf-8")

def write_text(path, text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")

# ---------------- Mesh: periodic rectangular triangulation ----------------
def build_periodic_grid(Lx, Ly, jitter, ax=1.0, ay=1.0, seed=0):
    """
    Returns:
      P: (V,2) vertex coords in [0,ax)×[0,ay)
      E: (E,2) undirected edges (min,max vertex ids)
      F: (F,3) faces
      d0: (E,V) incidence
      face_edges, face_signs: for each face, edge id and orientation sign
      v_ix, v_iy: per-vertex integer grid coordinates (0..Lx-1, 0..Ly-1)
    """
    rng = np.random.default_rng(seed)
    def vid(ix, iy): return (iy % Ly) * Lx + (ix % Lx)

    ix = np.arange(Lx)
    iy = np.arange(Ly)
    XX, YY = np.meshgrid(ix, iy, indexing="xy")
    v_ix = XX.ravel()
    v_iy = YY.ravel()

    X = (v_ix.astype(float) / Lx) * ax
    Y = (v_iy.astype(float) / Ly) * ay
    P = np.column_stack([X, Y])
    if jitter > 0.0:
        dx = (ax / Lx) * jitter
        dy = (ay / Ly) * jitter
        P += rng.uniform(-1, 1, size=P.shape) * np.array([dx, dy])

    faces = []
    for jy in range(Ly):
        for jx in range(Lx):
            a = vid(jx, jy)
            b = vid(jx+1, jy)
            c = vid(jx, jy+1)
            d = vid(jx+1, jy+1)
            faces.append([a, b, d])
            faces.append([a, d, c])
    F = np.array(faces, dtype=int)

    def undirected_key(u, v): return (u, v) if u < v else (v, u)

    undirected = {}
    E_pairs = []
    face_edges = np.zeros((F.shape[0], 3), dtype=int)
    face_signs = np.zeros((F.shape[0], 3), dtype=int)

    def edge_index_oriented(u, v):
        key = undirected_key(u, v)
        ei = undirected.get(key, None)
        if ei is None:
            ei = len(E_pairs)
            undirected[key] = ei
            E_pairs.append(key)
        a, b = key
        sgn = +1 if (u == a and v == b) else -1
        return ei, sgn

    for fi, (a, b, c) in enumerate(F):
        for k, (u, v) in enumerate([(a, b), (b, c), (c, a)]):
            ei, sgn = edge_index_oriented(u, v)
            face_edges[fi, k] = ei
            face_signs[fi, k] = sgn

    E = np.array(E_pairs, dtype=int)
    V = P.shape[0]; Ecount = E.shape[0]; Fcount = F.shape[0]

    # d0 (E×V)
    rows, cols, data = [], [], []
    for ei, (i, j) in enumerate(E):
        rows += [ei, ei]; cols += [i, j]; data += [-1.0, +1.0]
    d0 = sp.csr_matrix((data, (rows, cols)), shape=(Ecount, V))

    return P, E, F, d0, face_edges, face_signs, v_ix, v_iy

# ---------------- Hodge stars on 1-forms ----------------
def cot_angle(A, B, C):
    v1 = B - A; v2 = C - A
    dot = float(np.dot(v1, v2))
    nrm = math.sqrt(max(np.dot(v1, v1) * np.dot(v2, v2), 1e-30))
    c = max(min(dot / nrm, 1.0), -1.0)
    s = math.sqrt(max(1.0 - c * c, 0.0))
    return 0.0 if s < 1e-14 else c / s

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
        cot_opp[fi, 0] = cotC  # opp edge (A,B)
        cot_opp[fi, 1] = cotA  # opp (B,C)
        cot_opp[fi, 2] = cotB  # opp (C,A)

    vals = np.zeros(Ecount)
    for ei, (i, j) in enumerate(E):
        le = npl.norm(P[j] - P[i])
        csum = 0.0
        for fi in edge_faces[ei]:
            a, b, c = F[fi]
            for k, (u, v) in enumerate([(a, b), (b, c), (c, a)]):
                if (u == i and v == j) or (u == j and v == i):
                    csum += cot_opp[fi, k]
        vals[ei] = 0.5 * max(csum, 0.0) * max(le, 1e-12)
    vals = np.maximum(vals, 1e-12)
    return sp.diags(vals, 0, format="csc")

def triangle_circumcenter(A, B, C):
    # robust 2D circumcenter
    a = B - A; b = C - A
    aa = np.dot(a, a); bb = np.dot(b, b)
    cross = a[0]*b[1] - a[1]*b[0]
    denom = 2.0 * cross
    if abs(denom) < 1e-16:
        return (A + B + C) / 3.0
    ux = A[0] + (b[1]*aa - a[1]*bb) / denom
    uy = A[1] + (a[0]*bb - b[0]*aa) / denom
    return np.array([ux, uy], float)

def star1_circumcentric(P, E, F, face_edges):
    # dual length as sum of distances between the two circumcenters sharing the edge
    Ecount = E.shape[0]; Fcount = F.shape[0]
    # build face->circumcenter
    CC = np.zeros((Fcount, 2))
    for fi, (i, j, k) in enumerate(F):
        CC[fi, :] = triangle_circumcenter(P[i], P[j], P[k])

    # collect adjacent faces per edge, then dual length = |CC1-CC2|
    adj = [[] for _ in range(Ecount)]
    for fi in range(Fcount):
        for k in range(3):
            ei = face_edges[fi, k]
            adj[ei].append(fi)

    vals = np.zeros(Ecount)
    for ei, faces in enumerate(adj):
        if len(faces) == 2:
            f1, f2 = faces
            dual_len = npl.norm(CC[f1] - CC[f2])
        elif len(faces) == 1:
            # on a torus we shouldn't hit boundary; guard anyway
            dual_len = 1e-12
        else:
            dual_len = 1e-12
        primal_len = npl.norm(P[E[ei,1]] - P[E[ei,0]])
        vals[ei] = max(dual_len, 1e-12) / max(primal_len, 1e-12)
    vals = np.maximum(vals, 1e-12)
    return sp.diags(vals, 0, format="csc")

# ---------------- Exact fundamental cycles (with shifts) ----------------
def build_exact_cycles_shift(Lx, Ly, E, v_ix, v_iy, shift_x=0, shift_y=0):
    """
    Two 1-cycles (rows of C ∈ R^{2×E}):
      - row 0: horizontal loop along iy = shift_y
      - row 1: vertical   loop along ix = shift_x
    Always rank-2, independent of geometry/jitter.
    """
    def vid(ix, iy): return (iy % Ly) * Lx + (ix % Lx)
    # map undirected edge to index and oriented lookup
    undirected = {(min(i, j), max(i, j)): ei for ei, (i, j) in enumerate(E)}

    def eidx_oriented(i, j):
        a, b = (i, j) if i < j else (j, i)
        ei = undirected.get((a, b), None)
        if ei is None: return None, 0
        sgn = +1 if (i < j) else -1
        return ei, sgn

    C = np.zeros((2, E.shape[0]), float)
    y0 = shift_y % Ly
    for x in range(Lx):
        u = vid(x, y0); v = vid(x + 1, y0)
        ei, sgn = eidx_oriented(u, v)
        if ei is not None: C[0, ei] += sgn

    x0 = shift_x % Lx
    for y in range(Ly):
        u = vid(x0, y); v = vid(x0, y + 1)
        ei, sgn = eidx_oriented(u, v)
        if ei is not None: C[1, ei] += sgn

    if np.linalg.matrix_rank(C) < 2:
        raise RuntimeError("Cycle matrix C lost rank (<2).")
    return sp.csr_matrix(C)

# ---------------- Invariant and sin^2(theta_W) ----------------
def sin2_from_G(star1, C, ridge=1e-12, cond_cap=1e12):
    """
    Compute G^{-1} = C A^{-1} C^T, then G = inv(G^{-1}),
    sin^2 = λ_min(G) / (λ_min(G)+λ_max(G)).
    Returns (sin2, condG, G) or (nan, nan, None) on failure.
    """
    try:
        A = star1.tocsc()
        I = sp.eye(A.shape[0], format="csc")
        X = spla.spsolve(A + ridge * I, C.T)  # (E×2)
        Ginv = C @ X
        Ginv = 0.5 * (Ginv + Ginv.T)
        # 2×2 inverse and condition
        det = Ginv[0,0]*Ginv[1,1] - Ginv[0,1]*Ginv[1,0]
        if not np.isfinite(det) or abs(det) < 1e-18:
            return np.nan, np.nan, None
        t = Ginv[0,0] + Ginv[1,1]
        d = (Ginv[0,0]-Ginv[1,1])**2 + 4*Ginv[0,1]*Ginv[1,0]
        d = max(d, 0.0)
        s = math.sqrt(d)
        lam_min_inv = 0.5*(t - s)
        lam_max_inv = 0.5*(t + s)
        if lam_min_inv <= 0 or lam_max_inv <= 0:
            return np.nan, np.nan, None
        cond = lam_max_inv / lam_min_inv
        if not np.isfinite(cond) or cond > cond_cap:
            return np.nan, np.nan, None

        G = (1.0/det) * np.array([[ Ginv[1,1], -Ginv[0,1]],
                                  [-Ginv[1,0],  Ginv[0,0]]], float)
        trG = G[0,0] + G[1,1]
        dG  = (G[0,0]-G[1,1])**2 + 4*G[0,1]*G[1,0]
        dG  = max(dG, 0.0)
        sG  = math.sqrt(dG)
        lam_min = 0.5*(trG - sG)
        lam_max = 0.5*(trG + sG)
        if lam_min <= 0 or lam_max <= 0:
            return np.nan, np.nan, None
        sin2 = lam_min / (lam_min + lam_max)
        return float(sin2), float(cond), G
    except Exception:
        return np.nan, np.nan, None

def one_setting_mean_sin2(L, jitter, n, ax, ay, star_kind="cotan",
                          ridge=1e-12, shift_x=0, shift_y=0, seed0=2025):
    vals = []; conds = []; ok = 0
    for k in range(n):
        seed = seed0 + 97*k
        P,E,F,d0,fe,fs,vix,viy = build_periodic_grid(L, L, jitter, ax=ax, ay=ay, seed=seed)
        if star_kind == "cotan":
            A = star1_cotan(P, E, F, fe)
        else:
            A = star1_circumcentric(P, E, F, fe)
        C = build_exact_cycles_shift(L, L, E, vix, viy, shift_x=shift_x, shift_y=shift_y)
        s2, cond, _ = sin2_from_G(A, C, ridge=ridge)
        if np.isfinite(s2):
            vals.append(s2); conds.append(cond); ok += 1
    if ok == 0:
        return np.nan, np.nan, ok
    return float(np.mean(vals)), float(np.std(vals)), ok

# ---------------- Sub-tests ----------------
def test_ridge_sweep(L, jitter, n, ax, ay, outdir):
    outdir = Path(outdir) / "ridge_sweep"; ensure_dir(outdir)
    ridges = np.logspace(-14, -9, 10)
    rows = []
    for r in ridges:
        mu, sd, ok = one_setting_mean_sin2(L, jitter, n, ax, ay, "cotan", ridge=r)
        print(f"ridge={r:.1e}  sin2={mu:.6f} ± {sd:.6f}  ok={ok}/{n}")
        rows.append([r, mu, sd, ok])
    write_csv(outdir/"ridge_sweep.csv", ["ridge","sin2_mean","sin2_std","ok"], rows)

    # plot
    xs = [row[0] for row in rows]
    ys = [row[1] for row in rows]
    es = [row[2] for row in rows]
    plt.figure(figsize=(6,4), dpi=140)
    plt.errorbar(xs, ys, yerr=es, fmt="o-", capsize=3)
    plt.axhline(0.231, ls="--", lw=1.0)
    plt.xscale("log")
    plt.xlabel("ridge ρ"); plt.ylabel("sin^2(theta_W)")
    plt.tight_layout(); plt.savefig(outdir/"ridge_sweep.png"); plt.close()

def test_cycle_swap(L, jitter, n, ax, ay, outdir):
    outdir = Path(outdir) / "cycle_swap"; ensure_dir(outdir)
    shifts = [(0,0), (1,0), (0,1), (3,2), (L//4, L//4)]
    rows = []
    for sx, sy in shifts:
        mu, sd, ok = one_setting_mean_sin2(L, jitter, n, ax, ay, "cotan", ridge=1e-11,
                                           shift_x=sx, shift_y=sy)
        print(f"shift=(x={sx},y={sy})  sin2={mu:.6f} ± {sd:.6f}  ok={ok}/{n}")
        rows.append([sx, sy, mu, sd, ok])
    write_csv(outdir/"cycle_swap.csv", ["shift_x","shift_y","sin2_mean","sin2_std","ok"], rows)

def test_star_swap(L, jitter, n, ax, ay, outdir):
    outdir = Path(outdir) / "star_swap"; ensure_dir(outdir)
    rows = []
    for kind in ["cotan", "circum"]:
        mu, sd, ok = one_setting_mean_sin2(L, jitter, n, ax, ay, kind, ridge=1e-11)
        print(f"star={kind:8s}  sin2={mu:.6f} ± {sd:.6f}  ok={ok}/{n}")
        rows.append([kind, mu, sd, ok])
    write_csv(outdir/"star_swap.csv", ["star","sin2_mean","sin2_std","ok"], rows)

def test_size_scaling(jitter, n, ax, ay, outdir):
    outdir = Path(outdir) / "size_scaling"; ensure_dir(outdir)
    sizes = [16, 20, 24, 28, 32]
    rows = []
    for L in sizes:
        mu, sd, ok = one_setting_mean_sin2(L, jitter, n, ax, ay, "cotan", ridge=1e-11)
        print(f"L={L:2d}  sin2={mu:.6f} ± {sd:.6f}  ok={ok}/{n}")
        rows.append([L, mu, sd, ok])
    write_csv(outdir/"size_scaling.csv", ["L","sin2_mean","sin2_std","ok"], rows)

    # plot mean ± std vs L, plus 1/sqrt(N_e) trend hint
    xs = [row[0] for row in rows]
    ys = [row[1] for row in rows]
    es = [row[2] for row in rows]
    plt.figure(figsize=(6,4), dpi=140)
    plt.errorbar(xs, ys, yerr=es, fmt="o-", capsize=3)
    plt.axhline(0.231, ls="--", lw=1.0)
    plt.xlabel("L (grid size)"); plt.ylabel("sin^2(theta_W)")
    plt.tight_layout(); plt.savefig(outdir/"size_scaling.png"); plt.close()

# ---------------- Driver ----------------
def main():
    ap = argparse.ArgumentParser(description="Robustness suite for sin^2(theta_W) from G invariant")
    ap.add_argument("--run", type=str, default="all",
                    choices=["ridge","cycles","stars","size","all"],
                    help="which sub-test(s) to run")
    ap.add_argument("--L", type=int, default=20, help="grid length (for single-size tests)")
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=64, help="meshes per setting")
    ap.add_argument("--ax", type=float, default=2.57)
    ap.add_argument("--ay", type=float, default=0.78)
    ap.add_argument("--out", type=str, default="out_robust")
    args = ap.parse_args()

    print("=== EW robustness suite ===")
    print(f"run={args.run}  L={args.L}  jitter={args.jitter}  n={args.n}")
    print(f"(ax,ay)=({args.ax},{args.ay})  out={args.out}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.run in ("ridge","all"):
        print("\n[Ridge sweep]")
        test_ridge_sweep(args.L, args.jitter, args.n, args.ax, args.ay, out)

    if args.run in ("cycles","all"):
        print("\n[Cycle swap]")
        test_cycle_swap(args.L, args.jitter, args.n, args.ax, args.ay, out)

    if args.run in ("stars","all"):
        print("\n[Hodge star swap]")
        test_star_swap(args.L, args.jitter, args.n, args.ax, args.ay, out)

    if args.run in ("size","all"):
        print("\n[Size scaling]")
        test_size_scaling(args.jitter, args.n, args.ax, args.ay, out)

if __name__ == "__main__":
    main()