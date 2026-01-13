# ew_lock_report.py
# Lock and report sin^2(theta_W) using the invariant from G = (C A^{-1} C^T)^{-1}
# - star1 = cotan on edges
# - C = exact fundamental x/y cycles (topological, rank-2)
# - sin2 = lambda_min(G) / (lambda_min(G)+lambda_max(G))
# Produces per-setting CSVs and histograms, plus an aggregates table.

import argparse, math
from pathlib import Path
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ---------------- I/O helpers ----------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def write_text(path, text):
    path = Path(path); ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")

def write_csv(path, header, rows_float2d):
    arr = np.array(rows_float2d, float)
    p = Path(path); ensure_dir(p.parent)
    np.savetxt(p, arr, delimiter=",", fmt="%.10g",
               header=",".join(header), comments="", encoding="utf-8")

# ---------------- Mesh: periodic rectangular triangulation ----------------
def build_periodic_grid(Lx, Ly, jitter, ax=1.0, ay=1.0, seed=0):
    """
    Returns:
      P: (V,2) vertex coords in [0,ax)×[0,ay)
      E: (E,2) undirected edges with increasing (min,max) vertex ids
      F: (F,3) faces
      d0: (E,V) incidence
      face_edges, face_signs: for each face, the edge id and orientation sign
      ix,iy: per-vertex integer grid coordinates (0..Lx-1, 0..Ly-1)
    """
    rng = np.random.default_rng(seed)
    def vid(ix, iy): return (iy % Ly) * Lx + (ix % Lx)

    # integer grid coords (for exact cycles)
    ix = np.arange(Lx)
    iy = np.arange(Ly)
    XX, YY = np.meshgrid(ix, iy, indexing="xy")
    v_ix = XX.ravel()
    v_iy = YY.ravel()

    # geometric coords
    X = (v_ix.astype(float) / Lx) * ax
    Y = (v_iy.astype(float) / Ly) * ay
    P = np.column_stack([X, Y])
    if jitter > 0.0:
        dx = (ax / Lx) * jitter
        dy = (ay / Ly) * jitter
        P += rng.uniform(-1, 1, size=P.shape) * np.array([dx, dy])

    # faces: two triangles per cell
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

    # undirected edge list with an index
    def undirected_key(u, v):
        return (u, v) if u < v else (v, u)

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
        # orientation sign wrt stored (min,max)
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

# ---------------- DEC: cotan star on 1-forms ----------------
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
        cot_opp[fi, 0] = cotC  # opp (A,B)
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

# ---------------- Exact fundamental cycles (topological) ----------------
def build_exact_cycles(Lx, Ly, E, v_ix, v_iy):
    """
    Build two cycle functionals C (2×E), each row selects edges along a single
    fundamental loop with ±1 orientation:
      - row 0: horizontal loop through row iy=0
      - row 1: vertical loop through column ix=0
    Guaranteed rank-2 and independent of geometry/jitter.
    """
    def vid(ix, iy): return (iy % Ly) * Lx + (ix % Lx)
    # map undirected edge to index and provide an oriented lookup
    undirected = {(min(i, j), max(i, j)): ei for ei, (i, j) in enumerate(E)}

    def edge_index_oriented(i, j):
        a, b = (i, j) if i < j else (j, i)
        ei = undirected.get((a, b), None)
        if ei is None:
            return None, 0
        sgn = +1 if (i < j) else -1
        return ei, sgn

    C = np.zeros((2, E.shape[0]), float)

    # horizontal loop (y=0), forward +x
    y0 = 0
    for x in range(Lx):
        u = vid(x, y0); v = vid(x + 1, y0)
        ei, sgn = edge_index_oriented(u, v)
        if ei is not None:
            C[0, ei] += sgn

    # vertical loop (x=0), forward +y
    x0 = 0
    for y in range(Ly):
        u = vid(x0, y); v = vid(x0, y + 1)
        ei, sgn = edge_index_oriented(u, v)
        if ei is not None:
            C[1, ei] += sgn

    if np.linalg.matrix_rank(C) < 2:
        raise RuntimeError("Cycle matrix C lost rank (<2).")
    return sp.csr_matrix(C)

# ---------------- Invariant and sin^2(theta_W) ----------------
def sin2_from_G(star1, C, ridge=1e-11, cond_cap=1e12):
    """
    Compute G^{-1} = C A^{-1} C^T, then G = inv(G^{-1}),
    then sin2 = λ_min(G)/(λ_min(G)+λ_max(G)).
    Returns (sin2, condG, G) or (np.nan, np.nan, None) on failure.
    """
    try:
        A = star1.tocsc()
        # solve A X = C^T for X (E×2)
        X = spla.spsolve(A + ridge * sp.eye(A.shape[0], format="csc"), C.T)
        Ginv = (C @ X).astype(float)     # 2×2
        Ginv = 0.5 * (Ginv + Ginv.T)     # symmetrize

        # 2×2 safe condition via eigenvalues of Ginv
        t = Ginv[0,0] + Ginv[1,1]
        d = (Ginv[0,0]-Ginv[1,1])**2 + 4*Ginv[0,1]*Ginv[1,0]
        if d < 0: d = 0.0
        s = math.sqrt(d)
        lam_min_inv = 0.5*(t - s)
        lam_max_inv = 0.5*(t + s)
        if lam_min_inv <= 0 or lam_max_inv <= 0:
            return np.nan, np.nan, None
        cond = lam_max_inv / lam_min_inv
        if not np.isfinite(cond) or cond > cond_cap:
            return np.nan, np.nan, None

        # invert 2×2
        det = Ginv[0,0]*Ginv[1,1] - Ginv[0,1]*Ginv[1,0]
        if not np.isfinite(det) or abs(det) < 1e-18:
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

# ---------------- One fixed setting (average over n seeds) ----------------
def one_setting_run(Lx, Ly, jitter, ax, ay, n, seed0, ridge=1e-11):
    sin2_list = []; cond_list = []; oks = 0
    for k in range(n):
        seed = seed0 + 97*k
        P,E,F,d0,fe,fs, vix, viy = build_periodic_grid(Lx, Ly, jitter, ax, ay, seed)
        star1 = star1_cotan(P, E, F, fe)
        C = build_exact_cycles(Lx, Ly, E, vix, viy)
        sin2, condG, G = sin2_from_G(star1, C, ridge=ridge)
        if np.isfinite(sin2):
            sin2_list.append(sin2); cond_list.append(condG); oks += 1

    res = dict(Lx=Lx, Ly=Ly, jitter=jitter, ax=ax, ay=ay,
               n=n, ok=oks,
               sin2_mean=float(np.mean(sin2_list)) if oks>0 else np.nan,
               sin2_std=float(np.std(sin2_list)) if oks>0 else np.nan,
               cond_mean=float(np.mean(cond_list)) if oks>0 else np.nan,
               cond_std=float(np.std(cond_list)) if oks>0 else np.nan)
    return res, np.array(sin2_list, float), np.array(cond_list, float)

# ---------------- Plotting ----------------
def plot_hist(vals, title, xlabel, outpng):
    if vals.size == 0: return
    plt.figure(figsize=(6,4), dpi=140)
    plt.hist(vals, bins=24, edgecolor="k")
    plt.axvline(np.mean(vals), ls="--", lw=1, label=f"mean={np.mean(vals):.3f}")
    plt.xlabel(xlabel); plt.ylabel("count"); plt.title(title)
    plt.legend(); ensure_dir(Path(outpng).parent); plt.tight_layout(); plt.savefig(outpng); plt.close()

# ---------------- Driver: multiple jitters and/or sizes ----------------
def parse_float_list(s):
    return [float(t) for t in s.split(",") if t.strip()]

def run_lock(ax, ay, Lx_list, Ly_list, jitter_list, n, outdir, seed0, ridge=1e-11, target=0.231):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary_lines = []
    for Lx, Ly in zip(Lx_list, Ly_list):
        for jit in jitter_list:
            res, sin2_vals, cond_vals = one_setting_run(Lx, Ly, jit, ax, ay, n, seed0, ridge=ridge)
            rows.append([res["ax"],res["ay"],res["Lx"],res["Ly"],res["jitter"],res["n"],res["ok"],
                         res["sin2_mean"],res["sin2_std"],res["cond_mean"],res["cond_std"]])
            msg = (f"(Lx,Ly)=({Lx},{Ly})  jitter={jit:.3f}  "
                   f"sin2={res['sin2_mean']:.3f} ± {res['sin2_std']:.3f}  "
                   f"cond={res['cond_mean']:.3f} ± {res['cond_std']:.3f}  "
                   f"(ok={res['ok']}/{res['n']},  |Δ|={abs(res['sin2_mean']-target):.3f})")
            print(msg); summary_lines.append(msg+"\n")

            # per-setting outputs
            tag = f"L{Lx}x{Ly}_jit{jit:.3f}"
            write_csv(outdir / f"per_{tag}.csv",
                      header=["ax","ay","Lx","Ly","jitter","n","ok","sin2_mean","sin2_std","cond_mean","cond_std"],
                      rows_float2d=[rows[-1]])
            plot_hist(sin2_vals, f"sin^2 distribution @ {tag}", "sin2", outdir / f"hist_sin2_{tag}.png")
            plot_hist(cond_vals,  f"cond(G^-1) distribution @ {tag}", "cond", outdir / f"hist_cond_{tag}.png")

    # aggregates
    write_csv(outdir / "aggregates.csv",
              header=["ax","ay","Lx","Ly","jitter","n","ok","sin2_mean","sin2_std","cond_mean","cond_std"],
              rows_float2d=rows)

    write_text(outdir / "summary.txt",
               "=== DEC EW lock & report (invariant from G) ===\n"
               f"ax={ax} ay={ay}\n"
               f"L-sizes={list(zip(Lx_list,Ly_list))}\n"
               f"jitters={jitter_list}\n"
               f"n={n} ridge={ridge}\n\n" + "".join(summary_lines))

def main():
    ap = argparse.ArgumentParser(description="Lock + report sin^2(theta_W) at fixed (ax,ay) using invariant from G")
    ap.add_argument("--ax", type=float, default=2.57)
    ap.add_argument("--ay", type=float, default=0.78)
    ap.add_argument("--Lx_list", type=str, default="20,24")
    ap.add_argument("--Ly_list", type=str, default="20,24")
    ap.add_argument("--jitters", type=str, default="0.01,0.02,0.03")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--ridge", type=float, default=1e-11)
    ap.add_argument("--target", type=float, default=0.231)
    ap.add_argument("--out", type=str, default="out_lock_report")
    args = ap.parse_args()

    Lx_list = [int(x) for x in parse_float_list(args.Lx_list)]
    Ly_list = [int(y) for y in parse_float_list(args.Ly_list)]
    assert len(Lx_list) == len(Ly_list), "Lx_list and Ly_list must have same length"
    jitter_list = parse_float_list(args.jitters)

    run_lock(ax=args.ax, ay=args.ay,
             Lx_list=Lx_list, Ly_list=Ly_list,
             jitter_list=jitter_list, n=args.n,
             outdir=args.out, seed0=args.seed,
             ridge=args.ridge, target=args.target)

if __name__ == "__main__":
    main()