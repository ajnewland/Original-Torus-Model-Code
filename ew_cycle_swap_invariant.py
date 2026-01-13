# ew_cycle_swap_invariant.py
# Cycle-swap robustness using exact fundamental loops (rank-2), with
# gauge-invariant two-star readout:
#   sin^2 = λ_min(G_mix) / (λ_min(G_mix) + λ_max(G_mix))
# where G_mix = (C A_ref^{-1} C^T)^{-1} (C A_mix^{-1} C^T).
# Defaults: ref=cotan, mix=circum (strongest signal).
# Writes per-shift CSV and a small bar plot.

import argparse, math
from pathlib import Path
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ---------------- I/O ----------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def write_text(path, text):
    path = Path(path); ensure_dir(path.parent); path.write_text(text, encoding="utf-8")

def write_csv(path, header, rows):
    path = Path(path); ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

# ---------------- Mesh: periodic rectangular triangulation ----------------
def build_periodic_grid(Lx, Ly, jitter, ax=1.0, ay=1.0, seed=0):
    """
    Returns:
      P: (V,2) coordinates
      E: (E,2) undirected edges (min,max)
      F: (F,3) faces
      d0: (E,V) incidence
      face_edges, face_signs: per face edge id and orientation
      v_ix, v_iy: integer grid coords of each vertex
    """
    rng = np.random.default_rng(seed)
    def vid(ix, iy): return (iy % Ly) * Lx + (ix % Lx)

    # integer coords
    ix = np.arange(Lx); iy = np.arange(Ly)
    XX, YY = np.meshgrid(ix, iy, indexing="xy")
    v_ix = XX.ravel()
    v_iy = YY.ravel()

    # geometric coords on torus cell
    X = (v_ix.astype(float) / Lx) * ax
    Y = (v_iy.astype(float) / Ly) * ay
    P = np.column_stack([X, Y])
    if jitter > 0.0:
        dx = (ax / Lx) * jitter
        dy = (ay / Ly) * jitter
        P += rng.uniform(-1, 1, size=P.shape) * np.array([dx, dy])

    # faces: 2 per cell
    faces = []
    for jy in range(Ly):
        for jx in range(Lx):
            a = vid(jx,   jy)
            b = vid(jx+1, jy)
            c = vid(jx,   jy+1)
            d = vid(jx+1, jy+1)
            faces.append([a,b,d])
            faces.append([a,d,c])
    F = np.array(faces, dtype=int)

    # undirected edges
    def undirected_key(u,v): return (u,v) if u<v else (v,u)
    undirected = {}
    E_pairs = []
    face_edges = np.zeros((F.shape[0],3), dtype=int)
    face_signs = np.zeros((F.shape[0],3), dtype=int)

    def edge_index_oriented(u,v):
        key = undirected_key(u,v)
        ei = undirected.get(key)
        if ei is None:
            ei = len(E_pairs)
            undirected[key] = ei
            E_pairs.append(key)
        a,b = key
        sgn = +1 if (u==a and v==b) else -1
        return ei, sgn

    for fi,(a,b,c) in enumerate(F):
        for k,(u,v) in enumerate([(a,b),(b,c),(c,a)]):
            ei, sgn = edge_index_oriented(u,v)
            face_edges[fi,k] = ei
            face_signs[fi,k] = sgn

    E = np.array(E_pairs, dtype=int)
    V = P.shape[0]; Ecount = E.shape[0]

    # d0 (E x V)
    rows, cols, data = [], [], []
    for ei,(i,j) in enumerate(E):
        rows += [ei, ei]; cols += [i, j]; data += [-1.0, +1.0]
    d0 = sp.csr_matrix((data, (rows, cols)), shape=(Ecount, V))

    return P, E, F, d0, face_edges, face_signs, v_ix, v_iy

# ---------------- DEC: Hodge stars on 1-forms ----------------
def cot_angle(A,B,C):
    v1 = B-A; v2 = C-A
    dot = float(np.dot(v1,v2))
    nrm = math.sqrt(max(np.dot(v1,v1)*np.dot(v2,v2), 1e-30))
    c = max(min(dot/nrm,1.0),-1.0)
    s = math.sqrt(max(1.0-c*c, 0.0))
    return 0.0 if s < 1e-14 else c/s

def star1_cotan(P, E, F, face_edges):
    Ecount = E.shape[0]; Fcount = F.shape[0]
    edge_faces = [[] for _ in range(Ecount)]
    for fi in range(Fcount):
        for k in range(3):
            edge_faces[face_edges[fi,k]].append(fi)
    face_pts = P[F]  # (F,3,2)
    cot_opp = np.zeros((Fcount,3))
    for fi in range(Fcount):
        A,B,C = face_pts[fi]
        cotA = cot_angle(A,B,C)
        cotB = cot_angle(B,C,A)
        cotC = cot_angle(C,A,B)
        cot_opp[fi,0] = cotC  # opp (A,B)
        cot_opp[fi,1] = cotA  # opp (B,C)
        cot_opp[fi,2] = cotB  # opp (C,A)
    vals = np.zeros(Ecount)
    for ei,(i,j) in enumerate(E):
        le = npl.norm(P[j]-P[i])
        csum = 0.0
        for fi in edge_faces[ei]:
            a,b,c = F[fi]
            for k,(u,v) in enumerate([(a,b),(b,c),(c,a)]):
                if (u==i and v==j) or (u==j and v==i):
                    csum += cot_opp[fi,k]
        vals[ei] = 0.5 * max(csum,0.0) * max(le,1e-12)
    vals = np.maximum(vals, 1e-12)
    return sp.diags(vals, 0, format="csc")

def star1_circum(P, E, F, face_edges):
    # simple circumcentric dual-length surrogate (sum of dual lengths per edge)
    # robust & positive; exact constants cancel in the invariant.
    V = P
    vals = np.zeros(E.shape[0])
    # build for each face its three edge ids
    tri_edges = face_edges
    # circumcenter of a triangle
    def circumcenter(A,B,C):
        a = B - A; b = C - A
        aa = np.dot(a,a); bb = np.dot(b,b); ab = np.dot(a,b)
        det = aa*bb - ab*ab
        if abs(det) < 1e-20:  # near-collinear fallback to centroid
            return (A+B+C)/3.0
        x = (bb*np.dot(a,[1,0]) - ab*np.dot(b,[1,0]))/det
        y = (bb*np.dot(a,[0,1]) - ab*np.dot(b,[0,1]))/det
        return A + x*a + y*b
    # accumulate dual lengths from adjacent faces (Voronoi)
    face_cc = np.zeros((F.shape[0],2))
    for fi,(i,j,k) in enumerate(F):
        face_cc[fi] = circumcenter(V[i],V[j],V[k])
    # each edge gets contributions from its incident faces (distance between cc projected)
    for fi,(i,j,k) in enumerate(F):
        cc = face_cc[fi]
        for kidx,(u,v) in enumerate([(i,j),(j,k),(k,i)]):
            ei = tri_edges[fi,kidx]
            mid = 0.5*(V[u]+V[v])
            vals[ei] += npl.norm(cc - mid)
    vals = np.maximum(vals, 1e-12)
    return sp.diags(vals, 0, format="csc")

def get_star(kind, P, E, F, face_edges):
    if kind == "cotan":   return star1_cotan(P,E,F,face_edges)
    if kind == "circum":  return star1_circum(P,E,F,face_edges)
    raise ValueError("star kind must be 'cotan' or 'circum'")

# ---------------- Exact cycles with shift ----------------
def build_exact_cycles_shift(Lx, Ly, E, xshift=0, yshift=0):
    """
    Two rank-2 cycles:
      row 0: horizontal loop along row y=yshift
      row 1: vertical   loop along col x=xshift
    Independent of geometry/jitter; guaranteed rank-2.
    """
    def vid(ix, iy): return (iy % Ly) * Lx + (ix % Lx)
    # undirected edge index
    umap = {(min(i,j), max(i,j)): ei for ei,(i,j) in enumerate(E)}
    def eidx(i,j):
        a,b = (i,j) if i<j else (j,i)
        ei = umap.get((a,b), None)
        if ei is None: return None, 0
        sgn = +1 if (i<j) else -1
        return ei, sgn

    C = np.zeros((2, E.shape[0]), float)
    y0 = int(yshift) % Ly
    for x in range(Lx):
        u = vid(x, y0); v = vid(x+1, y0)
        ei, s = eidx(u,v)
        if ei is not None: C[0, ei] += s
    x0 = int(xshift) % Lx
    for y in range(Ly):
        u = vid(x0, y); v = vid(x0, y+1)
        ei, s = eidx(u,v)
        if ei is not None: C[1, ei] += s

    if np.linalg.matrix_rank(C) < 2:
        raise RuntimeError("Cycle matrix C lost rank (<2).")
    return sp.csr_matrix(C)

# ---------------- Invariant from two stars ----------------
def sin2_two_star(star_ref, star_mix, C, ridge=1e-12, cond_cap=1e12):
    """
    Compute sin^2 from the mixed invariant:
      G_ref = C A_ref^{-1} C^T   (2x2)
      G_mix = C A_mix^{-1} C^T   (2x2)
      M = G_ref^{-1} G_mix
      sin^2 = λ_min(M) / (λ_min(M) + λ_max(M))
    Returns (sin2, condM).
    """
    try:
        Aref = star_ref.tocsc(); Amix = star_mix.tocsc()
        Iref = ridge * sp.eye(Aref.shape[0], format="csc")
        Imix = ridge * sp.eye(Amix.shape[0], format="csc")
        Xr = spla.spsolve(Aref + Iref, C.T)  # (E x 2)
        Xm = spla.spsolve(Amix + Imix, C.T)  # (E x 2)
        Gref = (C @ Xr)
        Gmix = (C @ Xm)
        # symmetrize
        Gref = 0.5*(Gref + Gref.T)
        Gmix = 0.5*(Gmix + Gmix.T)
        # invert 2x2 Gref
        det = Gref[0,0]*Gref[1,1] - Gref[0,1]*Gref[1,0]
        if not np.isfinite(det) or abs(det) < 1e-18: return np.nan, np.nan
        Gref_inv = (1.0/det) * np.array([[ Gref[1,1], -Gref[0,1]],
                                         [-Gref[1,0],  Gref[0,0]]], float)
        M = Gref_inv @ Gmix
        # eigenvalues of 2x2 M
        tr = M[0,0]+M[1,1]
        d  = (M[0,0]-M[1,1])**2 + 4*M[0,1]*M[1,0]
        d  = max(d, 0.0); s = math.sqrt(d)
        lam_min = 0.5*(tr - s); lam_max = 0.5*(tr + s)
        if lam_min <= 0 or lam_max <= 0: return np.nan, np.nan
        cond = lam_max/lam_min
        if not np.isfinite(cond) or cond > cond_cap: return np.nan, np.nan
        sin2 = lam_min / (lam_min + lam_max)
        return float(sin2), float(cond)
    except Exception:
        return np.nan, np.nan

# ---------------- Per-shift evaluation ----------------
def eval_shift(L, jitter, n, ax, ay, xshift, yshift, star_ref_kind, star_mix_kind, ridge, seed0=2025):
    sin2s=[]; conds=[]
    ok=0
    for k in range(n):
        seed = seed0 + 97*k
        P,E,F,d0,fe,fs,vix,viy = build_periodic_grid(L, L, jitter, ax=ax, ay=ay, seed=seed)
        C = build_exact_cycles_shift(L, L, E, xshift, yshift)
        star_ref = get_star(star_ref_kind, P,E,F,fe)
        star_mix = get_star(star_mix_kind, P,E,F,fe)
        s, c = sin2_two_star(star_ref, star_mix, C, ridge=ridge)
        if np.isfinite(s): sin2s.append(s); conds.append(c); ok+=1
    if ok==0:
        return dict(xshift=xshift,yshift=yshift, ok=0, sin2_mean=np.nan, sin2_std=np.nan,
                    cond_mean=np.nan, cond_std=np.nan)
    return dict(xshift=xshift,yshift=yshift, ok=ok,
                sin2_mean=float(np.mean(sin2s)),
                sin2_std=float(np.std(sin2s)),
                cond_mean=float(np.mean(conds)),
                cond_std=float(np.std(conds)))

# ---------------- Plot ----------------
def plot_bars(results, target, outpng):
    labels = [f"{r['xshift']},{r['yshift']}" for r in results]
    means  = [r['sin2_mean'] for r in results]
    stds   = [r['sin2_std']  for r in results]
    x = np.arange(len(labels))
    plt.figure(figsize=(7.2,4.2), dpi=140)
    plt.errorbar(x, means, yerr=stds, fmt="o", capsize=3)
    plt.axhline(target, ls="--", lw=1.0, label=f"target {target:.3f}")
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("sin^2(theta_W)"); plt.xlabel("cycle shift (x,y)")
    plt.grid(alpha=0.2); plt.legend()
    ensure_dir(Path(outpng).parent); plt.tight_layout(); plt.savefig(outpng); plt.close()

# ---------------- Main ----------------
def parse_shifts(s: str):
    out=[]
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok: continue
        a,b = tok.split(",")
        out.append((int(a), int(b)))
    return out

def main():
    ap = argparse.ArgumentParser(description="Cycle-swap robustness (exact cycles, two-star invariant)")
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ax", type=float, default=2.57)
    ap.add_argument("--ay", type=float, default=0.78)
    ap.add_argument("--shifts", type=str, default="0,0;1,0;0,1;3,2;5,5")
    ap.add_argument("--star_ref", type=str, choices=["cotan","circum"], default="cotan")
    ap.add_argument("--star_mix", type=str, choices=["cotan","circum"], default="circum")
    ap.add_argument("--ridge", type=float, default=1e-11)
    ap.add_argument("--target", type=float, default=0.231)
    ap.add_argument("--out", type=str, default="out_cycles")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    outdir = Path(args.out); ensure_dir(outdir)
    shifts = parse_shifts(args.shifts)

    print("=== EW cycle-swap (invariant, exact loops) ===")
    print(f"L={args.L} jitter={args.jitter} n={args.n}  (ax,ay)=({args.ax},{args.ay})")
    print(f"stars: ref={args.star_ref}  mix={args.star_mix}  ridge={args.ridge:.1e}")
    print(f"shifts={shifts}  out={args.out}")

    results=[]
    lines=[]
    for (sx,sy) in shifts:
        r = eval_shift(args.L, args.jitter, args.n, args.ax, args.ay,
                       sx, sy, args.star_ref, args.star_mix, args.ridge, seed0=args.seed)
        results.append(r)
        line = (f"shift=(x={sx},y={sy})  "
                f"sin2={r['sin2_mean']:.6f} ± {r['sin2_std']:.6f}  "
                f"cond~{r['cond_mean']:.3f}  (ok={r['ok']}/{args.n})")
        print(line); lines.append(line+"\n")

    # CSV
    header = ["xshift","yshift","sin2_mean","sin2_std","cond_mean","cond_std","ok","n"]
    rows = [[r["xshift"], r["yshift"], f"{r['sin2_mean']:.10g}", f"{r['sin2_std']:.10g}",
             f"{r['cond_mean']:.10g}", f"{r['cond_std']:.10g}", r["ok"], args.n] for r in results]
    write_csv(outdir / "cycle_swap.csv", header, rows)

    # Summary & plot
    write_text(outdir / "summary.txt",
               "=== EW cycle-swap (invariant, exact loops) ===\n"
               f"L={args.L} jitter={args.jitter} n={args.n}  (ax,ay)=({args.ax},{args.ay})\n"
               f"stars: ref={args.star_ref}  mix={args.star_mix}  ridge={args.ridge:.1e}\n"
               f"shifts={shifts}\n" + "".join(lines))
    plot_bars(results, args.target, outdir / "cycle_swap_sin2.png")

if __name__ == "__main__":
    main()