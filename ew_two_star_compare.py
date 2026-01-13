# ew_two_star_compare.py
# Compare four Hodge-star pairings for the two-star invariant:
#   (ref,mix) in {(cotan,cotan), (circum,circum), (cotan,circum), (circum,cotan)}
# For each pairing, sweep ridge over 1e-14..1e-9 and report sin^2(theta_W) mean±std.

import argparse, math
from pathlib import Path
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ---------- I/O ----------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def write_csv(path, header, rows):
    ensure_dir(Path(path).parent)
    arr = np.array(rows, float)
    np.savetxt(path, arr, delimiter=",", fmt="%.10g",
               header=",".join(header), comments="", encoding="utf-8")

# ---------- Mesh (periodic rectangular triangulation) ----------
def build_periodic_grid(Lx, Ly, jitter, ax=1.0, ay=1.0, seed=0):
    rng = np.random.default_rng(seed)
    def vid(ix, iy): return (iy % Ly) * Lx + (ix % Lx)

    # integer grid coordinates
    ix = np.arange(Lx); iy = np.arange(Ly)
    XX, YY = np.meshgrid(ix, iy, indexing="xy")
    v_ix = XX.ravel(); v_iy = YY.ravel()

    # geometric coordinates
    X = (v_ix.astype(float) / Lx) * ax
    Y = (v_iy.astype(float) / Ly) * ay
    P = np.column_stack([X, Y])
    if jitter > 0:
        dx = (ax / Lx) * jitter; dy = (ay / Ly) * jitter
        P += rng.uniform(-1, 1, size=P.shape) * np.array([dx, dy])

    # faces: two triangles per cell
    faces = []
    for jy in range(Ly):
        for jx in range(Lx):
            a = vid(jx,   jy)
            b = vid(jx+1, jy)
            c = vid(jx,   jy+1)
            d = vid(jx+1, jy+1)
            faces.append([a, b, d])
            faces.append([a, d, c])
    F = np.array(faces, dtype=int)

    # undirected edges + per-face edge indices and signs
    def undirected_key(u,v): return (u,v) if u<v else (v,u)
    undirected = {}
    E_pairs = []
    face_edges = np.zeros((F.shape[0],3), dtype=int)
    face_signs = np.zeros((F.shape[0],3), dtype=int)

    def edge_index_oriented(u,v):
        key = undirected_key(u,v)
        ei  = undirected.get(key)
        if ei is None:
            ei = len(E_pairs); undirected[key] = ei; E_pairs.append(key)
        a,b = key
        sgn = +1 if (u==a and v==b) else -1
        return ei, sgn

    for fi,(a,b,c) in enumerate(F):
        for k,(u,v) in enumerate([(a,b),(b,c),(c,a)]):
            ei,sgn = edge_index_oriented(u,v)
            face_edges[fi,k] = ei
            face_signs[fi,k] = sgn

    E = np.array(E_pairs, dtype=int)
    V = P.shape[0]; Ecount = E.shape[0]; Fcount = F.shape[0]

    # d0 (E x V)
    rows, cols, data = [], [], []
    for ei,(i,j) in enumerate(E):
        rows += [ei, ei]; cols += [i, j]; data += [-1.0, +1.0]
    d0 = sp.csr_matrix((data, (rows, cols)), shape=(Ecount, V))

    return P, E, F, d0, face_edges, face_signs, v_ix, v_iy

# ---------- DEC stars ----------
def cot_angle(A,B,C):
    v1 = B-A; v2 = C-A
    dot = float(np.dot(v1,v2))
    nrm = math.sqrt(max(np.dot(v1,v1)*np.dot(v2,v2), 1e-30))
    c = max(min(dot/nrm, 1.0), -1.0)
    s = math.sqrt(max(1.0 - c*c, 0.0))
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

def circumcenter(A,B,C):
    # robust 2D circumcenter
    a = B - A; b = C - A
    adot = np.dot(a,a); bdot = np.dot(b,b)
    cross = a[0]*b[1] - a[1]*b[0]
    denom = 2.0 * (cross if abs(cross)>1e-18 else math.copysign(1e-18,cross if cross!=0 else 1.0))
    ux = A[0] + (bdot*a[1] - adot*b[1]) / denom
    uy = A[1] + (adot*b[0] - bdot*a[0]) / denom
    return np.array([ux, uy], float)

def star1_circum(P, E, F, face_edges):
    # Diagonal entries ≈ dual_edge_length / primal_edge_length
    Ecount = E.shape[0]; Fcount = F.shape[0]
    # face circumcenters
    Cc = np.zeros((Fcount,2))
    for fi,(i,j,k) in enumerate(F):
        Cc[fi] = circumcenter(P[i], P[j], P[k])

    # map edges -> adjacent faces (two on a torus)
    edge_faces = [[] for _ in range(Ecount)]
    for fi in range(Fcount):
        for k in range(3):
            edge_faces[face_edges[fi,k]].append(fi)

    vals = np.zeros(Ecount)
    for ei,(i,j) in enumerate(E):
        le = npl.norm(P[j]-P[i]) + 1e-12
        adj = edge_faces[ei]
        if len(adj)==2:
            Ld = npl.norm(Cc[adj[1]] - Cc[adj[0]])
        elif len(adj)==1:
            # fallback (shouldn’t happen on torus triangulation)
            Ld = npl.norm(Cc[adj[0]] - 0.5*(P[i]+P[j]))
        else:
            Ld = le
        vals[ei] = max(Ld,1e-12) / le
    vals = np.maximum(vals, 1e-12)
    return sp.diags(vals, 0, format="csc")

# ---------- Exact fundamental cycles (rank-2, topological) ----------
def build_exact_cycles(Lx, Ly, E):
    def vid(ix, iy): return (iy % Ly) * Lx + (ix % Lx)
    undirected = {(min(i, j), max(i, j)): ei for ei,(i,j) in enumerate(E)}

    def edge_index_oriented(i, j):
        a,b = (i,j) if i<j else (j,i)
        ei = undirected.get((a,b))
        if ei is None: return None, 0
        sgn = +1 if (i<j) else -1
        return ei, sgn

    C = np.zeros((2, E.shape[0]), float)

    # horizontal loop at y=0
    y0=0
    for x in range(Lx):
        u = vid(x,y0); v = vid(x+1,y0)
        ei,sgn = edge_index_oriented(u,v)
        if ei is not None: C[0,ei] += sgn

    # vertical loop at x=0
    x0=0
    for y in range(Ly):
        u = vid(x0,y); v = vid(x0,y+1)
        ei,sgn = edge_index_oriented(u,v)
        if ei is not None: C[1,ei] += sgn

    if np.linalg.matrix_rank(C) < 2:
        raise RuntimeError("Cycle matrix C lost rank.")
    return sp.csr_matrix(C)

# ---------- Two-star invariant ----------
def sin2_two_star_once(L, jitter, ax, ay, ridge, star_ref, star_mix, seed):
    P,E,F,d0,fe,fs,_,_ = build_periodic_grid(L, L, jitter, ax, ay, seed)
    C = build_exact_cycles(L, L, E)

    # choose stars
    if star_ref == "cotan":  A = star1_cotan(P,E,F,fe)
    else:                    A = star1_circum(P,E,F,fe)
    if star_mix == "cotan":  A2 = star1_cotan(P,E,F,fe)
    else:                    A2 = star1_circum(P,E,F,fe)

    # solve A X = C^T  and A2 Y = C^T
    I = sp.eye(A.shape[0], format="csc")
    X = spla.spsolve(A  + ridge*I, C.T)  # (E x 2)
    Y = spla.spsolve(A2 + ridge*I, C.T)

    Ginv = (C @ X)  # 2x2
    Ginv2= (C @ Y)  # 2x2
    # sandwich with the mixed metric: G = (Ginv)^{-1/2} * Ginv2 * (Ginv)^{-1/2}
    # but for ratio we can equivalently compute eigs of (Ginv^{-1} Ginv2)
    # safer: form M = inv(Ginv) @ Ginv2
    det = Ginv[0,0]*Ginv[1,1] - Ginv[0,1]*Ginv[1,0]
    if not np.isfinite(det) or abs(det) < 1e-18:
        return np.nan
    G = (1.0/det) * np.array([[ Ginv[1,1], -Ginv[0,1]],
                              [-Ginv[1,0],  Ginv[0,0]]], float)
    M = G @ Ginv2
    # symmetrize for safety (it should be SPD)
    Ms = 0.5*(M + M.T)
    tr = Ms[0,0] + Ms[1,1]
    d  = (Ms[0,0]-Ms[1,1])**2 + 4*Ms[0,1]*Ms[1,0]
    d  = max(d, 0.0)
    s  = math.sqrt(d)
    lam_min = 0.5*(tr - s)
    lam_max = 0.5*(tr + s)
    if lam_min <= 0 or lam_max <= 0: return np.nan
    return float(lam_min / (lam_min + lam_max))

def sin2_two_star(L, jitter, n, ax, ay, star_ref, star_mix, ridge, seed0=2025):
    vals=[]
    for k in range(n):
        s = seed0 + 97*k
        v = sin2_two_star_once(L, jitter, ax, ay, ridge, star_ref, star_mix, s)
        if np.isfinite(v): vals.append(v)
    if len(vals)==0: return np.nan, np.nan
    arr=np.array(vals,float)
    return float(arr.mean()), float(arr.std(ddof=0))

# ---------- Runner ----------
def main():
    ap = argparse.ArgumentParser(description="Two-star invariant: compare star pairings over ridge sweep")
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ax", type=float, default=2.57)
    ap.add_argument("--ay", type=float, default=0.78)
    ap.add_argument("--out", type=str, default="out_two_star_compare")
    args = ap.parse_args()

    outdir = Path(args.out); ensure_dir(outdir)

    ridge_list = np.logspace(-14, -9, 10)
    pairings = [("cotan","cotan"), ("circum","circum"),
                ("cotan","circum"), ("circum","cotan")]

    print("=== Two-star invariant comparison ===")
    print(f"L={args.L} jitter={args.jitter} n={args.n}  (ax,ay)=({args.ax},{args.ay})")
    rows = [["star_ref","star_mix","ridge","sin2_mean","sin2_std"]]

    for sref, smix in pairings:
        print(f"\nstars: ref={sref}  mix={smix}")
        for r in ridge_list:
            m,s = sin2_two_star(args.L, args.jitter, args.n, args.ax, args.ay,
                                star_ref=sref, star_mix=smix, ridge=r, seed0=2025)
            print(f"ridge={r: .1e}  sin2={m:.6f} ± {s:.6f}")
            rows.append([sref, smix, r, m, s])

    write_csv(outdir / "two_star_ridge.csv",
              rows[0], rows[1:])

    # quick plot: one curve per pairing
    plt.figure(figsize=(6.6,4.4), dpi=140)
    for sref, smix in pairings:
        mask = [ (rr[0]==sref and rr[1]==smix) for rr in zip(
                 [r[0] for r in rows[1:]], [r[1] for r in rows[1:]]) ]
        data = np.array([rows[1:][i] for i,b in enumerate(mask) if b], float)
        if data.size==0: continue
        rvals = data[:,2]; mvals = data[:,3]; svals = data[:,4]
        plt.errorbar(rvals, mvals, yerr=svals, fmt="o-", capsize=3, label=f"{sref}→{smix}")
    plt.axhline(0.231, ls="--", lw=1, label="target 0.231")
    plt.xscale("log")
    plt.xlabel("ridge"); plt.ylabel("sin^2(theta_W)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "two_star_ridge.png")
    plt.close()

if __name__ == "__main__":
    main()