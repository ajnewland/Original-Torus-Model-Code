# ew_cycle_invariant_pure.py
# Cycle-swap test using ONLY the invariant sin^2 from G = (C A^{-1} C^T)^{-1}.
# - Periodic triangulated torus (Lx x Ly cells, two triangles per cell)
# - Hodge star on edges: cotan or circumcentric (choose with --star)
# - Exact fundamental loops C built at shifts (dx,dy) to prove cycle independence
# Prints sin^2 mean/std for each shift; also condition number of G^{-1}.

import argparse, math
from pathlib import Path
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------- I/O helpers ----------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def write_text(path, text):
    path = Path(path); ensure_dir(path.parent); path.write_text(text, encoding="utf-8")

# ---------------- periodic rectangular triangulation ----------------
def build_periodic_grid(Lx, Ly, jitter, ax=1.0, ay=1.0, seed=0):
    """
    Returns:
      P: (V,2) vertex coords
      E: (E,2) undirected edges with increasing (min,max)
      F: (F,3) faces (CCW)
      d0: (E,V) incidence
      face_edges, face_signs: per face edge ids & orientation
      v_ix, v_iy: integer grid coords of each vertex
    """
    rng = np.random.default_rng(seed)

    def vid(ix, iy): return (iy % Ly) * Lx + (ix % Lx)

    v_ix, v_iy = np.meshgrid(np.arange(Lx), np.arange(Ly), indexing="xy")
    v_ix = v_ix.ravel(); v_iy = v_iy.ravel()

    X = (v_ix.astype(float) / Lx) * ax
    Y = (v_iy.astype(float) / Ly) * ay
    P = np.column_stack([X, Y])
    if jitter > 0:
        P += rng.uniform(-1, 1, size=P.shape) * np.array([ax/Lx*jitter, ay/Ly*jitter])

    # faces
    faces = []
    for iy in range(Ly):
        for ix in range(Lx):
            a = vid(ix, iy); b = vid(ix+1, iy); c = vid(ix, iy+1); d = vid(ix+1, iy+1)
            faces.append([a, b, d])
            faces.append([a, d, c])
    F = np.array(faces, dtype=int)

    # undirected edges with index, plus per-face mapping
    def ukey(i,j): return (i,j) if i<j else (j,i)
    undirected = {}
    E_pairs = []
    face_edges = np.zeros((F.shape[0],3), dtype=int)
    face_signs = np.zeros((F.shape[0],3), dtype=int)
    def edge_index_oriented(u,v):
        k = ukey(u,v); ei = undirected.get(k)
        if ei is None:
            ei = len(E_pairs); undirected[k]=ei; E_pairs.append(k)
        a,b = k; sgn = +1 if (u==a and v==b) else -1
        return ei, sgn
    for fi,(a,b,c) in enumerate(F):
        for k,(u,v) in enumerate([(a,b),(b,c),(c,a)]):
            ei,sgn = edge_index_oriented(u,v)
            face_edges[fi,k]=ei; face_signs[fi,k]=sgn
    E = np.array(E_pairs, dtype=int)
    V = P.shape[0]; Ee = E.shape[0]; Ff = F.shape[0]

    # d0 (E,V)
    rows, cols, data = [], [], []
    for ei,(i,j) in enumerate(E):
        rows += [ei, ei]; cols += [i, j]; data += [-1.0, +1.0]
    d0 = sp.csr_matrix((data, (rows, cols)), shape=(Ee, V))
    return P,E,F,d0,face_edges,face_signs,v_ix,v_iy

# ---------------- Hodge stars on edges ----------------
def cot_angle(A,B,C):
    v1 = B-A; v2 = C-A
    dot = float(np.dot(v1,v2))
    nrm = math.sqrt(max(np.dot(v1,v1)*np.dot(v2,v2), 1e-30))
    c = max(min(dot/nrm,1.0),-1.0)
    s = math.sqrt(max(1.0-c*c,0.0))
    return 0.0 if s<1e-14 else c/s

def star1_cotan(P,E,F,face_edges):
    Ee = E.shape[0]; Ff = F.shape[0]
    edge_faces = [[] for _ in range(Ee)]
    for fi in range(Ff):
        for k in range(3):
            edge_faces[face_edges[fi,k]].append(fi)

    face_pts = P[F]  # (F,3,2)
    cot_opp = np.zeros((Ff,3))
    for fi in range(Ff):
        A,B,C = face_pts[fi]
        cotA = cot_angle(A,B,C)
        cotB = cot_angle(B,C,A)
        cotC = cot_angle(C,A,B)
        cot_opp[fi,0] = cotC
        cot_opp[fi,1] = cotA
        cot_opp[fi,2] = cotB

    vals = np.zeros(Ee)
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

def tri_area(p,q,r):
    # robust 2D area (works in NumPy 2.x: promote to 3D cross if needed)
    v1 = np.array([q[0]-p[0], q[1]-p[1], 0.0])
    v2 = np.array([r[0]-p[0], r[1]-p[1], 0.0])
    return 0.5 * np.linalg.norm(np.cross(v1,v2))

def star1_circum(P,E,F,face_edges):
    # dual length along circumcentric dual per edge: sum of adjacent dual segments
    Ee = E.shape[0]; Ff = F.shape[0]
    # build per-face circumcenters (2D)
    def circumcenter(A,B,C):
        a = B - A; b = C - A
        adot = np.dot(a,a); bdot = np.dot(b,b)
        cross = a[0]*b[1]-a[1]*b[0]
        if abs(cross) < 1e-14:
            return (A+B+C)/3.0
        cx = A + (bdot*np.array([a[1],-a[0]]) - adot*np.array([b[1],-b[0]])) / (2*cross)
        return cx
    centers = np.zeros((Ff,2))
    for fi,(i,j,k) in enumerate(F):
        centers[fi] = circumcenter(P[i],P[j],P[k])

    # edge to incident faces
    edge_faces = [[] for _ in range(Ee)]
    for fi in range(Ff):
        for k in range(3):
            edge_faces[face_edges[fi,k]].append(fi)

    vals = np.zeros(Ee)
    for ei,(i,j) in enumerate(E):
        le = npl.norm(P[j]-P[i])
        inc = edge_faces[ei]
        if len(inc)==2:
            c1, c2 = centers[inc[0]], centers[inc[1]]
            dual_len = npl.norm(c2-c1)
        elif len(inc)==1:
            # topologically shouldn’t happen on a torus; fall back
            dual_len = le
        else:
            dual_len = le
        vals[ei] = max(dual_len, 1e-12)
    return sp.diags(vals, 0, format="csc")

# ---------------- exact fundamental cycles at shift (dx,dy) ----------------
def build_exact_cycles_shifted(Lx, Ly, E, dx=0, dy=0):
    """
    C (2xE). Row 0: horizontal loop at row y=dy. Row 1: vertical loop at col x=dx.
    Orientation along +x and +y respectively (periodic wrap).
    """
    def vid(ix,iy): return (iy % Ly)*Lx + (ix % Lx)
    undirected = {(min(i,j),max(i,j)): ei for ei,(i,j) in enumerate(E)}
    def edge_index_oriented(i,j):
        a,b = (i,j) if i<j else (j,i)
        ei = undirected.get((a,b))
        if ei is None: return None, 0
        sgn = +1 if (i<j) else -1
        return ei, sgn

    C = np.zeros((2, E.shape[0]), float)

    y = dy % Ly
    for x in range(Lx):
        u = vid(x, y); v = vid(x+1, y)
        ei, s = edge_index_oriented(u,v)
        if ei is not None: C[0, ei] += s

    x = dx % Lx
    for y in range(Ly):
        u = vid(x, y); v = vid(x, y+1)
        ei, s = edge_index_oriented(u,v)
        if ei is not None: C[1, ei] += s

    # rank check
    if np.linalg.matrix_rank(C) < 2:
        raise RuntimeError("Cycle matrix C has rank < 2 (degenerate loops).")
    return sp.csr_matrix(C)

# ---------------- invariant sin^2 from G ----------------
def sin2_from_G(star1, C, ridge=1e-11, cond_cap=1e12):
    """
    Solve (A+ridge I) X = C^T, G^{-1} = C X, return sin^2 from eigenvalues of G.
    """
    A = star1.tocsc()
    I = sp.eye(A.shape[0], format="csc")
    # factorize once, solve for two RHS columns
    solver = spla.factorized(A + ridge*I)
    Ct = C.T.toarray()  # (E,2)
    X = np.column_stack([solver(Ct[:,0]), solver(Ct[:,1])])  # (E,2)
    Ginv = (C @ X)  # 2x2 dense
    Ginv = 0.5 * (Ginv + Ginv.T)
    # condition of Ginv (2x2) via eigenvalues
    t = float(Ginv[0,0] + Ginv[1,1])
    d = (Ginv[0,0]-Ginv[1,1])**2 + 4.0*Ginv[0,1]*Ginv[1,0]
    d = max(d, 0.0)
    s = math.sqrt(d)
    lam_min_inv = 0.5*(t - s); lam_max_inv = 0.5*(t + s)
    if lam_min_inv <= 0 or lam_max_inv <= 0: return np.nan, np.nan
    condGinv = lam_max_inv / lam_min_inv
    if not np.isfinite(condGinv) or condGinv > cond_cap: return np.nan, np.nan
    # invert 2x2 explicitly
    det = Ginv[0,0]*Ginv[1,1] - Ginv[0,1]*Ginv[1,0]
    if abs(det) < 1e-18: return np.nan, np.nan
    G = (1.0/det) * np.array([[ Ginv[1,1], -Ginv[0,1]],
                              [-Ginv[1,0],  Ginv[0,0]]], float)
    w = np.linalg.eigvalsh(G)
    if w[0] <= 0 or w[1] <= 0: return np.nan, np.nan
    sin2 = float(w[0] / (w[0] + w[1]))
    return sin2, condGinv

# ---------------- star selector ----------------
def star1_from_name(name, P,E,F,fe):
    if name == "cotan":   return star1_cotan(P,E,F,fe)
    if name == "circum":  return star1_circum(P,E,F,fe)
    raise ValueError("star must be 'cotan' or 'circum'")

# ---------------- driver ----------------
def parse_shifts(s: str):
    out=[]
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok: continue
        a,b = tok.split(",")
        out.append((int(a), int(b)))
    return out

def one_setting(L, jitter, ax, ay, star, dx, dy, n, seed0, ridge):
    sin2s=[]; conds=[]
    for k in range(n):
        P,E,F,d0,fe,fs,vx,vy = build_periodic_grid(L, L, jitter, ax=ax, ay=ay, seed=seed0+97*k)
        A = star1_from_name(star, P,E,F,fe)
        C = build_exact_cycles_shifted(L, L, E, dx=dx, dy=dy)
        s2, cond = sin2_from_G(A, C, ridge=ridge)
        if np.isfinite(s2):
            sin2s.append(s2); conds.append(cond)
    if len(sin2s)==0:
        return np.nan, np.nan, 0
    return float(np.mean(sin2s)), float(np.std(sin2s)), len(sin2s), float(np.mean(conds))

def main():
    ap = argparse.ArgumentParser(description="Cycle-swap test using invariant sin^2 from G")
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ax", type=float, default=2.57)
    ap.add_argument("--ay", type=float, default=0.78)
    ap.add_argument("--star", type=str, default="cotan", choices=["cotan","circum"])
    ap.add_argument("--shifts", type=str, default="0,0;1,0;0,1;3,2;5,5",
                    help="semicolon list of dx,dy pairs")
    ap.add_argument("--ridge", type=float, default=1e-11)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    shifts = parse_shifts(args.shifts)
    print("=== EW cycle-swap (invariant, exact loops) ===")
    print(f"L={args.L} jitter={args.jitter} n={args.n}  (ax,ay)=({args.ax},{args.ay})")
    print(f"star={args.star}  ridge={args.ridge:.1e}")
    print(f"shifts={shifts}")

    lines=[]
    for (dx,dy) in shifts:
        m,s,ok,cond = one_setting(args.L, args.jitter, args.ax, args.ay,
                                  args.star, dx, dy, args.n, args.seed, args.ridge)
        line = f"shift=(x={dx},y={dy})  sin2={m:.6f} ± {s:.6f}  cond~{cond:.3f}  (ok={ok}/{args.n})"
        print(line); lines.append(line+"\n")

    if args.out:
        write_text(Path(args.out)/"cycle_swap_summary.txt",
                   "=== EW cycle-swap (invariant, exact loops) ===\n"+
                   f"L={args.L} jitter={args.jitter} n={args.n}  (ax,ay)=({args.ax},{args.ay})\n"+
                   f"star={args.star}  ridge={args.ridge:.1e}\n"+
                   f"shifts={shifts}\n" + "".join(lines))

if __name__ == "__main__":
    main()