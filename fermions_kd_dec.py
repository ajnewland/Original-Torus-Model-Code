# fermions_kd_dec.py
# Kähler–Dirac (toy) on periodic triangulated torus + robust smallest-eigs
# Usage example (Windows cmd with ^ continuation):
#   python fermions_kd_dec.py ^
#     --Lx 20 --Ly 20 --jitter 0.02 ^
#     --ax 2.59 --ay 0.78 ^
#     --star cotan --y 0.5 --v 246 ^
#     --n 16 --k 16 ^
#     --out out_kd_free

import argparse, math
from pathlib import Path
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------- I/O ----------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def write_text(path, s):
    path = Path(path); ensure_dir(path.parent); path.write_text(s, encoding="utf-8")
def save_csv(path, header, rows):
    path = Path(path); ensure_dir(path.parent)
    arr = np.array(rows, float)
    np.savetxt(path, arr, delimiter=",", fmt="%.10g",
               header=",".join(header), comments="", encoding="utf-8")

# ---------- Geometry ----------
def tri_area(p, q, r):
    # robust scalar area (no 3D cross needed)
    return 0.5 * abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]))

def build_periodic_grid(Lx, Ly, jitter, ax=1.0, ay=1.0, seed=0):
    rng = np.random.default_rng(seed)
    def vid(ix, iy): return (iy % Ly) * Lx + (ix % Lx)

    # vertices on rectangle [0,ax)×[0,ay)
    xs = (np.arange(Lx, dtype=float)/Lx) * ax
    ys = (np.arange(Ly, dtype=float)/Ly) * ay
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    P = np.column_stack([xx.ravel(), yy.ravel()])
    if jitter>0:
        dx = (ax/Lx)*jitter; dy=(ay/Ly)*jitter
        P += rng.uniform(-1,1,size=P.shape)*np.array([dx,dy])

    # faces (two triangles per cell)
    faces=[]
    for iy in range(Ly):
        for ix in range(Lx):
            a=vid(ix,iy); b=vid(ix+1,iy); c=vid(ix,iy+1); d=vid(ix+1,iy+1)
            faces.append([a,b,d]); faces.append([a,d,c])
    F = np.array(faces, int)

    # undirected edges with index + oriented lookup for faces
    undirected={}
    E_pairs=[]
    face_edges=np.zeros((F.shape[0],3), int)
    face_signs=np.zeros((F.shape[0],3), int)

    def ekey(u,v):
        return (u,v) if u<v else (v,u)
    def edge_index_oriented(u,v):
        key=ekey(u,v)
        ei = undirected.get(key)
        if ei is None:
            ei=len(E_pairs); undirected[key]=ei; E_pairs.append(key)
        a,b=key
        sgn = +1 if (u==a and v==b) else -1
        return ei, sgn

    for fi,(a,b,c) in enumerate(F):
        for k,(u,v) in enumerate([(a,b),(b,c),(c,a)]):
            ei, sgn = edge_index_oriented(u,v)
            face_edges[fi,k]=ei; face_signs[fi,k]=sgn

    E = np.array(E_pairs, int)
    V = P.shape[0]; Ecount=E.shape[0]; Fcount=F.shape[0]

    # d0: (E×V)
    r=[]; c=[]; d=[]
    for ei,(i,j) in enumerate(E):
        r+= [ei, ei]; c+= [i, j]; d+= [-1.0, +1.0]
    d0 = sp.csr_matrix((d,(r,c)), shape=(Ecount,V))

    # d1: (F×E)
    r=[]; c=[]; d=[]
    for fi in range(Fcount):
        for k in range(3):
            r.append(fi); c.append(face_edges[fi,k]); d.append(float(face_signs[fi,k]))
    d1 = sp.csr_matrix((d,(r,c)), shape=(Fcount,Ecount))

    return P, E, F, d0, d1, face_edges

# ---------- Stars ----------
def cot_angle(A,B,C):
    v1=B-A; v2=C-A
    dot=float(np.dot(v1,v2))
    nrm = math.sqrt(max(np.dot(v1,v1)*np.dot(v2,v2), 1e-30))
    c = max(min(dot/nrm,1.0),-1.0); s=math.sqrt(max(1.0-c*c,0.0))
    return 0.0 if s<1e-14 else c/s

def star1_cotan(P, E, F, face_edges):
    Ecount=E.shape[0]; Fcount=F.shape[0]
    edge_faces=[[] for _ in range(Ecount)]
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
        cot_opp[fi,0]=cotC; cot_opp[fi,1]=cotA; cot_opp[fi,2]=cotB

    vals = np.zeros(Ecount)
    for ei,(i,j) in enumerate(E):
        le = npl.norm(P[j]-P[i])
        csum=0.0
        for fi in edge_faces[ei]:
            a,b,c = F[fi]
            for k,(u,v) in enumerate([(a,b),(b,c),(c,a)]):
                if (u==i and v==j) or (u==j and v==i):
                    csum += cot_opp[fi,k]
        vals[ei] = 0.5 * max(csum,0.0) * max(le,1e-12)
    vals = np.maximum(vals, 1e-12)
    return sp.diags(vals, 0, format="csc")

def star1_circum(P, E, F, face_edges):
    # circumcentric dual length per edge: sum of adjacent dual segments
    V2E=[[] for _ in range(P.shape[0])]
    for ei,(i,j) in enumerate(E):
        V2E[i].append((ei,j)); V2E[j].append((ei,i))

    # dual length from circumcenters
    Ecount=E.shape[0]; Fcount=F.shape[0]
    # map edge->faces
    edge_faces=[[] for _ in range(Ecount)]
    for fi in range(Fcount):
        for k in range(3):
            edge_faces[face_edges[fi,k]].append(fi)

    # circumcenter helper
    def circumcenter(A,B,C):
        a = B-A; b = C-A
        ad = np.dot(a,a); bd = np.dot(b,b); det = a[0]*b[1]-a[1]*b[0]
        if abs(det)<1e-14:  # nearly degenerate
            return (A+B+C)/3.0
        cx = A[0] + ( b[1]*ad - a[1]*bd )/(2*det)
        cy = A[1] + (-b[0]*ad + a[0]*bd )/(2*det)
        return np.array([cx,cy], float)

    vals=np.zeros(Ecount)
    for ei,(i,j) in enumerate(E):
        adj=edge_faces[ei]
        if len(adj)==2:
            fi,fj = adj
            Ai,Bi,Ci = P[F[fi]]
            Aj,Bj,Cj = P[F[fj]]
            ci = circumcenter(Ai,Bi,Ci)
            cj = circumcenter(Aj,Bj,Cj)
            dual_len = npl.norm(ci-cj)
        else:
            dual_len = 0.0
        primal_len = npl.norm(P[j]-P[i])
        vals[ei] = max(dual_len,1e-12)/max(primal_len,1e-12)
    vals = np.maximum(vals,1e-12)
    return sp.diags(vals, 0, format="csc")

# ---------- Kähler–Dirac toy operator (squared form) ----------
def kd_DtD(d0, d1, star0, star1, star2, y_mass=0.0):
    """
    Very simple block that mimics (D^\dagger D) across 0,1,2-forms:
    DtD ≈ diag( d0^T * ⋆1 * d0 + y^2 I_V,  d1^T * ⋆2 * d1 + ⋆1^{-1} d0 d0^T ⋆1 + y^2 I_E,  d1 d1^T + y^2 I_F ).
    This is *not* a full KD lattice fermion, but enough to test small-spectrum robustness.
    """
    V = star0.shape[0]; E = star1.shape[0]; F = star2.shape[0]
    # scalar Laplacian piece
    L0 = d0.T @ (star1 @ d0)  # V×V
    # 1-form piece: curlcurl + graddiv
    L1 = d1.T @ (star2 @ d1) + (star1.power(-1) @ (d0 @ (d0.T @ star1)))  # E×E
    # 2-form piece
    L2 = d1 @ d1.T  # F×F

    I0 = sp.eye(V, format="csc")
    I1 = sp.eye(E, format="csc")
    I2 = sp.eye(F, format="csc")

    B0 = L0 + (y_mass**2) * I0
    B1 = L1 + (y_mass**2) * I1
    B2 = L2 + (y_mass**2) * I2

    DtD = sp.block_diag((B0,B1,B2), format="csc")
    # Numerical symmetrization for safety
    DtD = 0.5*(DtD + DtD.T)
    return DtD

# ---------- robust smallest eigenpairs of SPD ----------
def smallest_eigs_spd(A, k, ridge=1e-12, maxiter=200000, tol=1e-9, seed=0):
    n = A.shape[0]
    A = 0.5*(A + A.T)  # enforce symmetry
    # Try shift-invert (sigma=0) on slightly ridged matrix
    try:
        w, V = spla.eigsh(A + ridge*sp.eye(n,format="csc"),
                          k=min(k, n-2), which="LM", sigma=0.0,
                          maxiter=maxiter, tol=tol)
        w = np.maximum(w - ridge, 0.0)
        return np.sort(w), True, "eigsh_shiftinvert"
    except Exception:
        pass
    # Try plain SM with more iters
    try:
        w, V = spla.eigsh(A, k=min(k, n-2), which="SM",
                          maxiter=maxiter, tol=tol)
        return np.sort(w), True, "eigsh_SM"
    except Exception:
        pass
    # Fallback: LOBPCG with diagonal preconditioner
    rng = np.random.default_rng(seed)
    X0 = rng.normal(size=(n, min(k, max(1, n//50))))
    Mdiag = A.diagonal()
    Mdiag = np.where(Mdiag>0, 1.0/Mdiag, 1.0)
    M = sp.diags(Mdiag,0,format="csc")
    try:
        w, V = spla.lobpcg(A, X0, M=M, tol=tol, maxiter=2000, largest=False)
        w = np.sort(np.maximum(w, 0.0))[:k]
        return w, True, "lobpcg"
    except Exception:
        # Last resort: dense for tiny matrices
        if n <= 4096:
            W = npl.eigvalsh(A.toarray())
            return np.sort(np.maximum(W,0.0))[:k], True, "dense_eigh"
        return np.array([]), False, "failed"

# ---------- run one geometry ----------
def run_one(Lx, Ly, jitter, ax, ay, star_kind, y, v, k_low, seed):
    P,E,F,d0,d1,fe = build_periodic_grid(Lx,Ly,jitter,ax,ay,seed=seed)
    # Hodge stars: simple areas for 0- and 2-forms
    V = P.shape[0]; Fcount = F.shape[0]
    # vertex lumped area (1/3 of incident faces)
    A0 = np.zeros(V)
    for (a,b,c) in F:
        A = tri_area(P[a],P[b],P[c])
        A0[a]+=A/3; A0[b]+=A/3; A0[c]+=A/3
    star0 = sp.diags(np.maximum(A0,1e-12), 0, format="csc")

    A2 = np.array([tri_area(P[a],P[b],P[c]) for (a,b,c) in F])
    star2 = sp.diags(np.maximum(A2,1e-12), 0, format="csc")

    if star_kind=="cotan":
        star1 = star1_cotan(P,E,F,fe)
    else:
        star1 = star1_circum(P,E,F,fe)

    mY = y * v / math.sqrt(2.0)  # Yukawa mass scale (toy)
    DtD = kd_DtD(d0,d1,star0,star1,star2, y_mass=mY)

    # get k_low smallest eigenvalues
    w, ok, method = smallest_eigs_spd(DtD, k_low, ridge=1e-12, seed=seed)
    if not ok or w.size==0:
        raise RuntimeError("Eigen solver failed to converge.")

    # interpret sqrt(w) as “singular values”/modes of D
    svals = np.sqrt(np.maximum(w,0.0))
    info = dict(Lx=Lx,Ly=Ly,jitter=jitter,ax=ax,ay=ay,star=star_kind,
                y=y,v=v, mY=mY, method=method)
    return info, svals

# ---------- driver ----------
def main():
    ap = argparse.ArgumentParser(description="Kähler–Dirac toy spectrum with robust solver")
    ap.add_argument("--Lx", type=int, default=20)
    ap.add_argument("--Ly", type=int, default=20)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--ax", type=float, default=2.59)
    ap.add_argument("--ay", type=float, default=0.78)
    ap.add_argument("--star", type=str, choices=["cotan","circum"], default="cotan")
    ap.add_argument("--y", type=float, default=0.5, help="Yukawa coupling (toy)")
    ap.add_argument("--v", type=float, default=246.0)
    ap.add_argument("--n", type=int, default=16, help="# meshes (seeds)")
    ap.add_argument("--k", type=int, default=16, help="# smallest eigenvalues")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", type=str, default="out_kd_free")
    args = ap.parse_args()

    outdir = Path(args.out); ensure_dir(outdir)
    rows=[]; spectra=[]
    for t in range(args.n):
        seed = args.seed + 97*t
        info, svals = run_one(args.Lx,args.Ly,args.jitter,
                              args.ax,args.ay,args.star,
                              y=args.y, v=args.v, k_low=args.k, seed=seed)
        rows.append([seed] + list(svals[:args.k]))
        spectra.append(svals)

    spectra = np.array(spectra, float)
    means = np.nanmean(spectra, axis=0)
    stds  = np.nanstd(spectra, axis=0)

    # Print quick summary
    print("=== KD toy spectrum (robust) ===")
    print(f"Lx={args.Lx} Ly={args.Ly} jitter={args.jitter}  n={args.n}")
    print(f"(ax,ay)=({args.ax},{args.ay})  star={args.star}  y={args.y}  v={args.v}")
    print(f"solver: mixed (shift-invert/LOBPCG fallback)")
    for i,(m,s) in enumerate(zip(means,stds), start=1):
        print(f"sigma[{i:02d}] = {m:.6g} ± {s:.6g}")
    # save
    hdr = ["seed"] + [f"s{i+1}" for i in range(args.k)]
    save_csv(outdir/"kd_spectra.csv", hdr, rows)
    write_text(outdir/"summary.txt",
               "=== KD toy spectrum (robust) ===\n"
               f"Lx={args.Lx} Ly={args.Ly} jitter={args.jitter}  n={args.n}\n"
               f"(ax,ay)=({args.ax},{args.ay})  star={args.star}  y={args.y}  v={args.v}\n"
               f"Means: {means.tolist()}\nStds:  {stds.tolist()}\n")

if __name__ == "__main__":
    main()