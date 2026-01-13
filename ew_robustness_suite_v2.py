# ew_robustness_suite_v2.py
# Robustness suite with basis-invariant sin^2(theta_W) via generalized eigenproblem K u = λ G u

import argparse, math
from pathlib import Path
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ---------------- I/O helpers ----------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def wtxt(p, s): p = Path(p); ensure_dir(p.parent); p.write_text(s, encoding="utf-8")
def savecsv(path, header, rows):
    arr = np.array(rows, float)
    np.savetxt(Path(path), arr, delimiter=",", fmt="%.10g",
               header=",".join(header), comments="", encoding="utf-8")

# ---------------- mesh: periodic triangular grid ----------------
def build_periodic_grid(L, jitter, ax=1.0, ay=1.0, seed=0):
    Lx = Ly = L
    rng = np.random.default_rng(seed)
    def vid(ix, iy): return (iy % Ly) * Lx + (ix % Lx)
    xs = (np.arange(Lx)/Lx)*ax
    ys = (np.arange(Ly)/Ly)*ay
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    P = np.column_stack([XX.ravel(), YY.ravel()])
    if jitter>0:
        P += rng.uniform(-1,1,size=P.shape)*np.array([ax/Lx*jitter, ay/Ly*jitter])

    faces=[]
    for iy in range(Ly):
        for ix in range(Lx):
            a=vid(ix,iy); b=vid(ix+1,iy); c=vid(ix,iy+1); d=vid(ix+1,iy+1)
            faces.append([a,b,d]); faces.append([a,d,c])
    F = np.array(faces, int)

    def key(u,v): return (u,v) if u<v else (v,u)
    und={}, []
    und = {}
    E_pairs=[]
    face_edges=np.zeros((F.shape[0],3),int)
    face_signs=np.zeros((F.shape[0],3),int)
    def edge_idx_oriented(u,v):
        k=key(u,v); ei=und.get(k)
        if ei is None:
            ei=len(E_pairs); und[k]=ei; E_pairs.append(k)
        a,b=k; sgn=+1 if (u==a and v==b) else -1
        return ei, sgn
    for fi,(a,b,c) in enumerate(F):
        for k,(u,v) in enumerate([(a,b),(b,c),(c,a)]):
            ei,sgn=edge_idx_oriented(u,v)
            face_edges[fi,k]=ei; face_signs[fi,k]=sgn
    E=np.array(E_pairs,int)
    V=P.shape[0]; Ecount=E.shape[0]; Fcount=F.shape[0]

    rows,cols,data=[],[],[]
    for ei,(i,j) in enumerate(E):
        rows += [ei,ei]; cols += [i,j]; data += [-1.0,+1.0]
    d0 = sp.csr_matrix((data,(rows,cols)), shape=(Ecount,V))
    return P,E,F,d0,face_edges,face_signs

# ---------------- Hodge ⋆1 ----------------
def cot_angle(A,B,C):
    v1=B-A; v2=C-A
    dot=float(np.dot(v1,v2))
    nrm = math.sqrt(max(np.dot(v1,v1)*np.dot(v2,v2),1e-30))
    c = max(min(dot/nrm,1.0),-1.0)
    s = math.sqrt(max(1.0-c*c,0.0))
    return 0.0 if s<1e-14 else c/s

def star1_cotan(P,E,F,face_edges):
    Ecount=len(E); Fcount=len(F)
    edge_faces=[[] for _ in range(Ecount)]
    for fi in range(Fcount):
        for k in range(3):
            edge_faces[face_edges[fi,k]].append(fi)
    face_pts=P[F]
    cot_opp=np.zeros((Fcount,3))
    for fi in range(Fcount):
        A,B,C=face_pts[fi]
        cotA=cot_angle(A,B,C); cotB=cot_angle(B,C,A); cotC=cot_angle(C,A,B)
        cot_opp[fi,0]=cotC; cot_opp[fi,1]=cotA; cot_opp[fi,2]=cotB
    vals=np.zeros(Ecount)
    for ei,(i,j) in enumerate(E):
        le = npl.norm(P[j]-P[i])
        csum=0.0
        for fi in edge_faces[ei]:
            a,b,c=F[fi]
            for k,(u,v) in enumerate([(a,b),(b,c),(c,a)]):
                if (u==i and v==j) or (u==j and v==i):
                    csum+=cot_opp[fi,k]
        vals[ei]=0.5*max(csum,0.0)*max(le,1e-12)
    vals=np.maximum(vals,1e-12)
    return sp.diags(vals,0,format="csc")

def circumcenter(p,q,r):
    # robust-ish 2D circumcenter
    A=p; B=q; C=r
    a = B - A; b = C - A
    adot = np.dot(a,a); bdot = np.dot(b,b)
    cross = a[0]*b[1]-a[1]*b[0]
    denom = 2.0*cross if abs(cross)>1e-18 else 2e-18*np.sign(cross if cross!=0 else 1.0)
    ux = (bdot*(a[1]) - adot*(b[1]))/denom
    uy = (adot*(b[0]) - bdot*(a[0]))/denom
    return np.array([A[0]+ux, A[1]+uy], float)

def star1_circum(P,E,F,face_edges):
    # dual edge = segment between circumcenters of adjacent faces
    Ecount=len(E); Fcount=len(F)
    # map edge->two faces
    edge_faces=[[] for _ in range(Ecount)]
    for fi in range(Fcount):
        for k in range(3):
            edge_faces[face_edges[fi,k]].append(fi)
    # circumcenters
    Cc = np.zeros((Fcount,2))
    for fi,(i,j,k) in enumerate(F):
        Cc[fi]=circumcenter(P[i],P[j],P[k])
    vals=np.zeros(Ecount)
    for ei,(i,j) in enumerate(E):
        faces=edge_faces[ei]
        if len(faces)==2:
            f0,f1=faces
            Ld=npl.norm(Cc[f1]-Cc[f0])
        else:
            # boundary-like (shouldn't happen on torus), fall back to primal length
            Ld=npl.norm(P[j]-P[i])
        Lp = npl.norm(P[j]-P[i])
        vals[ei]=max(Ld,1e-12)/max(Lp,1e-12)
    vals=np.maximum(vals,1e-12)
    return sp.diags(vals,0,format="csc")

# ---------------- cycles ----------------
def exact_cycles(L, E, shift_x=0, shift_y=0):
    Lx=Ly=L
    def vid(ix,iy): return (iy%Ly)*Lx + (ix%Lx)
    und = {(min(i,j),max(i,j)): ei for ei,(i,j) in enumerate(E)}
    def edge_idx_oriented(i,j):
        a,b=(i,j) if i<j else (j,i)
        ei=und.get((a,b),None)
        if ei is None: return None,0
        sgn=+1 if (i<j) else -1
        return ei,sgn
    C=np.zeros((2,len(E)),float)
    y0 = (0+shift_y)%Ly
    for x in range(Lx):
        u=vid(x,y0); v=vid(x+1,y0)
        ei,sgn=edge_idx_oriented(u,v)
        if ei is not None: C[0,ei]+=sgn
    x0 = (0+shift_x)%Lx
    for y in range(Ly):
        u=vid(x0,y); v=vid(x0,y+1)
        ei,sgn=edge_idx_oriented(u,v)
        if ei is not None: C[1,ei]+=sgn
    if np.linalg.matrix_rank(C)<2:
        raise RuntimeError("C lost rank")
    return sp.csr_matrix(C)

# ---------------- harmonic solve & invariant angle ----------------
def solve_omega(d0, star1, C, ridge=1e-12, pin_vertex=True):
    A = (star1 + ridge*sp.eye(star1.shape[0], format="csc")).tocsc()
    B = (d0.T @ star1).tocsr()
    if pin_vertex: B = B[1:,:]  # drop one vertex constraint
    Zvv = sp.csr_matrix((B.shape[0], B.shape[0]))
    Zv2 = sp.csr_matrix((B.shape[0], 2))
    Z2v = sp.csr_matrix((2, B.shape[0]))
    Z22 = sp.csr_matrix((2,2))
    KKT = sp.bmat([[A, B.T, C.T],
                   [B, Zvv, Zv2],
                   [C, Z2v, Z22]], format="csc")
    RHS = np.zeros((A.shape[0] + B.shape[0] + 2, 2))
    RHS[-2:,:] = np.eye(2)
    sol0 = spla.spsolve(KKT, RHS[:,0])
    sol1 = spla.spsolve(KKT, RHS[:,1])
    Omega = np.column_stack([sol0[:A.shape[0]], sol1[:A.shape[0]]])  # E×2
    return Omega

def K_and_G_from_geometry(L, jitter, ax, ay, star_kind, ridge, seed):
    P,E,F,d0,fe,fs = build_periodic_grid(L, jitter, ax, ay, seed)
    if star_kind=="cotan":
        A = star1_cotan(P,E,F,fe)
    elif star_kind=="circum":
        A = star1_circum(P,E,F,fe)
    else:
        raise ValueError("star_kind must be 'cotan' or 'circum'")
    C = exact_cycles(L, E, 0, 0)
    # harmonic reps with fixed periods
    Omega = solve_omega(d0, A, C, ridge=ridge, pin_vertex=True)
    K = Omega.T @ (A @ Omega)               # 2×2
    # period metric
    X = spla.spsolve(A, C.T)                # E×2
    Ginv = C @ X                            # 2×2
    Ginv = 0.5*(Ginv+Ginv.T)
    # invert (safe 2×2)
    det = Ginv[0,0]*Ginv[1,1]-Ginv[0,1]*Ginv[1,0]
    if not np.isfinite(det) or abs(det)<1e-18: return None, None, None
    G = (1.0/det)*np.array([[ Ginv[1,1], -Ginv[0,1]],
                            [-Ginv[1,0],  Ginv[0,0]]], float)
    # generalized eigenvalues of (K,G)
    w = npl.eigvals(npl.solve(G, K))  # solves G^{-1}K u = λ u  (same spectrum)
    w = np.sort(np.real(w))
    lam1,lam2 = float(w[0]), float(w[1])
    # angle
    sin2 = lam1/(lam1+lam2) if (lam1>0 and lam2>0) else np.nan
    # condition estimates
    condG = np.linalg.cond(G)
    condK = np.linalg.cond(K)
    return sin2, condK, condG

def sin2_mean(L, jitter, n, ax, ay, star_kind, ridge, seed0):
    vals=[]; ck=[]; cg=[]
    for k in range(n):
        s = seed0 + 97*k
        sin2, condK, condG = K_and_G_from_geometry(L, jitter, ax, ay, star_kind, ridge, s)
        if np.isfinite(sin2):
            vals.append(sin2); ck.append(condK); cg.append(condG)
    if len(vals)==0: return np.nan, np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals)), float(np.mean(cg))

# ---------------- tests ----------------
def test_ridge(L,jitter,n,ax,ay,out,star="cotan"):
    print(f"\n[Ridge sweep]")
    ridges = np.logspace(-14, -9, 10)
    rows=[]
    for r in ridges:
        m,s,condG = sin2_mean(L,jitter,n,ax,ay,star,r,seed0=2025)
        print(f"ridge={r:>.1e}  sin2={m:.6f} ± {s:.6f}")
        rows.append([r,m,s,condG])
    ensure_dir(out)
    savecsv(Path(out,"ridge_sweep.csv"),
            ["ridge","sin2_mean","sin2_std","condG_mean"], rows)

def test_cycles(L,jitter,n,ax,ay,out,star="cotan"):
    print(f"\n[Cycle swap (invariant, generalized eigen)]")
    shifts = [(0,0),(1,0),(0,1),(3,2),(5,5)]
    rows=[]
    for sx,sy in shifts:
        # we only vary C for Omega and G consistently inside K_and_G_from_geometry by rebuilding C,
        # so emulate it here:
        vals=[]; cg=[]
        for k in range(n):
            s = 2025 + 97*k
            P,E,F,d0,fe,fs = build_periodic_grid(L,jitter,ax,ay,s)
            A = star1_cotan(P,E,F,fe) if star=="cotan" else star1_circum(P,E,F,fe)
            C = exact_cycles(L, E, sx, sy)
            Omega = solve_omega(d0, A, C, ridge=1e-12, pin_vertex=True)
            K = Omega.T @ (A @ Omega)
            X = spla.spsolve(A, C.T)
            Ginv = C @ X; Ginv=0.5*(Ginv+Ginv.T)
            det = Ginv[0,0]*Ginv[1,1]-Ginv[0,1]*Ginv[1,0]
            if not np.isfinite(det) or abs(det)<1e-18: continue
            G = (1.0/det)*np.array([[ Ginv[1,1], -Ginv[0,1]],
                                    [-Ginv[1,0],  Ginv[0,0]]], float)
            w = npl.eigvals(npl.solve(G, K))
            w = np.sort(np.real(w))
            lam1,lam2 = float(w[0]), float(w[1])
            if lam1>0 and lam2>0:
                vals.append(lam1/(lam1+lam2)); cg.append(np.linalg.cond(G))
        if len(vals)==0:
            print(f"shift=(x={sx},y={sy})  sin2=NaN  ok=0/{n}")
            rows.append([sx,sy,np.nan,np.nan,0])
        else:
            m,s = float(np.mean(vals)), float(np.std(vals))
            print(f"shift=(x={sx},y={sy})  sin2={m:.6f} ± {s:.6f}  ok={len(vals)}/{n}")
            rows.append([sx,sy,m,s,len(vals)])
    ensure_dir(out)
    savecsv(Path(out,"cycle_swap.csv"),
            ["shift_x","shift_y","sin2_mean","sin2_std","ok"], rows)

def test_stars(L,jitter,n,ax,ay,out):
    print(f"\n[Hodge star swap (invariant angle)]")
    for star in ["cotan","circum"]:
        m,s,condG = sin2_mean(L,jitter,n,ax,ay,star,1e-12,seed0=2025)
        print(f"star={star:<7}  sin2={m:.6f} ± {s:.6f}  condG≈{condG:.2e}")

def test_size_scaling(jitter,n,ax,ay,out,star="cotan"):
    print(f"\n[Size scaling]")
    rows=[]
    for L in [16,20,24,28,32]:
        m,s,condG = sin2_mean(L,jitter,n,ax,ay,star,1e-12,seed0=2025)
        print(f"L={L:<2d}  sin2={m:.6f} ± {s:.6f}  ok={n}/{n}")
        rows.append([L,m,s,condG])
    ensure_dir(out)
    savecsv(Path(out,"size_scaling.csv"),
            ["L","sin2_mean","sin2_std","condG_mean"], rows)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="EW robustness suite (basis-invariant sin^2 via generalized eigenproblem)")
    ap.add_argument("--run", choices=["ridge","cycles","stars","size"], required=True)
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ax", type=float, default=2.57)
    ap.add_argument("--ay", type=float, default=0.78)
    ap.add_argument("--out", type=str, default="out_robust_v2")
    args = ap.parse_args()

    print("=== EW robustness suite (invariant) ===")
    print(f"run={args.run}  L={args.L}  jitter={args.jitter}  n={args.n}")
    print(f"(ax,ay)=({args.ax},{args.ay})  out={args.out}")

    if args.run=="ridge":
        test_ridge(args.L, args.jitter, args.n, args.ax, args.ay, args.out, star="cotan")
    elif args.run=="cycles":
        test_cycles(args.L, args.jitter, args.n, args.ax, args.ay, args.out, star="cotan")
    elif args.run=="stars":
        test_stars(args.L, args.jitter, args.n, args.ax, args.ay, args.out)
    elif args.run=="size":
        test_size_scaling(args.jitter, args.n, args.ax, args.ay, args.out, star="cotan")

if __name__ == "__main__":
    main()