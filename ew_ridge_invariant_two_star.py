# ew_ridge_invariant_two_star.py
# Ridge sweep with basis-invariant sin^2 via generalized eigenproblem K u = λ G u,
# using TWO stars: A_ref for periods (build Ω and G), S_mix for kinetics (build K).

import argparse, math
from pathlib import Path
import numpy as np, numpy.linalg as npl
import scipy.sparse as sp, scipy.sparse.linalg as spla

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def savecsv(p, hdr, rows):
    a=np.array(rows,float); np.savetxt(p,a,delimiter=",",fmt="%.10g",
                                       header=",".join(hdr),comments="",encoding="utf-8")

# ---------- periodic triangulated grid ----------
def build_grid(L, jitter, ax, ay, seed):
    Lx=Ly=L
    rng=np.random.default_rng(seed)
    def vid(ix,iy): return (iy%Ly)*Lx+(ix%Lx)
    xs=(np.arange(Lx)/Lx)*ax; ys=(np.arange(Ly)/Ly)*ay
    XX,YY=np.meshgrid(xs,ys,indexing="xy")
    P=np.column_stack([XX.ravel(), YY.ravel()])
    if jitter>0:
        P += rng.uniform(-1,1,size=P.shape)*np.array([ax/Lx*jitter, ay/Ly*jitter])
    faces=[]
    for iy in range(Ly):
        for ix in range(Lx):
            a=vid(ix,iy); b=vid(ix+1,iy); c=vid(ix,iy+1); d=vid(ix+1,iy+1)
            faces.append([a,b,d]); faces.append([a,d,c])
    F=np.array(faces,int)
    def key(u,v): return (u,v) if u<v else (v,u)
    und={}; E_pairs=[]; face_edges=np.zeros((F.shape[0],3),int); face_signs=np.zeros_like(face_edges)
    def edge_idx_oriented(u,v):
        k=key(u,v); ei=und.get(k)
        if ei is None: ei=len(E_pairs); und[k]=ei; E_pairs.append(k)
        a,b=k; sgn=+1 if (u==a and v==b) else -1
        return ei,sgn
    for fi,(a,b,c) in enumerate(F):
        for k,(u,v) in enumerate([(a,b),(b,c),(c,a)]):
            ei,sgn=edge_idx_oriented(u,v); face_edges[fi,k]=ei; face_signs[fi,k]=sgn
    E=np.array(E_pairs,int); V=P.shape[0]; Ecount=E.shape[0]
    # d0
    rows,cols,data=[],[],[]
    for ei,(i,j) in enumerate(E): rows += [ei,ei]; cols += [i,j]; data += [-1.0,+1.0]
    d0=sp.csr_matrix((data,(rows,cols)), shape=(Ecount,V))
    return P,E,F,d0,face_edges

# ---------- stars ----------
def cot_angle(A,B,C):
    v1=B-A; v2=C-A
    dot=float(np.dot(v1,v2)); nrm=math.sqrt(max(np.dot(v1,v1)*np.dot(v2,v2),1e-30))
    c=max(min(dot/nrm,1.0),-1.0); s=math.sqrt(max(1.0-c*c,0.0))
    return 0.0 if s<1e-14 else c/s

def star1_cotan(P,E,F,FE):
    Ecount=len(E); Fcount=len(F)
    edge_faces=[[] for _ in range(Ecount)]
    for fi in range(Fcount):
        for k in range(3): edge_faces[FE[fi,k]].append(fi)
    face_pts=P[F]; cot_opp=np.zeros((Fcount,3))
    for fi in range(Fcount):
        A,B,C=face_pts[fi]
        cotA=cot_angle(A,B,C); cotB=cot_angle(B,C,A); cotC=cot_angle(C,A,B)
        cot_opp[fi,0]=cotC; cot_opp[fi,1]=cotA; cot_opp[fi,2]=cotB
    vals=np.zeros(Ecount)
    for ei,(i,j) in enumerate(E):
        le=np.linalg.norm(P[j]-P[i]); csum=0.0
        for fi in edge_faces[ei]:
            a,b,c=F[fi]
            for k,(u,v) in enumerate([(a,b),(b,c),(c,a)]):
                if (u==i and v==j) or (u==j and v==i): csum+=cot_opp[fi,k]
        vals[ei]=0.5*max(csum,0.0)*max(le,1e-12)
    vals=np.maximum(vals,1e-12); return sp.diags(vals,0,format="csc")

def circumcenter(p,q,r):
    a=q-p; b=r-p; adot=np.dot(a,a); bdot=np.dot(b,b); cross=a[0]*b[1]-a[1]*b[0]
    denom = 2.0*cross if abs(cross)>1e-18 else 2e-18*np.sign(cross if cross!=0 else 1.0)
    ux=(bdot*a[1]-adot*b[1])/denom; uy=(adot*b[0]-bdot*a[0])/denom
    return np.array([p[0]+ux,p[1]+uy],float)

def star1_circum(P,E,F,FE):
    Ecount=len(E); Fcount=len(F)
    edge_faces=[[] for _ in range(Ecount)]
    for fi in range(Fcount):
        for k in range(3): edge_faces[FE[fi,k]].append(fi)
    Cc=np.zeros((Fcount,2))
    for fi,(i,j,k) in enumerate(F): Cc[fi]=circumcenter(P[i],P[j],P[k])
    vals=np.zeros(Ecount)
    for ei,(i,j) in enumerate(E):
        faces=edge_faces[ei]
        if len(faces)==2: Ld=np.linalg.norm(Cc[faces[1]]-Cc[faces[0]])
        else: Ld=np.linalg.norm(P[j]-P[i])
        Lp=np.linalg.norm(P[j]-P[i]); vals[ei]=max(Ld,1e-12)/max(Lp,1e-12)
    vals=np.maximum(vals,1e-12); return sp.diags(vals,0,format="csc")

# ---------- exact cycles (rank-2) ----------
def exact_cycles(L,E,shift_x=0,shift_y=0):
    Lx=Ly=L
    def vid(ix,iy): return (iy%Ly)*Lx + (ix%Lx)
    und={(min(i,j),max(i,j)): ei for ei,(i,j) in enumerate(E)}
    def edge_idx_oriented(i,j):
        a,b=(i,j) if i<j else (j,i); ei=und.get((a,b));
        if ei is None: return None,0
        return ei,(+1 if i<j else -1)
    C=np.zeros((2,len(E)),float)
    y0=(0+shift_y)%Ly
    for x in range(Lx):
        u=vid(x,y0); v=vid(x+1,y0); ei,sgn=edge_idx_oriented(u,v);
        if ei is not None: C[0,ei]+=sgn
    x0=(0+shift_x)%Lx
    for y in range(Ly):
        u=vid(x0,y); v=vid(x0,y+1); ei,sgn=edge_idx_oriented(u,v);
        if ei is not None: C[1,ei]+=sgn
    if np.linalg.matrix_rank(C)<2: raise RuntimeError("C rank < 2")
    return sp.csr_matrix(C)

# ---------- invariant angle with two stars ----------
def sin2_two_star(L,jitter,n,ax,ay,star_ref="cotan",star_mix="circum",ridge=1e-12,seed0=2025):
    vals=[]
    for k in range(n):
        s=seed0+97*k
        P,E,F,d0,FE=build_grid(L,jitter,ax,ay,s)
        A = star1_cotan(P,E,F,FE) if star_ref=="cotan" else star1_circum(P,E,F,FE)
        S = star1_cotan(P,E,F,FE) if star_mix=="cotan" else star1_circum(P,E,F,FE)
        C = exact_cycles(L,E,0,0)
        # Ω from A_ref with period constraints; pin one vertex via dropping row in B
        B=(d0.T @ A).tocsr(); B=B[1:,:]
        Zvv=sp.csr_matrix((B.shape[0],B.shape[0])); Z2=sp.csr_matrix((2,2))
        KKT=sp.bmat([[A+ridge*sp.eye(A.shape[0],format="csc"), B.T, C.T],
                     [B,                                         Zvv, Z2],
                     [C,                                         Z2,  Z2]], format="csc")
        RHS=np.zeros((A.shape[0]+B.shape[0]+2,2)); RHS[-2:,:]=np.eye(2)
        sol0=spla.spsolve(KKT,RHS[:,0]); sol1=spla.spsolve(KKT,RHS[:,1])
        Omega=np.column_stack([sol0[:A.shape[0]], sol1[:A.shape[0]]])  # E×2
        # G from A_ref
        X=spla.spsolve(A+ridge*sp.eye(A.shape[0],format="csc"), C.T)    # E×2
        Ginv=C @ X; Ginv=0.5*(Ginv+Ginv.T)
        det=Ginv[0,0]*Ginv[1,1]-Ginv[0,1]*Ginv[1,0]
        if not np.isfinite(det) or abs(det)<1e-18: continue
        G=(1.0/det)*np.array([[Ginv[1,1],-Ginv[0,1]],[-Ginv[1,0],Ginv[0,0]]],float)
        # K from S_mix
        K = Omega.T @ (S @ Omega)
        # generalized eigen
        w = np.sort(np.real(npl.eigvals(npl.solve(G, K))))
        if w[0]<=0 or w[1]<=0: continue
        vals.append(w[0]/(w[0]+w[1]))
    if len(vals)==0: return np.nan, np.nan
    return float(np.mean(vals)), float(np.std(vals))

def main():
    ap=argparse.ArgumentParser(description="Ridge sweep with two-star invariant sin^2")
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ax", type=float, default=2.57)
    ap.add_argument("--ay", type=float, default=0.78)
    ap.add_argument("--star_ref", choices=["cotan","circum"], default="cotan")
    ap.add_argument("--star_mix", choices=["cotan","circum"], default="circum")
    ap.add_argument("--out", type=str, default="out_ridge_two_star")
    args=ap.parse_args()

    print("=== Ridge sweep (two-star invariant) ===")
    print(f"L={args.L} jitter={args.jitter} n={args.n}  (ax,ay)=({args.ax},{args.ay})")
    print(f"stars: ref={args.star_ref}  mix={args.star_mix}")

    ridges=np.logspace(-14,-9,10)
    rows=[]
    for r in ridges:
        m,s = sin2_two_star(args.L,args.jitter,args.n,args.ax,args.ay,
                            star_ref=args.star_ref, star_mix=args.star_mix,
                            ridge=r, seed0=2025)
        print(f"ridge={r:>.1e}  sin2={m:.6f} ± {s:.6f}")
        rows.append([r,m,s])
    ensure_dir(args.out)
    savecsv(Path(args.out,"ridge_two_star.csv"), ["ridge","sin2_mean","sin2_std"], rows)

if __name__=="__main__":
    main()