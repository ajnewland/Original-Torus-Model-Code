import argparse, math
from pathlib import Path
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ---------------- I/O ----------------
def parse_shifts(s):
    out=[]
    for tok in s.split(";"):
        tok=tok.strip()
        if not tok: continue
        a,b = tok.split(",")
        out.append((int(a), int(b)))
    return out

# ---------------- Periodic grid + faces/edges ----------------
def build_periodic_grid(Lx, Ly, jitter, ax=1.0, ay=1.0, seed=0):
    rng = np.random.default_rng(seed)
    def vid(ix, iy): return (iy % Ly) * Lx + (ix % Lx)

    # integer grid coords
    ix = np.arange(Lx); iy = np.arange(Ly)
    XX, YY = np.meshgrid(ix, iy, indexing="xy")
    v_ix = XX.ravel(); v_iy = YY.ravel()

    # geometric coords
    X = (v_ix.astype(float) / Lx) * ax
    Y = (v_iy.astype(float) / Ly) * ay
    P = np.column_stack([X, Y])
    if jitter > 0:
        dx = (ax / Lx) * jitter; dy = (ay / Ly) * jitter
        P += rng.uniform(-1, 1, size=P.shape) * np.array([dx, dy])

    # faces
    faces=[]
    for jy in range(Ly):
        for jx in range(Lx):
            a = vid(jx, jy)
            b = vid(jx+1, jy)
            c = vid(jx, jy+1)
            d = vid(jx+1, jy+1)
            faces.append([a,b,d])
            faces.append([a,d,c])
    F = np.array(faces, dtype=int)

    # undirected edges and per-face edge indices+signs
    def undirected_key(u,v): return (u,v) if u<v else (v,u)
    undirected={}
    E_pairs=[]
    face_edges = np.zeros((F.shape[0],3), dtype=int)
    face_signs = np.zeros((F.shape[0],3), dtype=int)

    def edge_index_oriented(u,v):
        key = undirected_key(u,v)
        ei = undirected.get(key, None)
        if ei is None:
            ei = len(E_pairs)
            undirected[key] = ei
            E_pairs.append(key)
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

    # d0 (E×V)
    rows,cols,data=[],[],[]
    for ei,(i,j) in enumerate(E):
        rows += [ei,ei]; cols += [i,j]; data += [-1.0,+1.0]
    d0 = sp.csr_matrix((data,(rows,cols)), shape=(Ecount,V))

    return P,E,F,d0,face_edges,face_signs,v_ix,v_iy,ax,ay

# ---------------- Helpers: periodic unwrap and circumcenters ----------------
def minimal_image_vec(d, L):
    # shift d by integer multiples of L to land in [-L/2, L/2]
    return d - L*np.round(d/L)

def unwrap_triangle(P, tri, ax, ay):
    a,b,c = tri
    A = P[a].copy()
    B = P[b].copy()
    C = P[c].copy()
    # move B and C near A in periodic cell
    dB = B - A; dB[0] = minimal_image_vec(dB[0], ax); dB[1] = minimal_image_vec(dB[1], ay); B = A + dB
    dC = C - A; dC[0] = minimal_image_vec(dC[0], ax); dC[1] = minimal_image_vec(dC[1], ay); C = A + dC
    return A,B,C

def circumcenter2D(A,B,C):
    # robust circumcenter in 2D
    a = B - A; b = C - A
    adot = np.dot(a,a); bdot = np.dot(b,b)
    cross = a[0]*b[1] - a[1]*b[0]
    denom = 2.0*cross
    if abs(denom) < 1e-16:
        return (A+B+C)/3.0
    u = np.array([ b[1]*adot - a[1]*bdot, -b[0]*adot + a[0]*bdot ]) / denom
    return A + u

# ---------------- ⋆₁: cotan and circumcentric ----------------
def cot_angle(A,B,C):
    v1=B-A; v2=C-A
    dot=float(np.dot(v1,v2))
    nrm=np.linalg.norm(v1)*np.linalg.norm(v2)+1e-30
    c = np.clip(dot/nrm, -1.0, 1.0)
    s = math.sqrt(max(1.0-c*c, 0.0))
    return 0.0 if s<1e-14 else c/s

def star1_cotan(P,E,F,face_edges):
    Ecount=E.shape[0]; Fcount=F.shape[0]
    edge_faces=[[] for _ in range(Ecount)]
    for fi in range(Fcount):
        for k in range(3):
            edge_faces[face_edges[fi,k]].append(fi)
    face_pts = P[F]
    cot_opp = np.zeros((Fcount,3))
    for fi in range(Fcount):
        A,B,C = face_pts[fi]
        cotA = cot_angle(A,B,C)
        cotB = cot_angle(B,C,A)
        cotC = cot_angle(C,A,B)
        cot_opp[fi,0] = cotC
        cot_opp[fi,1] = cotA
        cot_opp[fi,2] = cotB
    vals=np.zeros(Ecount)
    for ei,(i,j) in enumerate(E):
        le = np.linalg.norm(P[j]-P[i]) + 1e-30
        s=0.0
        for fi in edge_faces[ei]:
            a,b,c = F[fi]
            for k,(u,v) in enumerate([(a,b),(b,c),(c,a)]):
                if (u==i and v==j) or (u==j and v==i):
                    s += cot_opp[fi,k]
        vals[ei] = 0.5*max(s,0.0)*le
    vals = np.maximum(vals, 1e-12)
    return sp.diags(vals, 0, format="csc")

def star1_circum(P,E,F,face_edges,ax,ay):
    # dual length over primal length
    Ecount=E.shape[0]; Fcount=F.shape[0]
    # edge -> adjacent faces
    adj=[[] for _ in range(Ecount)]
    for fi in range(Fcount):
        for k in range(3):
            ei = face_edges[fi,k]
            adj[ei].append(fi)
    # circumcenters (unwrap faces first)
    CC = np.zeros((Fcount,2))
    for fi in range(Fcount):
        A,B,C = unwrap_triangle(P, F[fi], ax, ay)
        CC[fi] = circumcenter2D(A,B,C)
    vals=np.zeros(Ecount)
    for ei,(i,j) in enumerate(E):
        le = np.linalg.norm(P[j]-P[i]) + 1e-30
        fs = adj[ei]
        if len(fs)==2:
            c1=CC[fs[0]]; c2=CC[fs[1]]
            # minimal image between circumcenters
            dc = c2 - c1
            dc[0] = minimal_image_vec(dc[0], ax)
            dc[1] = minimal_image_vec(dc[1], ay)
            Ld = np.linalg.norm(dc)
        else:
            # should not happen on torus grid, but guard
            Ld = le
        vals[ei] = max(Ld/le, 1e-12)
    return sp.diags(vals, 0, format="csc")

# ---------------- Exact fundamental loops with shift ----------------
def build_exact_cycles_shifted(Lx,Ly,E,shift_x,shift_y):
    def vid(ix,iy): return (iy%Ly)*Lx + (ix%Lx)
    undirected = {(min(i,j),max(i,j)): ei for ei,(i,j) in enumerate(E)}
    def edge_index_oriented(i,j):
        a,b = (i,j) if i<j else (j,i)
        ei = undirected.get((a,b), None)
        if ei is None: return None,0
        sgn = +1 if (i<j) else -1
        return ei, sgn
    C = np.zeros((2, E.shape[0]), float)
    y0 = shift_y % Ly
    for x in range(Lx):
        u=vid(x, y0); v=vid(x+1, y0)
        ei,sgn=edge_index_oriented(u,v)
        if ei is not None: C[0,ei]+=sgn
    x0 = shift_x % Lx
    for y in range(Ly):
        u=vid(x0, y); v=vid(x0, y+1)
        ei,sgn=edge_index_oriented(u,v)
        if ei is not None: C[1,ei]+=sgn
    if np.linalg.matrix_rank(C) < 2:
        raise RuntimeError("Cycle matrix lost rank")
    return C  # dense 2×E

# ---------------- Harmonic-basis invariant ----------------
def invsqrt_2x2(M):
    # symmetric inverse square root via eigendecomp
    w,V = npl.eigh(M)
    w = np.maximum(w, 1e-18)
    Dm12 = np.diag(1.0/np.sqrt(w))
    return V @ Dm12 @ V.T

def sin2_harmonic_invariant(P,E,F,face_edges, ax,ay,
                            star_ref_kind="cotan", star_mix_kind="circum",
                            shift=(0,0), ridge=1e-11):
    # stars
    if star_ref_kind=="cotan":
        Aref = star1_cotan(P,E,F,face_edges)
    else:
        Aref = star1_circum(P,E,F,face_edges,ax,ay)
    if star_mix_kind=="cotan":
        Amix = star1_cotan(P,E,F,face_edges)
    else:
        Amix = star1_circum(P,E,F,face_edges,ax,ay)

    aref = Aref.diagonal()   # (E,)
    inv_aref = 1.0 / np.maximum(aref, ridge)

    C = build_exact_cycles_shifted(Lx= int(round(np.sqrt(P.shape[0]))),
                                   Ly= int(round(np.sqrt(P.shape[0]))),
                                   E=E, shift_x=shift[0], shift_y=shift[1])  # 2×E
    Ct = C.T  # (E×2)

    # Gref^{-1} = C A^{-1} C^T  -> compute safely with elementwise scaling
    # A^{-1} C^T == (Ct * inv_aref[:,None])
    X = Ct * inv_aref[:,None]              # (E×2)
    Ginv = C @ X                           # (2×2)
    Ginv = 0.5*(Ginv + Ginv.T)

    # Whitening to make Omega^T Aref Omega = I
    W = invsqrt_2x2(Ginv)                  # (Ginv^{-1/2})
    Omega = X @ W                          # (E×2)

    # K = Omega^T A_mix Omega
    # Since A_mix is diagonal, A_mix @ Omega == (Omega * diag)
    amix = Amix.diagonal()
    K = (Omega * amix[:,None]).T @ Omega   # (2×2) dense
    K = 0.5*(K + K.T)

    # eigenvalues and sin^2
    w,_ = npl.eigh(K)
    w = np.maximum(w, 1e-18)
    s2 = float(w[0] / (w[0]+w[1]))
    return s2

# ---------------- Batch driver ----------------
def one_setting(L, jitter, n, ax, ay, star_ref, star_mix, shift, seed0=2025):
    vals=[]
    for k in range(n):
        P,E,F,d0,fe,fs,vix,viy,AX,AY = build_periodic_grid(L, L, jitter, ax=ax, ay=ay, seed=seed0+97*k)
        try:
            s2 = sin2_harmonic_invariant(P,E,F,fe, AX,AY,
                                         star_ref_kind=star_ref, star_mix_kind=star_mix,
                                         shift=shift, ridge=1e-11)
            vals.append(s2)
        except Exception:
            # count failure as NaN, skip
            pass
    if len(vals)==0:
        return np.nan, np.nan, 0
    arr = np.array(vals, float)
    return float(arr.mean()), float(arr.std(ddof=0)), len(vals)

def main():
    ap = argparse.ArgumentParser(description="EW cycle-swap test (harmonic-basis invariant)")
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ax", type=float, default=2.57)
    ap.add_argument("--ay", type=float, default=0.78)
    ap.add_argument("--shifts", type=str, default="0,0;1,0;0,1;3,2;5,5")
    ap.add_argument("--star_ref", choices=["cotan","circum"], default="cotan")
    ap.add_argument("--star_mix", choices=["cotan","circum"], default="circum")
    ap.add_argument("--ridge", type=float, default=1e-11)
    args = ap.parse_args()

    shifts = parse_shifts(args.shifts)
    print("=== EW cycle-swap (harmonic-basis invariant) ===")
    print(f"L={args.L} jitter={args.jitter} n={args.n}  (ax,ay)=({args.ax},{args.ay})")
    print(f"stars: ref={args.star_ref}  mix={args.star_mix}  ridge={args.ridge:.1e}")
    print(f"shifts={shifts}")

    for sh in shifts:
        m,s,ok = one_setting(args.L, args.jitter, args.n, args.ax, args.ay,
                             args.star_ref, args.star_mix, sh)
        if ok==0 or not np.isfinite(m):
            print(f"shift=(x={sh[0]},y={sh[1]})  sin2=nan ± nan  (ok={ok}/{args.n})")
        else:
            print(f"shift=(x={sh[0]},y={sh[1]})  sin2={m:.6f} ± {s:.6f}  (ok={ok}/{args.n})")

if __name__ == "__main__":
    main()