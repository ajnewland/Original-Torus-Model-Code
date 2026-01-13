# ew_uneven_suite.py
# Non-even lattice robustness suite for the electroweak invariant:
#   sin^2(theta_W) from G = (C A^{-1} C^T)^{-1}
#
# Features
# - Periodic triangular mesh on (Lx x Ly) with jitter
# - Geometric unevenness via smooth warp and/or graded scaling
# - DEC operators: cotan star (default) or diagonal (length-based) fallback
# - Topological fundamental cycles (independent of geometry)
# - Invariant: sin^2 from G eigenvalues; diagnostics: cond(A), cond(G), min angles, min star
# - Modes: warp sweep, grade sweep, autosweep (ax,ay), cycles (cycle-shift stability)
#
# Usage examples (Windows cmd):
#   python ew_uneven_suite.py --mode warp --L 20 --jitter 0.02 --n 32 --ax 2.59 --ay 0.78 --warp_eps_min 0.0 --warp_eps_max 0.2 --warp_steps 6 --out out_warp
#   python ew_uneven_suite.py --mode grade --L 20 --jitter 0.02 --n 32 --ax 2.59 --ay 0.78 --grade_min 0.0 --grade_max 0.6 --grade_steps 7 --out out_grade
#   python ew_uneven_suite.py --mode autosweep --L 20 --jitter 0.02 --n 16 --ax_min 2.4 --ax_max 2.8 --ax_steps 9 --ay_min 0.72 --ay_max 0.84 --ay_steps 7 --out out_autosweep
#   python ew_uneven_suite.py --mode cycles --L 20 --jitter 0.02 --n 64 --ax 2.59 --ay 0.78 --shifts "0,0;1,0;0,1;3,2;5,5" --out out_cycles
#
# Notes
# - Circumcentric star is omitted here for brevity/robustness on uneven meshes; use cotan or diagonal.
# - All results written under --out. Plots require matplotlib; if missing, CSVs still save.

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
    path = Path(path); ensure_dir(path.parent); path.write_text(text, encoding="utf-8")

def write_csv(path, header, rows_float2d):
    arr = np.array(rows_float2d, float)
    ensure_dir(Path(path).parent)
    np.savetxt(Path(path), arr, delimiter=",", fmt="%.10g",
               header=",".join(header), comments="", encoding="utf-8")

# ---------------- Mesh construction ----------------
def build_periodic_grid(Lx, Ly, jitter, ax=1.0, ay=1.0, seed=0):
    """Periodic rectangular triangulation (two tris per cell), with jitter."""
    rng = np.random.default_rng(seed)
    def vid(ix, iy): return (iy % Ly)*Lx + (ix % Lx)

    # integer grid coords
    gx = np.arange(Lx)
    gy = np.arange(Ly)
    XX, YY = np.meshgrid(gx, gy, indexing="xy")
    v_ix = XX.ravel()
    v_iy = YY.ravel()

    # base geometric coords
    X = (v_ix.astype(float)/Lx)*ax
    Y = (v_iy.astype(float)/Ly)*ay
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
            faces.append([a,b,d])
            faces.append([a,d,c])
    F = np.array(faces, dtype=int)

    # undirected edge list with orientation helper
    def key(u,v): return (u,v) if u<v else (v,u)
    undirected = {}
    E_pairs = []
    face_edges = np.zeros((F.shape[0],3), dtype=int)
    face_signs = np.zeros((F.shape[0],3), dtype=int)

    def edge_index_oriented(u,v):
        k = key(u,v)
        ei = undirected.get(k, None)
        if ei is None:
            ei = len(E_pairs)
            undirected[k]=ei
            E_pairs.append(k)
        a,b = k
        sgn = +1 if (u==a and v==b) else -1
        return ei, sgn

    for fi,(a,b,c) in enumerate(F):
        for k,(u,v) in enumerate([(a,b),(b,c),(c,a)]):
            ei,sgn=edge_index_oriented(u,v)
            face_edges[fi,k]=ei
            face_signs[fi,k]=sgn

    E = np.array(E_pairs, dtype=int)
    V = P.shape[0]; Ecount = E.shape[0]

    # d0 (E x V)
    rows, cols, data = [], [], []
    for ei,(i,j) in enumerate(E):
        rows += [ei,ei]; cols += [i,j]; data += [-1.0, +1.0]
    d0 = sp.csr_matrix((data,(rows,cols)), shape=(Ecount, V))

    return P, E, F, d0, face_edges, face_signs, v_ix, v_iy

# ---------------- Unevenness transforms ----------------
def apply_warp(P, ax, ay, eps, mode="sinxy"):
    """Smooth geometry warp; preserves periodicity in index space."""
    if eps==0.0: return P
    x = P[:,0] / max(ax,1e-12)
    y = P[:,1] / max(ay,1e-12)
    if mode=="sinxy":
        fx = np.sin(2*np.pi*y)
        gy = np.sin(2*np.pi*x)
        Xp = P[:,0] + eps*0.20*ax*fx
        Yp = P[:,1] + eps*0.20*ay*gy
    elif mode=="shear":
        Xp = P[:,0] + eps*0.25*ax*(y-0.5)
        Yp = P[:,1] + eps*0.25*ay*(x-0.5)
    else:
        Xp, Yp = P[:,0], P[:,1]
    return np.column_stack([Xp, Yp])

def apply_grade(P, ax, ay, g, band="x"):
    """Graded scaling: stretch/compress smoothly across x or y band."""
    if g==0.0: return P
    x = P[:,0] / max(ax,1e-12)
    y = P[:,1] / max(ay,1e-12)
    if band=="x":
        s = 1.0 + g*(x-0.5)   # linear grade left->right
        Xp = (P[:,0]-0.5*ax)*s + 0.5*ax
        Yp = P[:,1]
    else:
        s = 1.0 + g*(y-0.5)
        Xp = P[:,0]
        Yp = (P[:,1]-0.5*ay)*s + 0.5*ay
    return np.column_stack([Xp, Yp])

# ---------------- Angles & DEC stars ----------------
def tri_area(A,B,C):
    v1 = B-A; v2 = C-A
    return 0.5*abs(v1[0]*v2[1] - v1[1]*v2[0])

def cot_angle(A,B,C):
    v1 = B-A; v2 = C-A
    dot = float(np.dot(v1,v2))
    nrm = math.sqrt(max(np.dot(v1,v1)*np.dot(v2,v2), 1e-30))
    c = np.clip(dot/nrm, -1.0, 1.0)
    s = math.sqrt(max(1.0-c*c,0.0))
    return 0.0 if s<1e-14 else c/s

def mesh_quality(P, F):
    """Return min triangle angle (deg) and min area."""
    mins = []
    minA = float("inf")
    for fi in range(F.shape[0]):
        a,b,c = F[fi]
        A,B,C = P[a],P[b],P[c]
        ca = cot_angle(A,B,C); cb = cot_angle(B,C,A); cc = cot_angle(C,A,B)
        # cot->angle
        def acot(u):  # angle in radians
            return math.atan2(1.0, max(u,1e-30))
        angs = [acot(ca), acot(cb), acot(cc)]
        mins.append(min(angs))
        minA = min(minA, tri_area(A,B,C))
    return (min(mins)*180.0/math.pi, minA)

def star1_cotan(P, E, F, face_edges):
    Ecount = E.shape[0]; Fcount = F.shape[0]
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
        cot_opp[fi,0] = cotC  # opp (A,B)
        cot_opp[fi,1] = cotA  # opp (B,C)
        cot_opp[fi,2] = cotB  # opp (C,A)

    vals = np.zeros(Ecount)
    for ei,(i,j) in enumerate(E):
        le = npl.norm(P[j]-P[i])
        csum=0.0
        for fi in edge_faces[ei]:
            a,b,c = F[fi]
            for k,(u,v) in enumerate([(a,b),(b,c),(c,a)]):
                if (u==i and v==j) or (u==j and v==i):
                    csum += cot_opp[fi,k]
        vals[ei] = 0.5*max(csum,0.0)*max(le,1e-12)
    vals = np.maximum(vals, 1e-12)
    return sp.diags(vals, 0, format="csc"), vals.min()

def star1_diagonal(P, E):
    """Length-based diagonal mass for 1-forms; very robust."""
    vals = np.array([npl.norm(P[j]-P[i]) for (i,j) in E], float)
    vals = np.maximum(vals, 1e-12)
    return sp.diags(vals, 0, format="csc"), float(vals.min())

# ---------------- Topological cycles (fundamental loops) ----------------
def build_cycles_exact(Lx, Ly, E):
    """Two cycles: x-loop along row y=0; y-loop along column x=0. Geometry-independent."""
    def vid(ix, iy): return (iy % Ly)*Lx + (ix % Lx)
    undirected = {(min(i,j), max(i,j)): ei for ei,(i,j) in enumerate(E)}
    def ei_s(i,j):
        a,b = (i,j) if i<j else (j,i)
        ei = undirected.get((a,b), None)
        if ei is None: return None, 0
        s = +1 if (i<j) else -1
        return ei, s

    C = np.zeros((2, E.shape[0]), float)
    y0=0
    for x in range(Lx):
        u=vid(x,y0); v=vid(x+1,y0)
        ei,s=ei_s(u,v)
        if ei is not None: C[0,ei]+=s
    x0=0
    for y in range(Ly):
        u=vid(x0,y); v=vid(x0,y+1)
        ei,s=ei_s(u,v)
        if ei is not None: C[1,ei]+=s
    if np.linalg.matrix_rank(C)<2:
        raise RuntimeError("Cycle matrix C lost rank.")
    return sp.csr_matrix(C)

def shift_cycles(C, Lx, Ly, dx, dy, E):
    """Return cycles shifted by (dx,dy) modulo the periodic tiling."""
    # Since we built exact cycles by IDs tied to (x=0) or (y=0), shifting is
    # equivalent to re-labelling which vertex row/column we traverse.
    # Implement by rebuilding new C with the chosen offsets.
    def vid(ix, iy): return (iy % Ly)*Lx + (ix % Lx)
    undirected = {(min(i,j), max(i,j)): ei for ei,(i,j) in enumerate(E)}
    def ei_s(i,j):
        a,b = (i,j) if i<j else (j,i)
        ei = undirected.get((a,b), None)
        if ei is None: return None,0
        s = +1 if (i<j) else -1
        return ei,s

    C2 = np.zeros((2, E.shape[0]), float)
    y0 = dy % Ly
    for x in range(Lx):
        u=vid(x,y0); v=vid(x+1,y0)
        ei,s=ei_s(u,v)
        if ei is not None: C2[0,ei]+=s
    x0 = dx % Lx
    for y in range(Ly):
        u=vid(x0,y); v=vid(x0,y+1)
        ei,s=ei_s(u,v)
        if ei is not None: C2[1,ei]+=s
    return sp.csr_matrix(C2)

# ---------------- Invariant & diagnostics ----------------
def sin2_from_G(star1, C, ridge=1e-11, cond_cap=1e12):
    """Compute G=(C A^{-1} C^T)^{-1} safely; return sin2, cond(G), G, cond(A)."""
    try:
        A = star1.tocsc()
        # cond(A) via diagonal bounds (cheap proxy):
        diag = A.diagonal()
        condA = float(np.max(diag)/max(np.min(diag),1e-18))

        X = spla.spsolve(A + ridge*sp.eye(A.shape[0], format="csc"), C.T)  # (E x 2)
        Ginv = (C @ X)  # 2x2
        Ginv = 0.5*(Ginv + Ginv.T)
        # invert 2x2
        det = Ginv[0,0]*Ginv[1,1] - Ginv[0,1]*Ginv[1,0]
        if not np.isfinite(det) or abs(det) < 1e-20:
            return np.nan, np.nan, None, condA
        G = (1.0/det)*np.array([[Ginv[1,1], -Ginv[0,1]],
                                 [-Ginv[1,0], Ginv[0,0]]], float)
        # eigen
        trG = G[0,0]+G[1,1]
        dG = (G[0,0]-G[1,1])**2 + 4*G[0,1]*G[1,0]
        dG = max(dG, 0.0)
        sG = math.sqrt(dG)
        lam_min = 0.5*(trG - sG)
        lam_max = 0.5*(trG + sG)
        if lam_min<=0 or lam_max<=0: return np.nan, np.nan, None, condA
        sin2 = lam_min/(lam_min+lam_max)
        condG = lam_max/lam_min
        if not np.isfinite(condG) or condG>cond_cap:
            return np.nan, np.nan, None, condA
        return float(sin2), float(condG), G, condA
    except Exception:
        return np.nan, np.nan, None, float("inf")

# ---------------- Per-mesh evaluation ----------------
def eval_one(Lx,Ly,jitter,ax,ay,star_type="cotan",ridge=1e-11, seed=2025,
             warp_eps=0.0, warp_mode="sinxy", grade=0.0, grade_band="x",
             cycle_shift=None):
    # base mesh
    P,E,F,d0,fe,fs,vix,viy = build_periodic_grid(Lx,Ly,jitter, ax=ax, ay=ay, seed=seed)
    # apply unevenness
    P = apply_warp(P, ax, ay, warp_eps, warp_mode)
    P = apply_grade(P, ax, ay, grade, grade_band)

    # DEC star
    if star_type=="cotan":
        star1, minstar = star1_cotan(P,E,F,fe)
    else:
        star1, minstar = star1_diagonal(P,E)

    # cycles
    C = build_cycles_exact(Lx,Ly,E)
    if cycle_shift is not None:
        dx,dy = cycle_shift
        C = shift_cycles(C, Lx, Ly, dx, dy, E)

    # invariant + quality
    sin2, condG, G, condA = sin2_from_G(star1, C, ridge=ridge)
    min_ang_deg, min_area = mesh_quality(P,F)

    return dict(sin2=sin2, condG=condG, condA=condA,
                min_angle=min_ang_deg, min_area=min_area, min_star=minstar)

def aggregate_over_n(Lx,Ly,jitter,ax,ay,n, star_type, ridge,
                     warp_eps, warp_mode, grade, grade_band, shifts=None, seed0=2025):
    def one(shift):
        vals=[]
        for k in range(n):
            seed = seed0 + 97*k
            r = eval_one(Lx,Ly,jitter,ax,ay,star_type,ridge,seed,
                         warp_eps,warp_mode,grade,grade_band, cycle_shift=shift)
            if np.isfinite(r["sin2"]): vals.append(r)
        if len(vals)==0:
            return dict(ok=0, sin2=np.nan, condG=np.nan, condA=np.nan,
                        min_angle=np.nan, min_area=np.nan, min_star=np.nan)
        def mean(key): return float(np.mean([v[key] for v in vals]))
        def std(key):  return float(np.std([v[key] for v in vals]))
        return dict(ok=len(vals), sin2=mean("sin2"), sin2_std=std("sin2"),
                    condG=mean("condG"), condA=mean("condA"),
                    min_angle=mean("min_angle"), min_area=mean("min_area"),
                    min_star=mean("min_star"))
    if shifts is None:
        return one(None)
    else:
        return [one(s) for s in shifts]

# ---------------- Plots ----------------
def plot_line(xs, ys, yerrs, xlabel, ylabel, outpng, title=None):
    try:
        plt.figure(figsize=(6,4), dpi=140)
        idx = np.arange(len(xs))
        plt.errorbar(idx, ys, yerr=yerrs, fmt="o", capsize=3)
        plt.xticks(idx, xs, rotation=45, ha="right")
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        if title: plt.title(title)
        ensure_dir(Path(outpng).parent)
        plt.tight_layout(); plt.savefig(outpng); plt.close()
    except Exception:
        pass

def plot_heat(axs, ays, grid, xlabel, ylabel, outpng, title=None):
    try:
        plt.figure(figsize=(6,5), dpi=140)
        plt.imshow(grid.T, origin="lower", aspect="auto",
                   extent=[axs[0], axs[-1], ays[0], ays[-1]])
        plt.colorbar(label="sin^2")
        plt.xlabel(xlabel); plt.ylabel(ylabel)
        if title: plt.title(title)
        ensure_dir(Path(outpng).parent)
        plt.tight_layout(); plt.savefig(outpng); plt.close()
    except Exception:
        pass

# ---------------- Modes ----------------
def run_warp(args):
    print("=== Uneven suite: warp sweep ===")
    print(f"L={args.L} jitter={args.jitter} n={args.n}  (ax,ay)=({args.ax},{args.ay})  star={args.star} ridge={args.ridge:.1e}")
    eps_list = np.linspace(args.warp_eps_min, args.warp_eps_max, args.warp_steps)
    rows=[["eps","sin2_mean","sin2_std","condG","condA","min_angle_deg","min_area","min_star","ok"]]
    xs=[]; ys=[]; es=[]
    for eps in eps_list:
        agg = aggregate_over_n(args.L,args.L,args.jitter,args.ax,args.ay,args.n,
                               args.star,args.ridge, warp_eps=eps, warp_mode=args.warp_mode,
                               grade=0.0, grade_band="x", shifts=None, seed0=args.seed)
        rows.append([eps, agg["sin2"], agg.get("sin2_std",np.nan), agg["condG"], agg["condA"],
                     agg["min_angle"], agg["min_area"], agg["min_star"], agg["ok"]])
        xs.append(f"{eps:.3f}"); ys.append(agg["sin2"]); es.append(agg.get("sin2_std",0.0))
        print(f"eps={eps:.3f}  sin2={agg['sin2']:.6f} ± {agg.get('sin2_std',0.0):.6f}  condG~{agg['condG']:.3f}  ok={agg['ok']}/{args.n}")
    write_csv(Path(args.out)/"warp_sweep.csv", rows[0], rows[1:])
    plot_line(xs, ys, es, "warp eps", "sin^2", Path(args.out)/"warp_sweep_sin2.png",
              title=f"Warp sweep (L={args.L}, star={args.star})")

def run_grade(args):
    print("=== Uneven suite: grade sweep ===")
    print(f"L={args.L} jitter={args.jitter} n={args.n}  (ax,ay)=({args.ax},{args.ay})  star={args.star} ridge={args.ridge:.1e}")
    g_list = np.linspace(args.grade_min, args.grade_max, args.grade_steps)
    rows=[["grade","sin2_mean","sin2_std","condG","condA","min_angle_deg","min_area","min_star","ok"]]
    xs=[]; ys=[]; es=[]
    for g in g_list:
        agg = aggregate_over_n(args.L,args.L,args.jitter,args.ax,args.ay,args.n,
                               args.star,args.ridge, warp_eps=0.0, warp_mode=args.warp_mode,
                               grade=g, grade_band=args.grade_band, shifts=None, seed0=args.seed)
        rows.append([g, agg["sin2"], agg.get("sin2_std",np.nan), agg["condG"], agg["condA"],
                     agg["min_angle"], agg["min_area"], agg["min_star"], agg["ok"]])
        xs.append(f"{g:.3f}"); ys.append(agg["sin2"]); es.append(agg.get("sin2_std",0.0))
        print(f"grade={g:.3f}  sin2={agg['sin2']:.6f} ± {agg.get('sin2_std',0.0):.6f}  condG~{agg['condG']:.3f}  ok={agg['ok']}/{args.n}")
    write_csv(Path(args.out)/"grade_sweep.csv", rows[0], rows[1:])
    plot_line(xs, ys, es, "grade", "sin^2", Path(args.out)/"grade_sweep_sin2.png",
              title=f"Grade sweep (L={args.L}, star={args.star})")

def run_autosweep(args):
    print("=== Uneven suite: (ax,ay) autosweep ===")
    print(f"L={args.L} jitter={args.jitter} n={args.n} star={args.star} ridge={args.ridge:.1e}")
    axs = np.linspace(args.ax_min, args.ax_max, args.ax_steps)
    ays = np.linspace(args.ay_min, args.ay_max, args.ay_steps)
    grid = np.full((len(axs), len(ays)), np.nan)
    rows=[["ax","ay","sin2_mean","sin2_std","condG","condA","min_angle_deg","min_area","min_star","ok"]]
    best=None
    for i,ax in enumerate(axs):
        for j,ay in enumerate(ays):
            agg = aggregate_over_n(args.L,args.L,args.jitter,ax,ay,args.n,
                                   args.star,args.ridge, warp_eps=args.warp_eps, warp_mode=args.warp_mode,
                                   grade=args.grade, grade_band=args.grade_band, shifts=None, seed0=args.seed)
            grid[i,j]=agg["sin2"]
            rows.append([ax,ay, agg["sin2"], agg.get("sin2_std",np.nan), agg["condG"], agg["condA"],
                         agg["min_angle"], agg["min_area"], agg["min_star"], agg["ok"]])
            print(f"(ax,ay)=({ax:.3f},{ay:.3f})  sin2={agg['sin2']:.6f} ± {agg.get('sin2_std',0.0):.6f}  condG~{agg['condG']:.3f}  ok={agg['ok']}/{args.n}")
            if best is None or abs(agg["sin2"]-args.target)<abs(best[2]-args.target):
                best=(ax,ay,agg["sin2"])
    write_csv(Path(args.out)/"autosweep_grid.csv", rows[0], rows[1:])
    plot_heat(axs, ays, grid, "ax", "ay", Path(args.out)/"autosweep_heatmap.png",
              title=f"Autosweep (L={args.L}, star={args.star})")
    if best:
        print(f"=== Best near target ===")
        print(f"ax={best[0]:.3f} ay={best[1]:.3f}  sin2={best[2]:.6f}  |Δ|={abs(best[2]-args.target):.6f}")

def run_cycles(args):
    print("=== Uneven suite: cycle-shift stability ===")
    print(f"L={args.L} jitter={args.jitter} n={args.n}  (ax,ay)=({args.ax},{args.ay})  star={args.star} ridge={args.ridge:.1e}")
    # parse shifts
    shifts=[]
    for tok in args.shifts.split(";"):
        tok=tok.strip()
        if not tok: continue
        a,b = tok.split(",")
        shifts.append((int(a), int(b)))
    rows=[["dx","dy","sin2_mean","sin2_std","condG","condA","min_angle_deg","min_area","min_star","ok"]]
    for (dx,dy) in shifts:
        agg = aggregate_over_n(args.L,args.L,args.jitter,args.ax,args.ay,args.n,
                               args.star,args.ridge, warp_eps=args.warp_eps, warp_mode=args.warp_mode,
                               grade=args.grade, grade_band=args.grade_band, shifts=[(dx,dy)], seed0=args.seed)[0]
        rows.append([dx,dy, agg["sin2"], agg.get("sin2_std",np.nan), agg["condG"], agg["condA"],
                     agg["min_angle"], agg["min_area"], agg["min_star"], agg["ok"]])
        print(f"shift=(x={dx},y={dy})  sin2={agg['sin2']:.6f} ± {agg.get('sin2_std',0.0):.6f}  condG~{agg['condG']:.3f}  (ok={agg['ok']}/{args.n})")
    write_csv(Path(args.out)/"cycle_shifts.csv", rows[0], rows[1:])

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Non-even lattice robustness suite for sin^2 invariant")
    ap.add_argument("--mode", type=str, default="warp", choices=["warp","grade","autosweep","cycles"])
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--ax", type=float, default=2.59)
    ap.add_argument("--ay", type=float, default=0.78)
    ap.add_argument("--star", type=str, default="cotan", choices=["cotan","diag"])
    ap.add_argument("--ridge", type=float, default=1e-11)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", type=str, default="out_uneven")

    # warp params
    ap.add_argument("--warp_mode", type=str, default="sinxy", choices=["sinxy","shear"])
    ap.add_argument("--warp_eps_min", type=float, default=0.0)
    ap.add_argument("--warp_eps_max", type=float, default=0.2)
    ap.add_argument("--warp_steps", type=int, default=6)
    ap.add_argument("--warp_eps", type=float, default=0.0)  # used by autosweep/cycles

    # grade params
    ap.add_argument("--grade_band", type=str, default="x", choices=["x","y"])
    ap.add_argument("--grade_min", type=float, default=0.0)
    ap.add_argument("--grade_max", type=float, default=0.6)
    ap.add_argument("--grade_steps", type=int, default=7)
    ap.add_argument("--grade", type=float, default=0.0)  # used by autosweep/cycles

    # autosweep ranges
    ap.add_argument("--ax_min", type=float, default=2.4)
    ap.add_argument("--ax_max", type=float, default=2.8)
    ap.add_argument("--ax_steps", type=int, default=9)
    ap.add_argument("--ay_min", type=float, default=0.72)
    ap.add_argument("--ay_max", type=float, default=0.84)
    ap.add_argument("--ay_steps", type=int, default=7)
    ap.add_argument("--target", type=float, default=0.231)

    # cycle shifts
    ap.add_argument("--shifts", type=str, default="0,0;1,0;0,1;3,2;5,5")

    args = ap.parse_args()
    outdir = Path(args.out); ensure_dir(outdir)

    if args.mode=="warp":
        run_warp(args)
    elif args.mode=="grade":
        run_grade(args)
    elif args.mode=="autosweep":
        run_autosweep(args)
    elif args.mode=="cycles":
        run_cycles(args)

if __name__ == "__main__":
    main()