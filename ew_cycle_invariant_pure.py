#!/usr/bin/env python3
# ew_cycle_invariant_pure.py
# Cycle-swap test using ONLY the invariant sin^2 from G = (C A^{-1} C^T)^{-1}.
# - Periodic triangulated torus (Lx x Ly cells, two triangles per cell)
# - Hodge star on edges: cotan or circumcentric (choose with --star)
# - Exact fundamental loops C built at shifts (dx,dy) to prove cycle independence
# Prints sin^2 mean/std for each shift; also condition number of G^{-1}.
# NOW ALSO:
# - Identifies canonical channels A=(0,0) and B=(1,0)
# - Builds A,B,cA,cB and r=ln(cA/cB), rho=cB/cA
# - Computes weight rules and predicted S vs S_tgt
# - Writes a full CSV with all columns needed downstream

import argparse, math
from pathlib import Path
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import csv
import os

# ---------------- I/O helpers ----------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def write_text(path, text):
    path = Path(path); ensure_dir(path.parent); path.write_text(text, encoding="utf-8")

def safe_mkdir_for(path: str) -> None:
    d = os.path.dirname(str(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

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
    v1 = np.array([q[0]-p[0], q[1]-p[1], 0.0])
    v2 = np.array([r[0]-p[0], r[1]-p[1], 0.0])
    return 0.5 * np.linalg.norm(np.cross(v1,v2))

def star1_circum(P,E,F,face_edges):
    Ee = E.shape[0]; Ff = F.shape[0]
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

    if np.linalg.matrix_rank(C) < 2:
        raise RuntimeError("Cycle matrix C has rank < 2 (degenerate loops).")
    return sp.csr_matrix(C)

# ---------------- invariant sin^2 from G ----------------
def sin2_from_G(star1, C, ridge=1e-11, cond_cap=1e12):
    """
    Solve (A+ridge I) X = C^T, G^{-1} = C X, return sin^2 from eigenvalues of G.
    Also returns cond number of G^{-1} (2x2) via its eigenvalues ratio.
    """
    A = star1.tocsc()
    I = sp.eye(A.shape[0], format="csc")
    solver = spla.factorized(A + ridge*I)
    Ct = C.T.toarray()  # (E,2)
    X = np.column_stack([solver(Ct[:,0]), solver(Ct[:,1])])  # (E,2)
    Ginv = (C @ X)  # 2x2 dense
    Ginv = 0.5 * (Ginv + Ginv.T)
    t = float(Ginv[0,0] + Ginv[1,1])
    d = (Ginv[0,0]-Ginv[1,1])**2 + 4.0*Ginv[0,1]*Ginv[1,0]
    d = max(d, 0.0); s = math.sqrt(d)
    lam_min_inv = 0.5*(t - s); lam_max_inv = 0.5*(t + s)
    if lam_min_inv <= 0 or lam_max_inv <= 0: return np.nan, np.nan, None, None, None
    condGinv = lam_max_inv / lam_min_inv
    if not np.isfinite(condGinv) or condGinv > cond_cap: return np.nan, np.nan, None, None, None
    det = Ginv[0,0]*Ginv[1,1] - Ginv[0,1]*Ginv[1,0]
    if abs(det) < 1e-18: return np.nan, np.nan, None, None, None
    G = (1.0/det) * np.array([[ Ginv[1,1], -Ginv[0,1]],
                              [-Ginv[1,0],  Ginv[0,0]]], float)
    w = np.linalg.eigvalsh(G)
    if w[0] <= 0 or w[1] <= 0: return np.nan, np.nan, None, None, None
    sin2 = float(w[0] / (w[0] + w[1]))
    # also return Ginv diag entries (proxies for per-channel "conditioning")
    return sin2, condGinv, float(Ginv[0,0]), float(Ginv[1,1]), G

# ---------------- star selector ----------------
def star1_from_name(name, P,E,F,fe):
    if name == "cotan":   return star1_cotan(P,E,F,fe)
    if name == "circum":  return star1_circum(P,E,F,fe)
    raise ValueError("star must be 'cotan' or 'circum'")

# ---------------- helpers: weights ----------------
def sigmoid(x: float) -> float:
    return 1.0/(1.0+math.exp(-x))

def ideal_weight(S_tgt: float, A: float, B: float) -> float:
    if abs(A - B) < 1e-15:
        return 0.5
    w = (S_tgt - B) / (A - B)
    return max(0.0, min(1.0, w))

def pow_softmax_weight(cA: float, cB: float, alpha: float = 1.0) -> float:
    # w = sigma(alpha * (ln cA - ln cB))
    r = math.log((cA + 1e-18) / (cB + 1e-18))
    return sigmoid(alpha * r)

def logistic_weight_from_contrast(k: float, b: float, cA: float, cB: float) -> float:
    r = math.log((cA + 1e-18) / (cB + 1e-18))
    return sigmoid(k * r + b)

# ---------------- parse shifts ----------------
def parse_shifts(s: str):
    out=[]
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok: continue
        a,b = tok.split(",")
        out.append((int(a), int(b)))
    return out

# ---------------- one setting (per shift; returns sin2/cond) ----------------
def one_setting(L, jitter, ax, ay, star, dx, dy, n, seed0, ridge):
    sin2s=[]; conds=[]; g0s=[]; g1s=[]
    for k in range(n):
        P,E,F,d0,fe,fs,vx,vy = build_periodic_grid(L, L, jitter, ax=ax, ay=ay, seed=seed0+97*k)
        A = star1_from_name(star, P,E,F,fe)
        C = build_exact_cycles_shifted(L, L, E, dx=dx, dy=dy)
        s2, cond, g00, g11, _ = sin2_from_G(A, C, ridge=ridge)
        if np.isfinite(s2):
            sin2s.append(s2); conds.append(cond)
            if g00 is not None and g11 is not None:
                g0s.append(g00); g1s.append(g11)
    if len(sin2s)==0:
        return np.nan, np.nan, 0, np.nan, np.nan
    return float(np.mean(sin2s)), float(np.std(sin2s)), len(sin2s), float(np.mean(conds)), (np.mean(g0s) if g0s else np.nan), (np.mean(g1s) if g1s else np.nan)

# ---------------- driver ----------------
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
    ap.add_argument("--out", type=str, required=True,
                    help="Output path. If ends with .csv, CSV written there; else folder will be created and 'cycle_rows.csv' written inside.")
    # weight-rule hyperparams + target
    ap.add_argument("--S_tgt", type=float, default=0.231)
    ap.add_argument("--alpha_pow", type=float, default=1.0)
    ap.add_argument("--k_log", type=float, default=2.2)
    ap.add_argument("--b_log", type=float, default=0.0)
    ap.add_argument("--delta_bias", type=float, default=0.0010)
    args = ap.parse_args()

    shifts = parse_shifts(args.shifts)
    print("=== EW cycle-swap (invariant, exact loops) ===")
    print(f"L={args.L} jitter={args.jitter} n={args.n}  (ax,ay)=({args.ax},{args.ay})")
    print(f"star={args.star}  ridge={args.ridge:.1e}")
    print(f"shifts={shifts}")

    # resolve CSV path
    if args.out.lower().endswith(".csv"):
        csv_path = args.out
        safe_mkdir_for(csv_path)
        out_dir = os.path.dirname(csv_path) or "."
    else:
        out_dir = args.out
        ensure_dir(out_dir)
        csv_path = os.path.join(out_dir, "cycle_rows.csv")

    # run all shifts, collect results
    per_shift = {}
    lines=[]
    for (dx,dy) in shifts:
        m,s,ok,cond,g00,g11 = one_setting(args.L, args.jitter, args.ax, args.ay,
                                          args.star, dx, dy, args.n, args.seed, args.ridge)
        per_shift[(dx,dy)] = dict(mean=m, std=s, ok=ok, cond=cond, g00=g00, g11=g11)
        line = f"shift=(x={dx},y={dy})  sin2={m:.6f} ± {s:.6f}  cond~{(cond if np.isfinite(cond) else float('nan')):.3f}  (ok={ok}/{args.n})"
        print(line); lines.append(line+"\n")

    # identify canonical A=(0,0) and B=(1,0)
    def need_shift(sx,sy):
        if (sx,sy) not in per_shift or not np.isfinite(per_shift[(sx,sy)]["mean"]):
            raise RuntimeError(f"Required shift ({sx},{sy}) missing or invalid. Present keys: {list(per_shift.keys())}")
        return per_shift[(sx,sy)]
    Arow = need_shift(0,0)
    Brow = need_shift(1,0)

    A  = float(Arow["mean"]);  cA = float(Arow["cond"])
    B  = float(Brow["mean"]);  cB = float(Brow["cond"])
    r   = math.log((cA + 1e-18)/(cB + 1e-18)) if (math.isfinite(cA) and math.isfinite(cB)) else float("nan")
    rho = (cB + 1e-18)/(cA + 1e-18) if (math.isfinite(cA) and math.isfinite(cB)) else float("nan")

    # weights & predictions (global per (ax,ay) using A,B,cA,cB)
    w_star     = ideal_weight(args.S_tgt, A, B)
    w_pow      = pow_softmax_weight(cA, cB, alpha=args.alpha_pow)
    w_log      = logistic_weight_from_contrast(args.k_log, args.b_log, cA, cB)
    S_star     = w_star*A + (1.0 - w_star)*B
    S_pow      = w_pow *A + (1.0 - w_pow )*B
    S_log      = w_log *A + (1.0 - w_log )*B
    S_log_bias = S_log - args.delta_bias
    err_star     = S_star     - args.S_tgt
    err_pow      = S_pow      - args.S_tgt
    err_log      = S_log      - args.S_tgt
    err_log_bias = S_log_bias - args.S_tgt

    # write CSV: one row per shift, but A,B,cA,cB,r,rho and the S_* fields are repeated (they are
    # global for the (ax,ay) setting and use the canonical A/B channels).
    cols = ["ax","ay","L","jitter","n","star","ridge",
            "shift_x","shift_y",
            "sin2","cond","ok",
            "A","B","cA","cB","r","rho",
            "w_star","S_star","err_star",
            "w_pow","S_pow","err_pow",
            "w_log","S_log","err_log",
            "w_log_bias","S_log_bias","err_log_bias"]
    safe_mkdir_for(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        wcsv = csv.DictWriter(fcsv, fieldnames=cols); wcsv.writeheader()
        for (dx,dy), rec in per_shift.items():
            row = dict(
                ax=args.ax, ay=args.ay, L=args.L, jitter=args.jitter, n=args.n, star=args.star, ridge=args.ridge,
                shift_x=dx, shift_y=dy,
                sin2=rec["mean"], cond=(rec["cond"] if np.isfinite(rec["cond"]) else ""),
                ok=rec["ok"],
                A=A, B=B, cA=cA, cB=cB, r=r, rho=rho,
                w_star=w_star, S_star=S_star, err_star=err_star,
                w_pow=w_pow, S_pow=S_pow, err_pow=err_pow,
                w_log=w_log, S_log=S_log, err_log=err_log,
                w_log_bias=S_log_bias, S_log_bias=S_log_bias, err_log_bias=err_log_bias
            )
            wcsv.writerow(row)

    # keep your text summary too (unchanged)
    if args.out:
        out_path = Path(args.out)
        if not str(out_path).lower().endswith(".csv"):
            # write the text file beside the CSV
            write_text(Path(args.out)/"cycle_swap_summary.txt",
                       "=== EW cycle-swap (invariant, exact loops) ===\n"+
                       f"L={args.L} jitter={args.jitter} n={args.n}  (ax,ay)=({args.ax},{args.ay})\n"+
                       f"star={args.star}  ridge={args.ridge:.1e}\n"+
                       f"shifts={shifts}\n" + "".join(lines))

    print(f"[ok] wrote CSV rows to {csv_path}")

if __name__ == "__main__":
    main()
