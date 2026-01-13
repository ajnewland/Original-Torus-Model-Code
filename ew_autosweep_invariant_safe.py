
# ew_autosweep_invariant_safe.py
# Robust autosweep for sin^2(theta_W) using the invariant from
#   G = (C A^{-1} C^T)^{-1}
# with exact topological fundamental cycles C and cotan star_1 (A).
# - Handles early stop safely
# - Writes partial grids with NaNs masked in plots

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

# ---------------- Mesh: periodic rectangular triangulation ----------------
def build_periodic_grid(Lx, Ly, jitter, ax=1.0, ay=1.0, seed=0):
    """
    Returns:
      P: (V,2) vertex coords in [0,ax)×[0,ay)
      E: (E,2) undirected edges with increasing (min,max) vertex ids
      F: (F,3) faces (CCW)
      d0: (E,V) incidence
      face_edges, face_signs: per-face edge id and orientation sign
      v_ix, v_iy: per-vertex integer grid coordinates (0..Lx-1, 0..Ly-1)
    """
    rng = np.random.default_rng(seed)
    def vid(ix, iy): return (iy % Ly) * Lx + (ix % Lx)

    ix = np.arange(Lx)
    iy = np.arange(Ly)
    XX, YY = np.meshgrid(ix, iy, indexing="xy")
    v_ix = XX.ravel()
    v_iy = YY.ravel()

    X = (v_ix.astype(float) / Lx) * ax
    Y = (v_iy.astype(float) / Ly) * ay
    P = np.column_stack([X, Y])
    if jitter > 0.0:
        dx = (ax / Lx) * jitter
        dy = (ay / Ly) * jitter
        P += rng.uniform(-1, 1, size=P.shape) * np.array([dx, dy])

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
        a, b = key
        sgn = +1 if (u == a and v == b) else -1
        return ei, sgn

    for fi, (a, b, c) in enumerate(F):
        for k, (u, v) in enumerate([(a,b),(b,c),(c,a)]):
            ei, sgn = edge_index_oriented(u, v)
            face_edges[fi, k] = ei
            face_signs[fi, k] = sgn

    E = np.array(E_pairs, dtype=int)
    V = P.shape[0]; Ecount = E.shape[0]

    rows, cols, data = [], [], []
    for ei,(i,j) in enumerate(E):
        rows += [ei, ei]; cols += [i, j]; data += [-1.0, +1.0]
    d0 = sp.csr_matrix((data, (rows, cols)), shape=(Ecount, V))

    return P, E, F, d0, face_edges, face_signs, v_ix, v_iy

# ---------------- DEC: cotan star on 1-forms ----------------
def cot_angle(A, B, C):
    v1 = B - A; v2 = C - A
    dot = float(np.dot(v1, v2))
    nrm = math.sqrt(max(np.dot(v1, v1) * np.dot(v2, v2), 1e-30))
    c = max(min(dot / nrm, 1.0), -1.0)
    s = math.sqrt(max(1.0 - c*c, 0.0))
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
        A,B,C = face_pts[fi]
        cotA = cot_angle(A,B,C)
        cotB = cot_angle(B,C,A)
        cotC = cot_angle(C,A,B)
        cot_opp[fi,0] = cotC  # opp (A,B)
        cot_opp[fi,1] = cotA  # opp (B,C)
        cot_opp[fi,2] = cotB  # opp (C,A)

    vals = np.zeros(Ecount)
    for ei,(i,j) in enumerate(E):
        le = npl.norm(P[j] - P[i])
        csum = 0.0
        for fi in edge_faces[ei]:
            a,b,c = F[fi]
            for k,(u,v) in enumerate([(a,b),(b,c),(c,a)]):
                if (u==i and v==j) or (u==j and v==i):
                    csum += cot_opp[fi,k]
        vals[ei] = 0.5 * max(csum, 0.0) * max(le, 1e-12)
    vals = np.maximum(vals, 1e-12)
    return sp.diags(vals, 0, format="csc")

# ---------------- Exact fundamental cycles (topological) ----------------
def build_exact_cycles(Lx, Ly, E, v_ix, v_iy):
    """Two rank-2 fundamental loops: row y=0 and column x=0"""
    def vid(ix, iy): return (iy % Ly) * Lx + (ix % Lx)
    undirected = {(min(i,j), max(i,j)): ei for ei,(i,j) in enumerate(E)}

    def edge_index_oriented(i,j):
        a,b = (i,j) if i<j else (j,i)
        ei = undirected.get((a,b), None)
        if ei is None: return None, 0
        sgn = +1 if (i<j) else -1
        return ei, sgn

    C = np.zeros((2, E.shape[0]), float)

    # horizontal loop (y=0), +x
    y0 = 0
    for x in range(Lx):
        u = vid(x, y0); v = vid(x+1, y0)
        ei, sgn = edge_index_oriented(u, v)
        if ei is not None: C[0, ei] += sgn

    # vertical loop (x=0), +y
    x0 = 0
    for y in range(Ly):
        u = vid(x0, y); v = vid(x0, y+1)
        ei, sgn = edge_index_oriented(u, v)
        if ei is not None: C[1, ei] += sgn

    if np.linalg.matrix_rank(C) < 2:
        raise RuntimeError("Cycle matrix C lost rank (<2).")
    return sp.csr_matrix(C)

# ---------------- Invariant and sin^2(theta_W) ----------------
def sin2_from_G(star1, C, ridge=1e-11, cond_cap=1e4):
    """
    Compute G^{-1} = C A^{-1} C^T, then G = inv(G^{-1}),
    sin^2 = λ_min(G)/(λ_min(G)+λ_max(G)).
    Returns (sin2, condG, G) or (nan, nan, None) if ill-conditioned.
    """
    try:
        A = star1.tocsc()
        X = spla.spsolve(A + ridge*sp.eye(A.shape[0], format="csc"), C.T)  # (E×2)
        Ginv = C @ X  # 2×2 dense
        Ginv = 0.5*(Ginv + Ginv.T)

        # 2×2 condition via eigenvalues
        t = Ginv[0,0] + Ginv[1,1]
        d = (Ginv[0,0]-Ginv[1,1])**2 + 4*Ginv[0,1]*Ginv[1,0]
        d = max(d, 0.0); s = math.sqrt(d)
        lam_min_inv = 0.5*(t - s)
        lam_max_inv = 0.5*(t + s)
        if lam_min_inv <= 0 or lam_max_inv <= 0:
            return np.nan, np.nan, None
        cond = lam_max_inv / lam_min_inv
        if not np.isfinite(cond) or cond > cond_cap:
            return np.nan, np.nan, None

        det = Ginv[0,0]*Ginv[1,1] - Ginv[0,1]*Ginv[1,0]
        if abs(det) < 1e-18: return np.nan, np.nan, None
        G = (1.0/det)*np.array([[ Ginv[1,1], -Ginv[0,1]],
                                [-Ginv[1,0],  Ginv[0,0]]], float)
        trG = G[0,0] + G[1,1]
        dG  = (G[0,0]-G[1,1])**2 + 4*G[0,1]*G[1,0]
        dG  = max(dG, 0.0); sG = math.sqrt(dG)
        lam_min = 0.5*(trG - sG)
        lam_max = 0.5*(trG + sG)
        if lam_min <= 0 or lam_max <= 0:
            return np.nan, np.nan, None
        sin2 = lam_min / (lam_min + lam_max)
        return float(sin2), float(cond), G
    except Exception:
        return np.nan, np.nan, None

# ---------------- One setting (average over n meshes) ----------------
def one_setting_eval(Lx, Ly, jitter, ax, ay, n, seed0, ridge=1e-11, retries=5):
    sin2_list = []; cond_list = []
    ok = 0
    for k in range(n):
        seed = seed0 + 97*k
        # small retry loop (rarely needed if a solve fails)
        s_ok = False
        for _ in range(max(1,retries)):
            P,E,F,d0,fe,fs, vix, viy = build_periodic_grid(Lx, Ly, jitter, ax, ay, seed)
            star1 = star1_cotan(P, E, F, fe)
            C = build_exact_cycles(Lx, Ly, E, vix, viy)
            sin2, condG, _ = sin2_from_G(star1, C, ridge=ridge)
            if np.isfinite(sin2):
                sin2_list.append(sin2); cond_list.append(condG); ok += 1
                s_ok = True; break
            # else retry with a nudged seed
            seed += 1
        if not s_ok:
            # leave a hole for this replicate
            pass

    if ok == 0:
        return dict(ax=ax, ay=ay, ok=0, sin2=np.nan, cond=np.nan)
    return dict(ax=ax, ay=ay, ok=ok,
                sin2=float(np.mean(sin2_list)),
                cond=float(np.mean(cond_list)))

# ---------------- Plot helper ----------------
def save_heatmap(grid, xs, ys, title, path_png, vmin=0.0, vmax=1.0):
    ensure_dir(Path(path_png).parent)
    M = np.array(grid, float)
    mask = ~np.isfinite(M)
    Mmasked = np.ma.array(M, mask=mask)
    plt.figure(figsize=(6.6,4.6), dpi=130)
    im = plt.imshow(Mmasked.T, origin="lower", aspect="auto",
                    extent=[xs[0], xs[-1], ys[0], ys[-1]],
                    vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="value")
    plt.xlabel("ax"); plt.ylabel("ay"); plt.title(title)
    plt.tight_layout(); plt.savefig(path_png); plt.close()

# ---------------- Driver (autosweep) ----------------
def autosweep(Lx, Ly, jitter, n,
              ax_min, ax_max, ax_steps,
              ay_min, ay_max, ay_steps,
              target, tol,
              ridge, retries, seed0, outdir,
              early_stop=True):

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(ax_min, ax_max, ax_steps)
    ys = np.linspace(ay_min, ay_max, ay_steps)

    # initialize grids with NaN
    sin2_grid = np.full((ax_steps, ay_steps), np.nan, float)
    cond_grid = np.full((ax_steps, ay_steps), np.nan, float)
    ok_grid   = np.zeros((ax_steps, ay_steps), int)

    best = dict(err=np.inf, ax=np.nan, ay=np.nan, sin2=np.nan, cond=np.nan, ok=0)

    print(f"=== Safe autosweep (invariant from G) ===")
    print(f"Lx={Lx} Ly={Ly} jitter={jitter}  n={n}")
    print(f"ax in [{ax_min},{ax_max}] steps={ax_steps}  ay in [{ay_min},{ay_max}] steps={ay_steps}")
    print(f"target={target}  tol={tol}  ridge={ridge}  retries={retries}\n")

    stop_flag = False
    for ia, ax in enumerate(xs):
        for ja, ay in enumerate(ys):
            r = one_setting_eval(Lx, Ly, jitter, float(ax), float(ay),
                                 n=n, seed0=seed0, ridge=ridge, retries=retries)
            sin2_grid[ia, ja] = r["sin2"]
            cond_grid[ia, ja] = r["cond"]
            ok_grid[ia, ja]   = r["ok"]

            if r["ok"] > 0 and np.isfinite(r["sin2"]):
                print(f"(ax,ay)=({ax:0.3f},{ay:0.3f})  sin2={r['sin2']}  ok={r['ok']}  cond~{r['cond']}")
                err = abs(r["sin2"] - target)
                if err < best["err"]:
                    best = dict(err=err, ax=float(ax), ay=float(ay),
                                sin2=float(r["sin2"]), cond=float(r["cond"]), ok=int(r["ok"]))
                if early_stop and err <= tol:
                    print("\nEarly stop: target reached within tolerance.")
                    stop_flag = True
                    break
            else:
                print(f"(ax,ay)=({ax:0.3f},{ay:0.3f})  sin2=NaN  (degenerate periods or failure in some meshes)")
        if stop_flag:
            break

    # write CSV grid
    rows = []
    for ia, ax in enumerate(xs):
        for ja, ay in enumerate(ys):
            rows.append([ax, ay, ok_grid[ia,ja], sin2_grid[ia,ja], cond_grid[ia,ja]])
    write_csv(outdir/"autosweep_grid.csv",
              header=["ax","ay","ok","sin2","cond"],
              rows_float2d=rows)

    # summary text + best
    if np.isfinite(best["err"]):
        summary = (f"=== Best setting ===\n"
                   f"Best near target at ax={best['ax']:.3f}, ay={best['ay']:.3f}  "
                   f"sin2={best['sin2']:.6f}  |sin2-target|={best['err']:.6f}  "
                   f"ok={best['ok']}  cond~{best['cond']:.3f}\n")
    else:
        summary = "No valid settings found (all NaN). Consider relaxing cond cap or increasing retries.\n"

    write_text(outdir/"summary.txt",
               "=== Safe autosweep (invariant from G) ===\n"
               f"Lx={Lx} Ly={Ly} jitter={jitter}  n={n}\n"
               f"ax in [{ax_min},{ax_max}] steps={ax_steps}  ay in [{ay_min},{ay_max}] steps={ay_steps}\n"
               f"target={target}  tol={tol}  ridge={ridge}  retries={retries}\n\n" + summary)

    # plots (mask NaNs)
    save_heatmap(sin2_grid, xs, ys, "sin^2(theta_W)", outdir/"heatmap_sin2.png", vmin=0.0, vmax=1.0)
    save_heatmap(cond_grid,  xs, ys, "cond(G_inv)",    outdir/"heatmap_cond.png", vmin=0.0, vmax=np.nanmax(cond_grid)*1.05 if np.isfinite(np.nanmax(cond_grid)) else 1.0)

    print("\n" + summary.strip())
    print(f"Wrote: {outdir/'autosweep_grid.csv'}")
    print(f"Wrote: {outdir/'summary.txt'}")
    print(f"Wrote: {outdir/'heatmap_sin2.png'}")
    print(f"Wrote: {outdir/'heatmap_cond.png'}")

def main():
    ap = argparse.ArgumentParser(description="Autosweep sin^2(theta_W) from invariant G with exact cycles")
    ap.add_argument("--Lx", type=int, default=20)
    ap.add_argument("--Ly", type=int, default=20)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=16)

    ap.add_argument("--ax_min", type=float, default=1.30)
    ap.add_argument("--ax_max", type=float, default=1.55)
    ap.add_argument("--ax_steps", type=int, default=6)
    ap.add_argument("--ay_min", type=float, default=0.95)
    ap.add_argument("--ay_max", type=float, default=1.10)
    ap.add_argument("--ay_steps", type=int, default=4)

    ap.add_argument("--target", type=float, default=0.231)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--ridge", type=float, default=1e-11)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", type=str, default="out_autosweep")
    ap.add_argument("--no_early_stop", action="store_true", help="disable early stop when within tolerance")

    args = ap.parse_args()
    autosweep(Lx=args.Lx, Ly=args.Ly, jitter=args.jitter,
              n=args.n,
              ax_min=args.ax_min, ax_max=args.ax_max, ax_steps=args.ax_steps,
              ay_min=args.ay_min, ay_max=args.ay_max, ay_steps=args.ay_steps,
              target=args.target, tol=args.tol,
              ridge=args.ridge, retries=args.retries,
              seed0=args.seed, outdir=args.out,
              early_stop=(not args.no_early_stop))

if __name__ == "__main__":
    main()