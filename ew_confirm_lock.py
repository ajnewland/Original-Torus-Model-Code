# ew_confirm_lock_invariant.py
# Basis-invariant sin^2(theta_W) from eigenvalues of G = C A^{-1} C^T
# Robust "confirm-and-lock" across multiple (ax, ay) settings, no KKT systems.

import argparse, math
from pathlib import Path
import numpy as np
import numpy.linalg as npl
import scipy.sparse as sp
import matplotlib.pyplot as plt

# ---------------- I/O helpers ----------------
def write_text(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

def savetxt_utf8(path, arr2d, header):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr2d, delimiter=",", fmt="%.10g",
               header=header, comments="", encoding="utf-8")

# ---------------- Mesh (periodic rectangular triangulation) ----------------
def build_periodic_grid(Lx, Ly, jitter, ax=1.0, ay=1.0, seed=0):
    rng = np.random.default_rng(seed)

    def vid(ix, iy): return (iy % Ly) * Lx + (ix % Lx)

    X = np.linspace(0, ax, Lx, endpoint=False)
    Y = np.linspace(0, ay, Ly, endpoint=False)
    xx, yy = np.meshgrid(X, Y, indexing="xy")
    P = np.column_stack([xx.ravel(), yy.ravel()])

    if jitter > 0:
        dx = (ax / Lx) * jitter
        dy = (ay / Ly) * jitter
        P += rng.uniform(-1, 1, size=P.shape) * np.array([dx, dy])

    faces = []
    for iy in range(Ly):
        for ix in range(Lx):
            i  = vid(ix, iy)
            j  = vid(ix+1, iy)
            k  = vid(ix, iy+1)
            l  = vid(ix+1, iy+1)
            # two triangles per cell
            faces.append([i, j, l])
            faces.append([i, l, k])
    F = np.array(faces, dtype=int)

    undirected = {}
    E_pairs = []
    face_edges = np.zeros((F.shape[0], 3), dtype=int)
    face_signs = np.zeros((F.shape[0], 3), dtype=int)

    def edge_id(a, b):
        return ((a,b), +1) if a<b else ((b,a), -1)

    for fi,(a,b,c) in enumerate(F):
        for k,(u,v) in enumerate([(a,b),(b,c),(c,a)]):
            key, sgn = edge_id(u,v)
            if key not in undirected:
                undirected[key] = len(E_pairs)
                E_pairs.append(key)
            ei = undirected[key]
            face_edges[fi,k] = ei
            face_signs[fi,k] = sgn

    E = np.array(E_pairs, dtype=int)
    V = P.shape[0]; Ecount = E.shape[0]; Fcount = F.shape[0]

    # d0 (E x V)
    rows, cols, data = [], [], []
    for ei,(i,j) in enumerate(E):
        rows += [ei, ei]
        cols += [i, j]
        data += [-1.0, +1.0]
    d0 = sp.csr_matrix((data, (rows, cols)), shape=(Ecount, V))

    # d1 (F x E)
    rows, cols, data = [], [], []
    for fi in range(Fcount):
        for k in range(3):
            rows.append(fi); cols.append(face_edges[fi,k]); data.append(float(face_signs[fi,k]))
    d1 = sp.csr_matrix((data, (rows, cols)), shape=(Fcount, Ecount))

    return P, E, F, d0, d1, face_edges, face_signs

# ---------------- DEC: cotan star_1 (diagonal) ----------------
def cot_angle(A,B,C):
    v1 = B-A; v2 = C-A
    dot = float(np.dot(v1,v2))
    nrm = math.sqrt(max(np.dot(v1,v1)*np.dot(v2,v2), 1e-30))
    c = max(min(dot/nrm,1.0),-1.0)
    s = math.sqrt(max(1.0-c*c,0.0))
    return 0.0 if s<1e-14 else c/s

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

    star_vals = np.zeros(Ecount)
    for ei,(i,j) in enumerate(E):
        le = npl.norm(P[j]-P[i])
        csum=0.0
        for fi in edge_faces[ei]:
            A,B,C = F[fi]
            for k,(u,v) in enumerate([(A,B),(B,C),(C,A)]):
                if (u==i and v==j) or (u==j and v==i):
                    csum += cot_opp[fi,k]
        star_vals[ei] = 0.5 * csum * le

    # ensure strictly positive (guard numerical degeneracy)
    star_vals = np.maximum(star_vals, 1e-12)
    return sp.diags(star_vals, 0, format="csc"), star_vals

# ---------------- Period matrix C (2 x E), exact wraps ----------------
def period_matrix(P, E):
    """
    Row 0: edges that wrap across x (torus x-cycle)
    Row 1: edges that wrap across y (torus y-cycle)
    """
    Ecount = E.shape[0]
    xs = P[:,0]; ys = P[:,1]
    Lx = xs.max() - xs.min() + 1e-12
    Ly = ys.max() - ys.min() + 1e-12
    rows, cols, data = [], [], []

    # x-cycle: detect |delta x| > Lx/2
    for ei,(i,j) in enumerate(E):
        dx = xs[j]-xs[i]
        if abs(dx) > 0.5*Lx:
            rows.append(0); cols.append(ei); data.append(np.sign(dx))

    # y-cycle: detect |delta y| > Ly/2
    for ei,(i,j) in enumerate(E):
        dy = ys[j]-ys[i]
        if abs(dy) > 0.5*Ly:
            rows.append(1); cols.append(ei); data.append(np.sign(dy))

    C = sp.csr_matrix((data, (rows, cols)), shape=(2, Ecount))
    # Fallback: if a row is empty, fall back to majority direction edges
    for r in [0,1]:
        if C.getrow(r).nnz == 0:
            # pick top 1% longest edges in that direction
            if r==0:
                dabs = np.array([abs(xs[j]-xs[i]) for (i,j) in E])
            else:
                dabs = np.array([abs(ys[j]-ys[i]) for (i,j) in E])
            k = max(1, int(0.01*len(dabs)))
            idx = np.argpartition(-dabs, k)[:k]
            C[r, idx] = 1.0
            C.eliminate_zeros()
    return C.tocsr()

# ---------------- Single-geometry EW readout (invariant) ----------------
def ew_from_geometry(Lx, Ly, jitter, ax, ay, v_higgs, seed):
    P,E,F,d0,d1,face_edges,face_signs = build_periodic_grid(Lx, Ly, jitter, ax=ax, ay=ay, seed=seed)

    # Star_1 (A diagonal) and periods C
    star1, star_vals = star1_cotan(P, E, F, face_edges)  # A = diag(star_vals)
    Ainv = sp.diags(1.0 / star_vals, 0, format="csc")
    C = period_matrix(P, E)

    # G = C A^{-1} C^T  (2x2 SPD)
    G = (C @ (Ainv @ C.T)).toarray()
    # eigenvalues of G (ascending)
    evals = np.sort(npl.eigvalsh(G))
    # guard
    evals = np.clip(evals, 1e-14, None)
    gmin, gmax = float(evals[0]), float(evals[1])
    # Basis-invariant weak angle
    sin2_eig = gmin / (gmin + gmax)
    condG = gmax / gmin

    # For monitoring, also show the "lab proxy" in the period basis:
    Kphys = npl.inv(G)  # 2x2
    Kxx, Kyy = float(Kphys[0,0]), float(Kphys[1,1])
    sin2_lab = Kxx / max(Kxx + Kyy, 1e-16)

    # Toy masses (scale-free; reported just for consistency)
    g_mag  = math.sqrt(max(Kxx, 1e-16))
    gp_mag = math.sqrt(max(Kyy, 1e-16))
    mW = v_higgs * g_mag / 2.0
    mZ = v_higgs * math.sqrt(g_mag**2 + gp_mag**2) / 2.0

    return dict(ax=ax, ay=ay, Lx=Lx, Ly=Ly, jitter=jitter, seed=seed,
                sin2_eig=sin2_eig, sin2_lab=sin2_lab,
                gmin=gmin, gmax=gmax, condG=condG,
                Kxx=Kxx, Kyy=Kyy, Kxy=float(Kphys[0,1]),
                v=v_higgs, mW=mW, mZ=mZ)

# ---------------- Aggregation and plots ----------------
def summarize_rows(rows):
    arr = {k: np.array([r[k] for r in rows], float)
           for k in rows[0].keys() if isinstance(rows[0][k], (int,float))}
    out = {}
    def s(m): return float(np.nanmean(arr[m])), float(np.nanstd(arr[m]))
    for key in ["sin2_eig","sin2_lab","gmin","gmax","condG","Kxx","Kyy","Kxy","mW","mZ"]:
        mu, sd = s(key); out[f"{key}_mean"]=mu; out[f"{key}_std"]=sd
    return out

def plot_compare_sin2(settings, per_setting_rows, outpng):
    plt.figure(figsize=(6.6,4.6), dpi=140)
    xs, lab_means, eig_means, lab_std, eig_std = [], [], [], [], []
    for (ax,ay), rows in per_setting_rows:
        xs.append(f"{ax:.2f},{ay:.2f}")
        vlab = np.array([r["sin2_lab"] for r in rows])
        veig = np.array([r["sin2_eig"] for r in rows])
        lab_means.append(vlab.mean()); lab_std.append(max(vlab.std(), 1e-12))
        eig_means.append(veig.mean()); eig_std.append(max(veig.std(), 1e-12))
    idx = np.arange(len(xs))
    plt.errorbar(idx-0.06, lab_means, yerr=lab_std, fmt="o", capsize=3, label="sin^2 (lab proxy)")
    plt.errorbar(idx+0.06, eig_means, yerr=eig_std, fmt="s", capsize=3, label="sin^2 (eigen, invariant)")
    plt.axhline(0.231, ls="--", lw=1.0, label="target 0.231")
    plt.xticks(idx, xs, rotation=45, ha="right")
    plt.ylabel("sin^2(theta_W)")
    plt.xlabel("(ax, ay)")
    plt.legend()
    Path(outpng).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpng)
    plt.close()

# ---------------- Driver ----------------
def parse_settings(s: str):
    out=[]
    for tok in s.split(";"):
        tok = tok.strip()
        if tok:
            a,b = tok.split(",")
            out.append((float(a), float(b)))
    return out

def run_confirm_lock(Lx, Ly, jitter, settings, n, v, outdir, seed):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    per_setting_rows = []
    print("=== DEC EW confirm-and-lock (invariant from G) ===")
    print(f"Lx={Lx} Ly={Ly} jitter={jitter}  n={n}")
    print(f"settings={settings}")
    print("star1=cotan")

    lines=[]
    for (ax,ay) in settings:
        rows=[]
        for k in range(n):
            s = seed + 97*k
            rows.append(ew_from_geometry(Lx, Ly, jitter, ax, ay, v_higgs=v, seed=s))
        per_setting_rows.append(((ax,ay), rows))

        # per-setting CSV
        hdr = ["ax","ay","Lx","Ly","jitter","seed",
               "sin2_eig","sin2_lab","gmin","gmax","condG","Kxx","Kyy","Kxy","v","mW","mZ"]
        arr = np.array([[r[h] for h in hdr] for r in rows], float)
        savetxt_utf8(outdir / f"per_setting_ax{ax:.2f}_ay{ay:.2f}.csv", arr, header=",".join(hdr))

        # console line
        M = summarize_rows(rows)
        ln = (f"(ax,ay)=({ax:.3f},{ay:.3f})  "
              f"sin2_eig mean={M['sin2_eig_mean']:.3f} +- {M['sin2_eig_std']:.3f}  "
              f"(lab={M['sin2_lab_mean']:.3f} +- {M['sin2_lab_std']:.3f})  "
              f"[condG≈{(M['condG_mean']):.2e}]")
        print(ln); lines.append(ln+"\n")

    # aggregates.csv
    hdrA = ["ax","ay",
            "sin2_eig_mean","sin2_eig_std","sin2_lab_mean","sin2_lab_std",
            "gmin_mean","gmax_mean","condG_mean",
            "Kxx_mean","Kyy_mean","Kxy_mean","mW_mean","mZ_mean"]
    arrA=[]
    for (ax,ay), rows in per_setting_rows:
        M = summarize_rows(rows)
        arrA.append([ax,ay,
                     M["sin2_eig_mean"],M["sin2_eig_std"],M["sin2_lab_mean"],M["sin2_lab_std"],
                     M["gmin_mean"],M["gmax_mean"],M["condG_mean"],
                     M["Kxx_mean"],M["Kyy_mean"],M["Kxy_mean"],
                     M["mW_mean"],M["mZ_mean"]])
    savetxt_utf8(outdir / "aggregates.csv", np.array(arrA,float), header=",".join(hdrA))

    write_text(outdir / "summary.txt",
               "=== DEC EW confirm-and-lock (invariant from G) ===\n"
               f"Lx={Lx} Ly={Ly} jitter={jitter}  n={n}\n"
               f"settings={settings}\n" + "".join(lines))

    # plot
    plot_compare_sin2(settings, per_setting_rows, outdir / "compare_sin2.png")

def main():
    ap = argparse.ArgumentParser(description="Confirm-and-lock DEC EW (basis-invariant sin^2 from G)")
    ap.add_argument("--Lx", type=int, default=16)
    ap.add_argument("--Ly", type=int, default=16)
    ap.add_argument("--jitter", type=float, default=0.03)
    ap.add_argument("--settings", type=str, default="1.30,1.08;1.32,1.09")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--v", type=float, default=246.0)
    ap.add_argument("--out", type=str, default="out_confirm")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    run_confirm_lock(args.Lx, args.Ly, args.jitter,
                     parse_settings(args.settings), args.n, args.v, args.out, args.seed)

if __name__ == "__main__":
    main()