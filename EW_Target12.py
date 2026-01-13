import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, diags, bmat, csc_matrix
from scipy.sparse.linalg import spsolve

# ----------------------------
# Geometry & DEC utilities
# ----------------------------

def tri_area(p,q,r):
    return 0.5 * abs(np.cross(q - p, r - p))

def circumcenter(p,q,r):
    A = q - p
    B = r - p
    AdotA = np.dot(A,A)
    BdotB = np.dot(B,B)
    cross = A[0]*B[1] - A[1]*B[0]
    if abs(cross) < 1e-14:
        return (p+q+r)/3.0
    t = 0.5 * np.array([[ B[1], -A[1]],
                        [-B[0],  A[0]]]) @ np.array([AdotA, BdotB]) / (cross)
    return p + t

def build_torus_grid(Lx, Ly, jitter, ax, ay, seed):
    rng = np.random.default_rng(seed)

    def vid(i,j): return (i % Lx) + Lx * (j % Ly)

    xv = np.arange(Lx)
    yv = np.arange(Ly)
    X,Y = np.meshgrid(xv, yv, indexing='xy')
    X = X.reshape(-1)
    Y = Y.reshape(-1)

    P = np.stack([X/Lx, Y/Ly], axis=1).astype(float)
    P[:,0] *= ax
    P[:,1] *= ay
    if jitter > 0:
        P += jitter * np.column_stack([rng.standard_normal(P.shape[0]),
                                       rng.standard_normal(P.shape[0])]) / max(Lx,Ly)

    V = Lx*Ly

    F_list = []
    for j in range(Ly):
        for i in range(Lx):
            v00 = vid(i,j)
            v10 = vid(i+1,j)
            v01 = vid(i,j+1)
            v11 = vid(i+1,j+1)
            F_list.append((v00,v10,v11))
            F_list.append((v00,v11,v01))
    F = np.array(F_list, dtype=np.int32)
    Fcnt = F.shape[0]

    edge_dict = {}
    def add_edge(a,b):
        if a>b: a,b = b,a
        if (a,b) not in edge_dict:
            edge_dict[(a,b)] = len(edge_dict)
        return edge_dict[(a,b)]

    face_edges = np.zeros((Fcnt,3), dtype=np.int32)
    face_signs = np.zeros((Fcnt,3), dtype=np.int8)
    for fi,(a,b,c) in enumerate(F):
        trip = [(a,b),(b,c),(c,a)]
        for k,(u,v) in enumerate(trip):
            if u < v:
                ei = add_edge(u,v); s = +1
            else:
                ei = add_edge(v,u); s = -1
            face_edges[fi,k] = ei
            face_signs[fi,k] = s

    E_pairs = np.array(list(edge_dict.keys()), dtype=np.int32)
    Ecnt = E_pairs.shape[0]
    rows = np.repeat(np.arange(Ecnt), 2)
    cols = np.concatenate([E_pairs[:,0], E_pairs[:,1]])
    vals = np.concatenate([-np.ones(Ecnt), np.ones(Ecnt)])
    d0 = coo_matrix((vals,(rows,cols)), shape=(Ecnt, V)).tocsr()

    cx = np.zeros(Ecnt)
    cy = np.zeros(Ecnt)
    for i in range(Lx):
        a = vid(i,0); b = vid(i+1,0)
        u,v = (a,b) if a<b else (b,a)
        ei = edge_dict[(u,v)]
        s  = +1 if (a<b) else -1
        cx[ei] += s
    for j in range(Ly):
        a = vid(0,j); b = vid(0,j+1)
        u,v = (a,b) if a<b else (b,a)
        ei = edge_dict[(u,v)]
        s  = +1 if (a<b) else -1
        cy[ei] += s

    return P, V, E_pairs, F, d0, face_edges, face_signs, cx, cy

def circum_star1(P, E_pairs, F, face_edges, face_signs):
    V = P.shape[0]
    Ecnt = E_pairs.shape[0]
    Fcnt = F.shape[0]

    C = np.zeros((Fcnt,2))
    for fi,(a,b,c) in enumerate(F):
        C[fi,:] = circumcenter(P[a], P[b], P[c])

    adj = [[] for _ in range(Ecnt)]
    for fi in range(Fcnt):
        for k in range(3):
            ei = face_edges[fi,k]
            adj[ei].append(fi)

    primal_len = np.linalg.norm(P[E_pairs[:,1]] - P[E_pairs[:,0]], axis=1)
    dual_len   = np.zeros(Ecnt)
    for ei in range(Ecnt):
        faces = adj[ei]
        if len(faces) != 2:
            dual_len[ei] = 1e-9
        else:
            f0,f1 = faces
            dual_len[ei] = np.linalg.norm(C[f1] - C[f0])

    w = dual_len / np.maximum(primal_len, 1e-14)
    return diags(w), w

def assemble_coclosed_form(d0, star1):
    return (d0.T @ star1).tocsr()   # (V x E)

def solve_harmonic_with_periods(star1, delta, c, period):
    """
    Minimize (1/2) ωᵀ⋆1ω  s.t.  δ ω = 0  and  cᵀ ω = period.
    KKT:
      [⋆1   δᵀ   cᵀ][ω] = [0]
      [δ     0    0][λ]   [0]
      [c     0    0][μ]   [p]
    """
    Ecnt = star1.shape[0]
    Vcnt = delta.shape[0]

    A = star1.tocsc()
    B = delta.tocsc()                     # (V x E)
    C = coo_matrix(c.reshape(1,-1)).tocsr()

    Zvv = csc_matrix((Vcnt, Vcnt))
    Z11 = csc_matrix((1,1))
    Z1v = csc_matrix((Vcnt, 1))           # <-- FIX: (V x 1), not (1 x V)
    Zv1 = csc_matrix((1, Vcnt))           # (1 x V)

    KKT = bmat([[A, B.T, C.T],
                [B, Zvv, Z1v],
                [C, Zv1, Z11]], format="csc")

    rhs = np.zeros(Ecnt + Vcnt + 1)
    rhs[-1] = period

    sol = spsolve(KKT, rhs)
    omega = sol[:Ecnt]
    return omega

def stiffness_from_omegas(star1, omx, omy):
    Kxx = omx @ (star1 @ omx)
    Kyy = omy @ (star1 @ omy)
    Kxy = omx @ (star1 @ omy)
    return Kxx, Kyy, Kxy

# ----------------------------
# One mesh evaluation
# ----------------------------

def one_mesh_eval(Lx, Ly, jitter, ax, ay, seed):
    P, V, E, F, d0, face_edges, face_signs, cx, cy = build_torus_grid(
        Lx, Ly, jitter, ax, ay, seed
    )
    star1, _ = circum_star1(P, E, F, face_edges, face_signs)
    delta = assemble_coclosed_form(d0, star1)

    cx_sp = coo_matrix(cx.reshape(1,-1)).tocsr()
    cy_sp = coo_matrix(cy.reshape(1,-1)).tocsr()

    omx = solve_harmonic_with_periods(star1, delta, cx_sp.toarray().ravel(), 1.0)
    omy = solve_harmonic_with_periods(star1, delta, cy_sp.toarray().ravel(), 1.0)

    Kxx, Kyy, Kxy = stiffness_from_omegas(star1, omx, omy)
    R   = Kxx / max(Kyy, 1e-18)
    sin2 = Kxx / max(Kxx + Kyy, 1e-18)

    return dict(ax=ax, ay=ay,
                Kxx=float(Kxx), Kyy=float(Kyy), Kxy=float(Kxy),
                R=float(R), sin2=float(sin2))

# ----------------------------
# Batch: ~12 meshes total
# ----------------------------

def run_target12(Lx=16, Ly=16, jitter=0.03, outdir="out_target12", seed=2025):
    Path(outdir).mkdir(parents=True, exist_ok=True)

    settings = [
        (1.31, 0.95),
        (1.31, 1.85),
        (1.30, 1.08),
        (1.30, 1.75),
        (1.32, 1.10),
        (1.32, 1.09),
    ]
    meshes_per = 2
    results = []

    print("=== 12-mesh targeting run (circum *1, co-closed periods) ===")
    print(f"Lx={Lx} Ly={Ly} jitter={jitter}  out={outdir}")
    print("Settings (ax, ay):", settings)
    print("Goal: R=Kxx/Kyy ~ 0.30  ->  sin2 = R/(1+R) ~ 0.231\n")

    s0 = seed
    for (ax,ay) in settings:
        row_group = []
        for k in range(meshes_per):
            r = one_mesh_eval(Lx, Ly, jitter, ax, ay, s0 + 97*k)
            row_group.append(r)
            results.append(r)
        g = pd.DataFrame(row_group)
        print(f"(ax,ay)=({ax:.2f},{ay:.2f})  "
              f"sin2 mean={g['sin2'].mean():.3f} ± {g['sin2'].std(ddof=1):.3f}  "
              f"R mean={g['R'].mean():.3f}")

    df = pd.DataFrame(results)
    df.to_csv(Path(outdir,"target12.csv"), index=False)

    df["abs_err"] = np.abs(df["sin2"] - 0.231)
    best = df.loc[df["abs_err"].idxmin()].to_dict()
    print("\n=== Best single mesh (by |sin2-0.231|) ===")
    print(f"ax={best['ax']:.3f}  ay={best['ay']:.3f}  "
          f"sin2={best['sin2']:.3f}  R={best['R']:.3f}")
    print(f"Kxx={best['Kxx']:.6g}  Kyy={best['Kyy']:.6g}  Kxy={best['Kxy']:.6g}")

    agg = df.groupby(["ax","ay"]).agg(
        sin2_mean=("sin2","mean"),
        sin2_std =("sin2","std"),
        R_mean   =("R","mean"),
        R_std    =("R","std")
    ).reset_index()
    agg.to_csv(Path(outdir,"target12_aggregates.csv"), index=False)

    piv = agg.pivot_table(index="ay", columns="ax", values="sin2_mean")
    plt.figure(figsize=(6,4))
    im = plt.imshow(piv.values, origin="lower",
                    extent=[piv.columns.min(), piv.columns.max(),
                            piv.index.min(), piv.index.max()],
                    aspect="auto")
    plt.colorbar(im, label="sin2")
    try:
        X, Y = np.meshgrid(piv.columns.values, piv.index.values)
        plt.contour(X, Y, piv.values, levels=[0.231], colors="k", linewidths=1)
    except Exception:
        pass
    plt.xlabel("ax"); plt.ylabel("ay"); plt.title("sin2 mean over 2 meshes/setting")
    plt.tight_layout()
    plt.savefig(Path(outdir,"sin2_heatmap.png"), dpi=140)

    with open(Path(outdir,"summary.txt"), "w", encoding="utf-8") as f:
        f.write("12-mesh targeting run\n")
        f.write(f"Lx={Lx} Ly={Ly} jitter={jitter}\n")
        f.write(f"settings={settings}\n")
        f.write(f"Best single: ax={best['ax']:.3f} ay={best['ay']:.3f} "
                f"sin2={best['sin2']:.6f} R={best['R']:.6f}\n")

    print(f"\nWrote:\n  - {Path(outdir,'target12.csv')}\n"
          f"  - {Path(outdir,'target12_aggregates.csv')}\n"
          f"  - {Path(outdir,'sin2_heatmap.png')}\n"
          f"  - {Path(outdir,'summary.txt')}")

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lx", type=int, default=16)
    ap.add_argument("--Ly", type=int, default=16)
    ap.add_argument("--jitter", type=float, default=0.05)
    ap.add_argument("--out", type=str, default="out_target12")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()
    run_target12(Lx=args.Lx, Ly=args.Ly, jitter=args.jitter, outdir=args.out, seed=args.seed)

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    main()