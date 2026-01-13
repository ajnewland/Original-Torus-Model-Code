import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt

def load_grid(p):
    df = pd.read_csv(p)
    axv = np.sort(df["ax"].unique()); ayv = np.sort(df["ay"].unique())
    piv = df.pivot(index="ay", columns="ax", values="z_pred").sort_index().sort_index(axis=1)
    Z = piv.values  # (ny, nx)
    return axv, ayv, Z

def local_maxima_interior(Z):
    ny, nx = Z.shape; peaks=[]
    for j in range(1,ny-1):
        for i in range(1,nx-1):
            z=Z[j,i]; nbr=Z[j-1:j+2, i-1:i+2].copy(); nbr[1,1]=-np.inf
            if z>np.max(nbr): peaks.append((i,j,z))
    return peaks

def boundary_candidates(Z):
    ny,nx = Z.shape; cand=[]
    # top/bottom rows
    for i in range(nx):
        cand.append((i,0,Z[0,i])); cand.append((i,ny-1,Z[ny-1,i]))
    # left/right cols
    for j in range(ny):
        cand.append((0,j,Z[j,0])); cand.append((nx-1,j,Z[j,nx-1]))
    # unique them
    seen=set(); out=[]
    for i,j,z in cand:
        if (i,j) not in seen:
            out.append((i,j,z)); seen.add((i,j))
    return out

def min_sep(axv, ayv, i, j, known):
    ax, ay = axv[i], ayv[j]
    if known is None or len(known)==0: return np.inf
    d = np.hypot(known[:,0]-ax, known[:,1]-ay)
    return float(d.min())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--known", default=None)
    ap.add_argument("--min_sep", type=float, default=0.004)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--outdir", required=True)
    args=ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    axv, ayv, Z = load_grid(args.grid)
    known = None
    if args.known and os.path.exists(args.known):
        kd = pd.read_csv(args.known)
        if {"ax","ay"}.issubset(kd.columns):
            known = kd[["ax","ay"]].values

    peaks_int = local_maxima_interior(Z)
    peaks_bnd = boundary_candidates(Z)

    rows=[]
    # interior (type='interior')
    for i,j,z in peaks_int:
        sep = min_sep(axv, ayv, i, j, known)
        if sep >= args.min_sep:
            rows.append({"type":"interior","ax":axv[i],"ay":ayv[j],"z_pred":z,"sep_min":sep})
    # if none, keep the best boundary points so we can steer the box
    if not rows:
        # take top boundary by z
        peaks_bnd = sorted(peaks_bnd, key=lambda t:t[2], reverse=True)[:args.top_k]
        for i,j,z in peaks_bnd:
            sep = min_sep(axv, ayv, i, j, known)
            rows.append({"type":"boundary","ax":axv[i],"ay":ayv[j],"z_pred":z,"sep_min":sep})

    out = pd.DataFrame(rows).sort_values(["type","z_pred"], ascending=[True,False])
    csv_path = os.path.join(args.outdir,"peaks_or_boundary.csv"); out.to_csv(csv_path, index=False)

    # quick figure
    plt.figure(figsize=(7,6))
    extent=[axv.min(),axv.max(),ayv.min(),ayv.max()]
    plt.imshow(Z, origin="lower", aspect="auto", extent=extent)
    if len(out):
        plt.scatter(out["ax"], out["ay"], marker="x", s=20)
    plt.xlabel("ax"); plt.ylabel("ay"); plt.title("z heatmap (x marks interior/boundary picks)")
    plt.colorbar(); plt.tight_layout()
    fig_path=os.path.join(args.outdir,"marked.png"); plt.savefig(fig_path, dpi=160); plt.close()

    print(f"[DONE] wrote {csv_path}")
    print(f"[DONE] wrote {fig_path}")
    if len(out) and (out["type"]=="boundary").all():
        print("[INFO] Only boundary candidates found: expand the box in that direction.")

if __name__=="__main__":
    main()