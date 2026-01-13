import os, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def load_grid(path):
    df = pd.read_csv(path)
    axcol = next(c for c in df.columns if c.lower()=="ax")
    aycol = next(c for c in df.columns if c.lower()=="ay")
    zcol = "z_pred" if "z_pred" in df.columns else "z"
    axu = np.array(sorted(df[axcol].unique()))
    ayu = np.array(sorted(df[aycol].unique()))
    Nx, Ny = len(axu), len(ayu)
    if len(df) != Nx*Ny:
        piv = df.pivot_table(index=aycol, columns=axcol, values=zcol, aggfunc="mean")
        piv = piv.sort_index().sort_index(axis=1)
        axu, ayu, Z = piv.columns.values, piv.index.values, piv.values
    else:
        order = np.lexsort((df[axcol].values, df[aycol].values))
        dfo = df.iloc[order]
        Z = dfo[zcol].values.reshape(Ny, Nx)
    return axu, ayu, Z

def infer_w_from_z(Z):
    mu = float(np.nanmean(Z)); sd = float(np.nanstd(Z))
    if sd == 0: sd = 1.0
    Zs = (Z - mu)/sd
    return 1.0/(1.0 + np.exp(-Zs))

def label_components(mask):
    H,W = mask.shape
    labels = np.zeros((H,W), dtype=np.int32)
    comps = []; cid=0
    for r in range(H):
        for c in range(W):
            if mask[r,c] and labels[r,c]==0:
                cid+=1; q=[(r,c)]; labels[r,c]=cid
                sz=0
                while q:
                    y,x = q.pop(); sz+=1
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        yy,xx=y+dy,x+dx
                        if 0<=yy<H and 0<=xx<W and mask[yy,xx] and labels[yy,xx]==0:
                            labels[yy,xx]=cid; q.append((yy,xx))
                comps.append({"label":cid,"area_cells":sz})
    return labels, comps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--phi_thresh", type=float, default=0.5)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ax, ay, Z = load_grid(args.grid)
    w = infer_w_from_z(Z)
    phi = 4.0 * w * (1.0 - w)

    # summary
    comps_mask = phi >= args.phi_thresh
    labels, comps = label_components(comps_mask)
    summary = {
        "phi_mean": float(np.mean(phi)),
        "phi_med": float(np.median(phi)),
        "area_fraction_phi_ge_thresh": float(np.mean(comps_mask)),
        "n_components": len(comps)
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.outdir,"phi_summary.csv"), index=False)
    if comps:
        pd.DataFrame(comps).sort_values("area_cells", ascending=False)\
            .to_csv(os.path.join(args.outdir,"phi_components.csv"), index=False)

    # plots
    plt.figure(); plt.imshow(phi, origin="lower", aspect="auto")
    plt.colorbar(); plt.title("φ = 4 w (1-w)")
    plt.savefig(os.path.join(args.outdir,"phi_map.png"), dpi=150); plt.close()

    plt.figure(); plt.imshow(comps_mask, origin="lower", aspect="auto")
    plt.title(f"Mask(φ ≥ {args.phi_thresh})")
    plt.savefig(os.path.join(args.outdir,"phi_mask.png"), dpi=150); plt.close()

    print(f"[DONE] Wrote {args.outdir}")

if __name__ == "__main__":
    main()