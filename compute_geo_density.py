import os, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def load_grid_csv(path):
    df = pd.read_csv(path)
    cols = set(c.lower() for c in df.columns)
    axcol = next(c for c in df.columns if c.lower() == "ax")
    aycol = next(c for c in df.columns if c.lower() == "ay")
    if "z_pred" in df.columns: zcol = "z_pred"
    elif "z" in df.columns:   zcol = "z"
    else: raise ValueError("CSV must contain 'z' or 'z_pred' column.")
    axu = np.array(sorted(df[axcol].unique()))
    ayu = np.array(sorted(df[aycol].unique()))
    Nx, Ny = len(axu), len(ayu)
    if len(df) != Nx*Ny:
        # try robust reindex
        piv = df.pivot_table(index=aycol, columns=axcol, values=zcol, aggfunc="mean")
        piv = piv.sort_index().sort_index(axis=1)
        axu = piv.columns.values
        ayu = piv.index.values
        Z = piv.values
    else:
        order = np.lexsort((df[axcol].values, df[aycol].values))
        dfo = df.iloc[order]
        Z = dfo[zcol].values.reshape(Ny, Nx)
    return axu, ayu, Z

def second_derivative(arr, h, axis=1, stencil=3):
    # arr: (Ny,Nx), axis=0 => y, axis=1 => x
    if stencil == 3:
        # central second derivative (3-point): (f[i+1]-2f[i]+f[i-1]) / h^2
        kern = np.array([1, -2, 1]) / (h*h)
        pad = 1
    elif stencil == 5:
        # 5-point 2nd deriv: (-1,16,-30,16,-1)/(12 h^2)
        kern = np.array([-1, 16, -30, 16, -1]) / (12*h*h)
        pad = 2
    else:
        raise ValueError("--stencil must be 3 or 5")
    if axis == 0: arr = arr
    else:         arr = arr.T
    A = np.pad(arr, ((pad,pad),(0,0)), mode="edge")  # pad along axis
    out = np.zeros_like(arr)
    # conv along first dimension
    for i in range(arr.shape[0]):
        s = 0.0
        for k, c in enumerate(kern):
            s += c * A[i + k, :]
        out[i, :] = s
    if axis == 0: return out
    else:         return out.T

def label_components(mask):
    # 4-neighbor BFS labeling
    H, W = mask.shape
    labels = np.zeros((H,W), dtype=np.int32)
    comp = []
    cid = 0
    for r in range(H):
        for c in range(W):
            if mask[r,c] and labels[r,c] == 0:
                cid += 1
                q=[(r,c)]; labels[r,c]=cid; sz=0; rs=0; cs=0
                while q:
                    y,x = q.pop()
                    sz+=1; rs+=y; cs+=x
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        yy,xx = y+dy, x+dx
                        if 0<=yy<H and 0<=xx<W and mask[yy,xx] and labels[yy,xx]==0:
                            labels[yy,xx]=cid; q.append((yy,xx))
                comp.append({"label":cid,"area_cells":sz,"cy":rs/sz,"cx":cs/sz})
    return labels, comp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--stencil", type=int, default=5, choices=[3,5])
    ap.add_argument("--sigma_thresh", type=float, default=1.0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ax, ay, Z = load_grid_csv(args.grid)

    # spacings
    if len(ax) < 2 or len(ay) < 2:
        raise ValueError("Grid requires at least 2 unique ax & ay values.")
    hx = float(np.mean(np.diff(ax)))
    hy = float(np.mean(np.diff(ay)))

    Zxx = second_derivative(Z, hx, axis=1, stencil=args.stencil)
    Zyy = second_derivative(Z, hy, axis=0, stencil=args.stencil)
    lap = Zxx + Zyy

    # normalize -> z-score
    mu, sd = float(np.nanmean(lap)), float(np.nanstd(lap))
    if sd == 0: sd = 1.0
    lap_z = (lap - mu) / sd

    # rho_geo = positive part of Laplacian
    rho = np.clip(lap, 0, None)

    # threshold on z-score
    thr_mask = lap_z >= args.sigma_thresh
    labels, comps = label_components(thr_mask)

    # write stats
    pd.DataFrame({
        "ax_min":[ax.min()], "ax_max":[ax.max()],
        "ay_min":[ay.min()], "ay_max":[ay.max()],
        "hx":[hx], "hy":[hy],
        "lap_mean":[mu], "lap_std":[sd],
        "rho_mean":[float(np.mean(rho))],
        "rho_pos_area_frac":[float(np.mean(rho>0))],
        "n_components":[len(comps)]
    }).to_csv(os.path.join(args.outdir,"laplacian_stats.csv"), index=False)

    if comps:
        pd.DataFrame(comps).sort_values("area_cells", ascending=False)\
            .to_csv(os.path.join(args.outdir,"rho_geo_components.csv"), index=False)

    # plots
    plt.figure(); plt.imshow(lap, origin="lower", aspect="auto")
    plt.colorbar(); plt.title("Laplacian Δz")
    plt.savefig(os.path.join(args.outdir,"laplacian_map.png"), dpi=150); plt.close()

    plt.figure(); plt.imshow(rho, origin="lower", aspect="auto")
    plt.colorbar(); plt.title("ρ_geo = max(0, Δz)")
    plt.savefig(os.path.join(args.outdir,"rho_geo_map.png"), dpi=150); plt.close()

    plt.figure(); plt.imshow(thr_mask, origin="lower", aspect="auto")
    plt.title(f"Mask(Δz z-score ≥ {args.sigma_thresh})")
    plt.savefig(os.path.join(args.outdir,"rho_geo_threshold_mask.png"), dpi=150); plt.close()

    print(f"[DONE] Wrote {args.outdir}")

if __name__ == "__main__":
    main()