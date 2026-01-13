import argparse, os, itertools
import numpy as np
import pandas as pd

def load_latent(path):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    need = ['ax','ay','z_pred']
    for n in need:
        if n not in cols:
            raise ValueError(f"Input CSV must contain columns: ax, ay, z_pred (found {list(df.columns)})")
    return df.rename(columns={cols['ax']:'ax', cols['ay']:'ay', cols['z_pred']:'z_pred'})

def deform(df0, scale=1.0, tilt_ax=0.0, tilt_ay=0.0, shear=0.0, rotate_deg=0.0, bias=0.0, anis_ax=1.0, anis_ay=1.0):
    # Optional richer deformations: anisotropic rescale, in-plane shear, rotation, plus additive bias in z.
    ax = df0['ax'].to_numpy()
    ay = df0['ay'].to_numpy()
    z0 = df0['z_pred'].to_numpy().copy()

    # anisotropic stretch of coordinates (proxy for cycle-length drift)
    ax = (ax - ax.mean())*anis_ax + ax.mean()
    ay = (ay - ay.mean())*anis_ay + ay.mean()

    # simple shear in the chart: (ax, ay) -> (ax + s*ay, ay)
    if shear != 0.0:
        ax = ax + shear * (ay - ay.mean())

    # small rotation around the chart centroid
    if rotate_deg != 0.0:
        th = np.deg2rad(rotate_deg)
        axc, ayc = ax.mean(), ay.mean()
        dx, dy = ax-axc, ay-ayc
        ax = axc +  np.cos(th)*dx - np.sin(th)*dy
        ay = ayc +  np.sin(th)*dx + np.cos(th)*dy

    # axial tilts in z as linear forms in (ax, ay)
    z = scale*z0 + tilt_ax*(ax - ax.mean()) + tilt_ay*(ay - ay.mean()) + bias
    return z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latent', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scale_min', type=float, default=1.0)
    ap.add_argument('--scale_max', type=float, default=1.6)
    ap.add_argument('--scale_steps', type=int, default=20)
    ap.add_argument('--tilt_ax_min', type=float, default=-0.2)
    ap.add_argument('--tilt_ax_max', type=float, default=0.2)
    ap.add_argument('--tilt_ax_steps', type=int, default=21)
    ap.add_argument('--tilt_ay_min', type=float, default=-0.2)
    ap.add_argument('--tilt_ay_max', type=float, default=0.2)
    ap.add_argument('--tilt_ay_steps', type=int, default=21)
    # richer knobs:
    ap.add_argument('--shear_min', type=float, default=0.0)
    ap.add_argument('--shear_max', type=float, default=0.0)
    ap.add_argument('--shear_steps', type=int, default=1)
    ap.add_argument('--rot_min', type=float, default=0.0)   # degrees
    ap.add_argument('--rot_max', type=float, default=0.0)
    ap.add_argument('--rot_steps', type=int, default=1)
    ap.add_argument('--bias_min', type=float, default=0.0)
    ap.add_argument('--bias_max', type=float, default=0.0)
    ap.add_argument('--bias_steps', type=int, default=1)
    ap.add_argument('--anis_ax_min', type=float, default=1.0)
    ap.add_argument('--anis_ax_max', type=float, default=1.0)
    ap.add_argument('--anis_ax_steps', type=int, default=1)
    ap.add_argument('--anis_ay_min', type=float, default=1.0)
    ap.add_argument('--anis_ay_max', type=float, default=1.0)
    ap.add_argument('--anis_ay_steps', type=int, default=1)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df0 = load_latent(args.latent)

    def grid(vmin, vmax, n):
        return np.linspace(vmin, vmax, n) if n>1 else np.array([vmin])

    scales   = grid(args.scale_min, args.scale_max, args.scale_steps)
    tiltaxs  = grid(args.tilt_ax_min, args.tilt_ax_max, args.tilt_ax_steps)
    tiltays  = grid(args.tilt_ay_min, args.tilt_ay_max, args.tilt_ay_steps)
    shears   = grid(args.shear_min, args.shear_max, args.shear_steps)
    rots     = grid(args.rot_min, args.rot_max, args.rot_steps)
    biases   = grid(args.bias_min, args.bias_max, args.bias_steps)
    anaxs    = grid(args.anis_ax_min, args.anis_ax_max, args.anis_ax_steps)
    anays    = grid(args.anis_ay_min, args.anis_ay_max, args.anis_ay_steps)

    best = None
    rows = []
    for vals in itertools.product(scales, tiltaxs, tiltays, shears, rots, biases, anaxs, anays):
        sc, tx, ty, sh, rt, bz, axs, ays = vals
        z = deform(df0, scale=sc, tilt_ax=tx, tilt_ay=ty, shear=sh, rotate_deg=rt, bias=bz, anis_ax=axs, anis_ay=ays)
        min_abs = float(np.min(np.abs(z)))
        n_neg   = int(np.sum(z<0))
        n_pos   = int(np.sum(z>0))
        flipped = (n_neg>0 and n_pos>0)  # coarse sign-flip
        row = dict(scale=sc, tilt_ax=tx, tilt_ay=ty, shear=sh, rot_deg=rt, bias=bz,
                   anis_ax=axs, anis_ay=ays, min_abs_z=min_abs, n_neg=n_neg, n_pos=n_pos, flip=int(flipped))
        rows.append(row)
        if (best is None) or (min_abs < best['min_abs_z']):
            best = row

    df_out = pd.DataFrame(rows).sort_values('min_abs_z', ascending=True)
    df_out.to_csv(args.out, index=False)
    print("=== Zero-margin scan summary ===")
    print("best (smallest |z|):", {k:best[k] for k in best})
    print(f"[WROTE] {args.out}")

if __name__ == '__main__':
    main()