# make_three_neutrino_local_grids.py
# Build local (ax, ay) grids around three neutrino lock targets, evaluate z_pred, and save CSVs.

import argparse, csv, os, math

def read_latent(path):
    # Expect columns like: source_path, ax, ay, z, r  (we really need ax, ay, z)
    import csv
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                ax = float(row.get('ax', ''))
                ay = float(row.get('ay', ''))
                # accept z or z_pred
                z  = row.get('z', None)
                if z in (None, ''):
                    z = row.get('z_pred', None)
                if z in (None, ''):
                    continue
                z = float(z)
                rows.append((ax, ay, z))
            except:
                pass
    if not rows:
        raise RuntimeError(f'No (ax, ay, z) rows found in latent file: {path}')
    return rows

def fit_quad(ax_list, ay_list, z_list):
    # Fit z = c0 + c1*ax + c2*ay + c3*ax^2 + c4*ax*ay + c5*ay^2  (least squares)
    import numpy as np
    ax_arr = np.array(ax_list)
    ay_arr = np.array(ay_list)
    z_arr  = np.array(z_list)
    X = np.column_stack([
        np.ones_like(ax_arr),
        ax_arr,
        ay_arr,
        ax_arr**2,
        ax_arr*ay_arr,
        ay_arr**2
    ])
    coef, *_ = np.linalg.lstsq(X, z_arr, rcond=None)
    return coef

def z_predict(ax, ay, coef):
    import numpy as np
    X = np.array([1.0, ax, ay, ax*ax, ax*ay, ay*ay])
    return float(X @ coef)

def ensure_dir(p):
    d = os.path.dirname(os.path.abspath(p))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def write_csv(path, rows, header):
    ensure_dir(path)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latent', required=True, help='latent_z_merged2.csv (or similar) with ax, ay, z')
    ap.add_argument('--outdir', required=True, help='output folder')
    ap.add_argument('--halfspan', type=float, default=0.01, help='half extent around center (ax & ay)')
    ap.add_argument('--step', type=float, default=0.0005, help='grid step')
    # Three centers & targets (defaults are your latest best)
    ap.add_argument('--ax1', type=float, default=2.5795)
    ap.add_argument('--ay1', type=float, default=0.7405)
    ap.add_argument('--zt1', type=float, default=-1.4518)

    ap.add_argument('--ax2', type=float, default=2.55875)
    ap.add_argument('--ay2', type=float, default=0.73375)
    ap.add_argument('--zt2', type=float, default=-1.45165)

    ap.add_argument('--ax3', type=float, default=2.57025)
    ap.add_argument('--ay3', type=float, default=0.7375)
    ap.add_argument('--zt3', type=float, default=-1.4515)

    args = ap.parse_args()

    latent_rows = read_latent(args.latent)
    ax_list = [r[0] for r in latent_rows]
    ay_list = [r[1] for r in latent_rows]
    z_list  = [r[2] for r in latent_rows]
    coef = fit_quad(ax_list, ay_list, z_list)

    # grid builder
    def build_grid(center_ax, center_ay, z_target, label):
        rows = []
        ax_min = center_ax - args.halfspan
        ax_max = center_ax + args.halfspan
        ay_min = center_ay - args.halfspan
        ay_max = center_ay + args.halfspan

        # snap inclusive ranges
        def frange(a, b, step):
            n = int(round((b - a) / step))
            return [a + i*step for i in range(n+1)]

        ax_vals = frange(ax_min, ax_max, args.step)
        ay_vals = frange(ay_min, ay_max, args.step)

        for axv in ax_vals:
            for ayv in ay_vals:
                zp = z_predict(axv, ayv, coef)
                dz = zp - z_target
                rows.append({
                    'label': label,
                    'center_ax': center_ax,
                    'center_ay': center_ay,
                    'ax': axv,
                    'ay': ayv,
                    'z_target': z_target,
                    'z_pred': zp,
                    'dz': dz,
                    'abs_dz': abs(dz),
                })
        return rows

    all_rows = []
    g1 = build_grid(args.ax1, args.ay1, args.zt1, 'nu_target_1')
    g2 = build_grid(args.ax2, args.ay2, args.zt2, 'nu_target_2')
    g3 = build_grid(args.ax3, args.ay3, args.zt3, 'nu_target_3')
    all_rows.extend(g1); all_rows.extend(g2); all_rows.extend(g3)

    # Write individual grids
    write_csv(os.path.join(args.outdir, 'nu_target_1_grid.csv'), g1,
              ['label','center_ax','center_ay','ax','ay','z_target','z_pred','dz','abs_dz'])
    write_csv(os.path.join(args.outdir, 'nu_target_2_grid.csv'), g2,
              ['label','center_ax','center_ay','ax','ay','z_target','z_pred','dz','abs_dz'])
    write_csv(os.path.join(args.outdir, 'nu_target_3_grid.csv'), g3,
              ['label','center_ax','center_ay','ax','ay','z_target','z_pred','dz','abs_dz'])

    # Write merged CSV (this is the one you wanted populated)
    merged_path = os.path.join(args.outdir, 'neutrino_three_locks_local_grids.csv')
    write_csv(merged_path, all_rows,
              ['label','center_ax','center_ay','ax','ay','z_target','z_pred','dz','abs_dz'])

    # Also print the best points per grid
    def best_of(grid):
        return min(grid, key=lambda r: r['abs_dz'])

    b1 = best_of(g1)
    b2 = best_of(g2)
    b3 = best_of(g3)
    print('[OK] Local grids written.')
    print(f" Best near #1: ax={b1['ax']:.6f} ay={b1['ay']:.6f}  |dz|={b1['abs_dz']:.3e}")
    print(f" Best near #2: ax={b2['ax']:.6f} ay={b2['ay']:.6f}  |dz|={b2['abs_dz']:.3e}")
    print(f" Best near #3: ax={b3['ax']:.6f} ay={b3['ay']:.6f}  |dz|={b3['abs_dz']:.3e}")
    print(f" Output folder: {args.outdir}")

if __name__ == '__main__':
    main()