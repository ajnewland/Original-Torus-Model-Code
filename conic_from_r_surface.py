# conic_from_r_surface.py
import argparse, json, math
import numpy as np
import pandas as pd
from pathlib import Path

def fit_r_surface(df):
    # Expect columns: ax, ay, r  (string columns are OK; we'll coerce)
    ax = pd.to_numeric(df["ax"], errors="coerce").to_numpy()
    ay = pd.to_numeric(df["ay"], errors="coerce").to_numpy()
    rr = pd.to_numeric(df["r"],  errors="coerce").to_numpy()
    m = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(rr)
    ax, ay, rr = ax[m], ay[m], rr[m]
    if ax.size < 6:
        raise ValueError(f"Need >=6 points to fit quadratic; got {ax.size}.")
    X = np.column_stack([np.ones_like(ax), ax, ay, ax*ay, ax*ax, ay*ay])
    # Least squares
    coef, *_ = np.linalg.lstsq(X, rr, rcond=None)
    c0, c1, c2, c3, c4, c5 = coef.tolist()
    return (c0, c1, c2, c3, c4, c5), (ax.min(), ax.max(), ay.min(), ay.max())

def sample_conic(A,B,C,D,E,F, x_min, x_max, n=400):
    xs = np.linspace(x_min, x_max, n)
    pts = []
    for x in xs:
        # Quadratic in y:  B y^2 + (C x + E) y + (A x^2 + D x + F) = 0
        qa = B
        qb = C*x + E
        qc = A*x*x + D*x + F
        if abs(qa) < 1e-14:
            # Degenerates to linear in y
            if abs(qb) < 1e-14:
                continue
            y = -qc/qb
            pts.append((x, y))
            continue
        disc = qb*qb - 4*qa*qc
        if disc < 0:
            continue
        sqrt_d = math.sqrt(max(0.0, disc))
        y1 = (-qb + sqrt_d)/(2*qa)
        y2 = (-qb - sqrt_d)/(2*qa)
        pts.append((x, y1))
        if abs(y2 - y1) > 1e-12:
            pts.append((x, y2))
    return pts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", required=True, help="CSV with columns ax,ay,r (e.g., latent_z_merged2.csv)")
    ap.add_argument("--sin2", type=float, default=0.231)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--beta",  type=float, required=True)
    ap.add_argument("--out_csv", required=True, help="Where to write sampled conic points")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.latent).applymap(lambda v: v.strip() if isinstance(v,str) else v)
    if not set(["ax","ay","r"]).issubset(df.columns):
        raise SystemExit(f"Input {args.latent} must have columns ax, ay, r. Got: {list(df.columns)}")

    (c0,c1,c2,c3,c4,c5), (ax_min,ax_max,ay_min,ay_max) = fit_r_surface(df)
    r_tgt = (args.sin2 - args.alpha) / args.beta

    # Conic coefficients: A x^2 + B y^2 + C x y + D x + E y + F = 0
    A,B,C,D,E,F = c4, c5, c3, c1, c2, (c0 - r_tgt)

    # Sample within (slightly expanded) data box
    pad_x = 0.05*(ax_max - ax_min + 1e-9)
    x0, x1 = ax_min - pad_x, ax_max + pad_x
    pts = sample_conic(A,B,C,D,E,F, x0, x1, n=800)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pts, columns=["ax","ay"]).to_csv(out, index=False)

    print("[OK] r-surface fit:")
    print(f"  r(ax,ay) = {c0:.6f} + {c1:.6f} ax + {c2:.6f} ay + {c3:.6f} ax*ay + {c4:.6f} ax^2 + {c5:.6f} ay^2")
    print(f"[OK] r_target = (sin2 - alpha)/beta = ({args.sin2} - {args.alpha})/{args.beta} = {r_tgt:.9f}")
    print("[OK] weak-angle conic (implicit):")
    print(f"  {A:.6f} ax^2 + {B:.6f} ay^2 + {C:.6f} ax*ay + {D:.6f} ax + {E:.6f} ay + {F:.6f} = 0")
    print(f"[OK] wrote sampled curve with {len(pts)} points -> {out}")
    if args.report:
        rep = {
            "r_surface_coeffs": {"c0":c0,"c1":c1,"c2":c2,"c3":c3,"c4":c4,"c5":c5},
            "conic_coeffs": {"A":A,"B":B,"C":C,"D":D,"E":E,"F":F},
            "r_target": r_tgt,
            "box": {"ax_min":ax_min,"ax_max":ax_max,"ay_min":ay_min,"ay_max":ay_max},
            "sin2": args.sin2, "alpha": args.alpha, "beta": args.beta
        }
        rep_path = out.with_suffix(".json")
        rep_path.write_text(json.dumps(rep, indent=2))
        print(f"[OK] wrote {rep_path}")
if __name__ == "__main__":
    main()