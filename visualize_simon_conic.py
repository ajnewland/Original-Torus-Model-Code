# visualize_simon_conic.py  (robust, with box expansion + algebraic fallback)
import json, argparse, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

DEFAULT = {
    "r_surface_coeffs": {
        "c0": -8.129166588335748, "c1": 8.893322909257305, "c2": -7.591838650306579,
        "c3": 1.7682152039240624, "c4": -1.9005055341082102, "c5": 0.4503050645366164
    },
    "conic_coeffs": {
        "A": -1.9005055341082102, "B": 0.4503050645366164, "C": 1.7682152039240624,
        "D": 8.893322909257305, "E": -7.591838650306579, "F": -8.429944305825348
    },
    "box": {"ax_min": 2.40, "ax_max": 2.64, "ay_min": 0.72, "ay_max": 0.92},
    "fermions": [
        ["u",2.6225,0.7800],["d",2.6245,0.7900],
        ["s",2.579554620804375,0.8097866548499764],
        ["e",2.5385,0.7400],["mu",2.6064658236168006,0.8186872517066096],
        ["c",2.4883,0.81625],["tau",2.5755,0.8425],["b",2.4765,0.8300],
        ["t",2.6385,0.917701075461875],
    ],
    "nu_basin": [2.5695, 0.73725],
}

def z_surface(ax, ay, C):
    c0,c1,c2,c3,c4,c5 = (C[k] for k in ["c0","c1","c2","c3","c4","c5"])
    return c0 + c1*ax + c2*ay + c3*ax*ax + c4*ax*ay + c5*ay*ay

def F_conic(ax, ay, K):
    A,B,C,D,E,Fc = (K[k] for k in ["A","B","C","D","E","F"])
    return A*ax*ax + B*ax*ay + C*ay*ay + D*ax + E*ay + Fc

def algebraic_conic(ax_grid, K):
    A,B,C,D,E,Fc = (K[k] for k in ["A","B","C","D","E","F"])
    p = B*ax_grid + E
    q = A*ax_grid*ax_grid + D*ax_grid + Fc
    disc = p*p - 4.0*C*q
    ay1 = np.full_like(ax_grid, np.nan, float)
    ay2 = np.full_like(ax_grid, np.nan, float)
    m = disc >= 0
    if np.any(m):
        s = np.sqrt(disc[m]); den = 2.0*C
        ay1[m] = (-p[m] + s)/den
        ay2[m] = (-p[m] - s)/den
    return ay1, ay2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coeffs_json", default="", help="JSON with coeffs/box (optional)")
    ap.add_argument("--nx", type=int, default=401)
    ap.add_argument("--ny", type=int, default=401)
    ap.add_argument("--eps", type=float, default=1e-2, help="contour band around 0")
    ap.add_argument("--expand", type=float, default=0.0,
                    help="fractional box padding, e.g. 0.03 pads by 3% each side")
    ap.add_argument("--out", default="simon_conic.png")
    args = ap.parse_args()

    cfg = DEFAULT.copy()
    if args.coeffs_json:
        cfg.update(json.loads(Path(args.coeffs_json).read_text()))

    Csurf, K, box = cfg["r_surface_coeffs"], cfg["conic_coeffs"], cfg["box"]
    ax_min, ax_max = box["ax_min"], box["ax_max"]
    ay_min, ay_max = box["ay_min"], box["ay_max"]

    # optional padding
    if args.expand > 0:
        dax = (ax_max-ax_min)*args.expand
        day = (ay_max-ay_min)*args.expand
        ax_min -= dax; ax_max += dax
        ay_min -= day; ay_max += day

    ax = np.linspace(ax_min, ax_max, args.nx)
    ay = np.linspace(ay_min, ay_max, args.ny)
    AX, AY = np.meshgrid(ax, ay)
    R = z_surface(AX, AY, Csurf)
    F = F_conic(AX, AY, K)

    print(f"Grid: ax in [{ax_min:.4f},{ax_max:.4f}], ay in [{ay_min:.4f},{ay_max:.4f}]")
    print(f"F min/max on grid: {np.nanmin(F):.4g} / {np.nanmax(F):.4g}")

    plt.figure(figsize=(9,7.5))
    pm = plt.pcolormesh(AX, AY, R, shading="auto", cmap="coolwarm", alpha=0.25)
    plt.colorbar(pm, label="r(ax, ay) (signed)")

    levels = np.array([-args.eps, 0.0, +args.eps])
    cs = plt.contour(AX, AY, F, levels=levels, colors="k", linewidths=[0.8,1.5,0.8])
    # robust zero detection using levels + allsegs
    zero_found = False
    for lev, segs in zip(cs.levels, cs.allsegs):
        if abs(lev) < 1e-12 and len(segs) > 0:
            zero_found = True
            break

    if not zero_found:
        print("No F=0 contour inside box → algebraic fallback.")
        ax_dense = np.linspace(ax_min, ax_max, 2000)
        y1, y2 = algebraic_conic(ax_dense, K)
        shown = False
        for y in (y1, y2):
            mask = (~np.isnan(y)) & (y >= ay_min) & (y <= ay_max)
            if np.any(mask):
                plt.plot(ax_dense[mask], y[mask], "k-", lw=1.6, label="F=0 (algebraic)")
                shown = True
        if not shown:
            print("Algebraic branches exist but do not intersect the current box.")
    else:
        print("F=0 contour segments plotted.")

    # points
    for name, axp, ayp in cfg["fermions"]:
        plt.scatter(axp, ayp, s=30, edgecolor="k", facecolor="white", zorder=3)
        plt.text(axp+0.003, ayp+0.003, name, fontsize=10)
    nbx, nby = cfg["nu_basin"]
    plt.scatter(nbx, nby, s=50, marker="s", edgecolor="k", facecolor="white", zorder=3)
    plt.text(nbx+0.003, nby-0.010, "nu-basin", fontsize=10)

    plt.xlim(ax_min, ax_max); plt.ylim(ay_min, ay_max)
    plt.xlabel("ax"); plt.ylabel("ay")
    plt.title("Simon curve (F=0) over r-surface, with fermions + nu-basin")
    plt.tight_layout(); plt.savefig(args.out, dpi=150)
    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()