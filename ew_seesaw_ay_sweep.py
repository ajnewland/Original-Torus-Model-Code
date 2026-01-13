# ew_seesaw_ay_sweep.py
# Sweep ay, measure A (dx=0) & B (dx=1) classes, and report symmetrized S.
# Works on Windows; calls your existing ew_uneven_suite.py via subprocess.
import argparse, subprocess, sys, re, csv, math, shutil
from pathlib import Path

LINE_RE = re.compile(
    r"shift=\(x=(?P<dx>\d+),y=(?P<dy>\d+)\)\s+sin2=(?P<mean>[0-9\.eE+-]+)\s±\s(?P<std>[0-9\.eE+-]+)\s+condG~(?P<cond>[0-9\.eE+-]+)"
)

def run_cycles(suite_py, py_exe, L, jitter, n, ax, ay, star, ridge):
    """Call ew_uneven_suite.py --mode cycles with dx=0 and dx=1; parse A & B."""
    # Build command (Windows-friendly). We only ask for dx=0 and dx=1 to represent the two classes.
    shifts = "0,0;1,0"
    cmd = [
        py_exe, str(suite_py),
        "--mode", "cycles",
        "--L", str(L),
        "--jitter", str(jitter),
        "--n", str(n),
        "--ax", f"{ax:.12g}",
        "--ay", f"{ay:.12g}",
        "--shifts", shifts,
        "--warp_eps", "0.0",
        "--grade", "0.0",
        "--star", star,
        "--ridge", f"{ridge:.12g}",
    ]
    # Capture output
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        print("ERROR calling ew_uneven_suite.py:\n", e.output)
        raise

    A = B = None
    for line in out.splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        dx = int(m.group("dx"))
        mean = float(m.group("mean"))
        std = float(m.group("std"))
        cond = float(m.group("cond"))
        if dx == 0:
            A = (mean, std, cond)
        elif dx == 1:
            B = (mean, std, cond)

    if A is None or B is None:
        print("Failed to parse A/B from output. Full output:\n", out)
        raise RuntimeError("Parse error: missing dx=0 and/or dx=1 lines.")

    return A, B  # each is (mean, std, cond)

def main():
    p = argparse.ArgumentParser(description="Seesaw ay-sweep driver for DEC cycles.")
    p.add_argument("--suite", default="ew_uneven_suite.py",
                   help="Path to ew_uneven_suite.py (default: ew_uneven_suite.py)")
    p.add_argument("--py", dest="python_exe", default=sys.executable,
                   help="Python executable to run the suite (default: current python)")
    p.add_argument("--L", type=int, default=20)
    p.add_argument("--jitter", type=float, default=0.02)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--ax", type=float, default=2.59)
    p.add_argument("--ay_min", type=float, default=0.76)
    p.add_argument("--ay_max", type=float, default=0.90)
    p.add_argument("--ay_steps", type=int, default=15,
                   help="Number of ay sample points (inclusive endpoints).")
    p.add_argument("--star", choices=["cotan","circum"], default="cotan")
    p.add_argument("--ridge", type=float, default=1e-11)
    p.add_argument("--target", type=float, default=0.231)
    p.add_argument("--out", default="out_seesaw",
                   help="Output folder (CSV will be written here).")
    args = p.parse_args()

    suite_py = Path(args.suite)
    if not suite_py.exists():
        print(f"ERROR: cannot find suite script at: {suite_py.resolve()}")
        sys.exit(1)

    if shutil.which(args.python_exe) is None and not Path(args.python_exe).exists():
        print(f"ERROR: python executable not found: {args.python_exe}")
        sys.exit(1)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "seesaw_ay_sweep.csv"

    # Build ay grid
    if args.ay_steps < 2:
        ay_list = [args.ay_min]
    else:
        step = (args.ay_max - args.ay_min) / (args.ay_steps - 1)
        ay_list = [args.ay_min + i*step for i in range(args.ay_steps)]

    rows = []
    best = None  # (score, ay, Amean, Bmean, Smean)

    print(f"=== Seesaw ay-sweep ===")
    print(f"L={args.L} jitter={args.jitter} n={args.n} (ax,ay)~({args.ax},•) star={args.star} ridge={args.ridge:.1e}")
    print(f"ay in [{args.ay_min},{args.ay_max}] steps={args.ay_steps}  target≈{args.target}")

    for ay in ay_list:
        A, B = run_cycles(suite_py, args.python_exe, args.L, args.jitter, args.n,
                          args.ax, ay, args.star, args.ridge)
        Amean, Astd, Acond = A
        Bmean, Bstd, Bcond = B
        Smean = 0.5*(Amean + Bmean)
        # simple combined std (assuming independence)
        Sstd = 0.5*math.sqrt(Astd*Astd + Bstd*Bstd)

        dAB = Amean - Bmean
        dA = abs(Amean - args.target)
        dB = abs(Bmean - args.target)
        dS = abs(Smean - args.target)

        # score: equalize A/B and hit target with S
        score = abs(dAB) + dS
        if best is None or score < best[0]:
            best = (score, ay, Amean, Bmean, Smean, dAB, dS)

        print(f"ay={ay:.5f}  A(dx=0)={Amean:.6f}±{Astd:.6f}  "
              f"B(dx=1)={Bmean:.6f}±{Bstd:.6f}  S={Smean:.6f}±{Sstd:.6f}  "
              f"|A-B|={abs(dAB):.6f}  |S-θ|={dS:.6f}  condG(A)~{Acond:.3f}  condG(B)~{Bcond:.3f}")

        rows.append([ay, Amean, Astd, Acond, Bmean, Bstd, Bcond, Smean, Sstd, dAB, dS])

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ay","A_mean","A_std","A_condG","B_mean","B_std","B_condG","S_mean","S_std","A_minus_B","abs_S_minus_target"])
        for r in rows:
            w.writerow(r)

    # Print best candidates (by score)
    rows_sorted = sorted(rows, key=lambda r: (abs(r[9]) + r[10]))
    top = rows_sorted[:3]
    print("\n=== Best candidates (minimize |A-B| + |S-target|) ===")
    for ay, Amean, Astd, Acond, Bmean, Bstd, Bcond, Smean, Sstd, dAB, dS in top:
        print(f"ay={ay:.5f}  A={Amean:.6f}  B={Bmean:.6f}  S={Smean:.6f}  "
              f"|A-B|={abs(dAB):.6f}  |S-θ|={dS:.6f}  condG(A)~{Acond:.3f}  condG(B)~{Bcond:.3f}")

    print(f"\nWrote CSV: {csv_path.resolve()}")

if __name__ == "__main__":
    main()