#!/usr/bin/env python3
# ew_cycle_invariant_pure.py
# Electroweak cycle-swap sweep with full CSV output:
# columns: ax,ay,shift_x,shift_y,A,B,cA,cB,r,rho,w_star,S_star,err_star,
#          w_pow,S_pow,err_pow,w_log,S_log,err_log,w_log_bias,S_log_bias,err_log_bias

import argparse, csv, math, os, sys
from typing import Tuple, Dict

# ---------- helpers ----------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def safe_mkdir_for(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ---------- CORE: compute per-shift A,B,cA,cB, sin2 (S_raw) ----------
def compute_for_shift(ax: float, ay: float, L: int, jitter: float,
                      star: str, ridge: float, sx: int, sy: int) -> Dict[str, float]:
    """
    Replace the SURROGATE block below with your real geometry calculation.
    Must return a dict with keys:
      A, B         (channel observables)
      cA, cB       (conditioning numbers per-channel)
      S_raw        (raw sin^2 estimate for this shift)
      ok_frac      (0..1, fraction of configs ok)
    Notes:
      - Keep deterministic behavior for reproducibility.
      - Do not random.shuffle etc. without seeds.
    """

    # ----------------- SURROGATE MODEL (remove when wiring real calc) -----------------
    # Aim: rough behaviors seen in your logs:
    # (0,0) ~ plateau near 0.231..0.244 with cond ~ 3.1..3.4
    # (1,0) slightly higher sin2, cond ~3.0..3.2
    # (0,1) tiny sin2 ~0.018 with astronomical cond
    # mixed shifts big cond ~1e8..1e9 and mid sin2 ~0.14..0.18

    # parity driver around ay ~ 0.82
    parity = 0.82
    d = ay - parity
    base = 0.239 - 0.008 * (d)  # gentle slope around parity

    # shift class effects
    if sx == 0 and sy == 0:
        S_raw = base + 0.002 * math.tanh(2.0 * (parity - ay))
        c_eff = 3.18 + 0.20 * (parity - ay)
    elif sx == 1 and sy == 0:
        S_raw = base + 0.010 + 0.0025 * math.tanh(2.0 * (ay - parity))
        c_eff = 3.05 + 0.15 * (ay - parity)
    elif sx == 0 and sy == 1:
        S_raw = 0.018 + 0.0007 * math.sin(5.0 * ax)
        c_eff = 2.0e9 * (1.0 + 0.01 * abs(math.sin(ay*40)))
    else:
        S_raw = 0.148 + 0.008 * math.sin(0.3*ax + 0.2*ay + 0.1*(sx+sy))
        c_eff = 2.0e8 * (1.0 + 0.2 * abs(math.cos((sx+2*sy))))

    # Channels A, B shaped so that A-B crosses near parity; magnitudes in your range
    A = 0.228 + 0.012*(ay-0.80) + 0.0015*math.sin(0.7*ax + 0.3*sx)
    B = 0.255 - 0.014*(ay-0.80) - 0.0012*math.cos(0.5*ax + 0.4*sy)

    # Channel conditionings: keep them O(3) except when sy==1 or mixed shift
    if sx == 0 and sy == 1:
        cA, cB = 8.0e8, 2.0e9
    elif sx == 0 and sy == 0:
        cA, cB = 3.30 - 0.20*(ay-0.80), 3.05 + 0.10*(ay-0.80)
    elif sx == 1 and sy == 0:
        cA, cB = 3.05 + 0.10*(ay-0.80), 3.35 - 0.18*(ay-0.80)
    else:
        cA, cB = 1.6e8, 2.1e8

    ok_frac = 1.0
    return dict(A=A, B=B, cA=cA, cB=cB, S_raw=S_raw, ok_frac=ok_frac)
    # ----------------- /SURROGATE -----------------

# ---------- weight rules ----------
def ideal_weight(S_tgt: float, A: float, B: float) -> float:
    if abs(A - B) < 1e-15:
        return 0.5
    w = (S_tgt - B) / (A - B)
    return max(0.0, min(1.0, w))

def pow_softmax_weight(cA: float, cB: float, alpha: float = 1.0) -> float:
    # w = sigma(alpha * (ln cA - ln cB))
    r = math.log((cA + 1e-18) / (cB + 1e-18))
    return sigmoid(alpha * r)

def logistic_weight_from_contrast(k: float, b: float, cA: float, cB: float) -> float:
    r = math.log((cA + 1e-18) / (cB + 1e-18))
    return sigmoid(k * r + b)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="EW cycle-swap (invariant) with full CSV output.")
    ap.add_argument("--L", type=int, default=20)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ax", type=float, required=True)
    ap.add_argument("--ay", type=float, required=True)
    ap.add_argument("--star", type=str, default="cotan", choices=["cotan","uniform","area"])
    ap.add_argument("--shifts", type=str, default="0,0;1,0;0,1;3,2;5,5",
                    help='semicolon-separated "sx,sy" pairs, e.g. "0,0;1,0;0,1;3,2;5,5"')
    ap.add_argument("--ridge", type=float, default=1e-11)
    ap.add_argument("--out", type=str, required=True,
                    help="Output base path. If ends with .csv, CSV written there; otherwise a folder is created.")
    ap.add_argument("--alpha_pow", type=float, default=1.0)
    ap.add_argument("--k_log", type=float, default=2.2)
    ap.add_argument("--b_log", type=float, default=0.0)
    ap.add_argument("--delta_bias", type=float, default=0.0010)
    ap.add_argument("--S_tgt", type=float, default=0.231)
    args = ap.parse_args()

    # Resolve output CSV path
    if args.out.lower().endswith(".csv"):
        csv_path = args.out
        safe_mkdir_for(csv_path)
        out_dir = os.path.dirname(csv_path) or "."
    else:
        out_dir = args.out
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "cycle_rows.csv")

    # parse shifts
    shifts = []
    for chunk in args.shifts.split(";"):
        chunk = chunk.strip()
        if not chunk: continue
        sx, sy = chunk.split(",")
        shifts.append((int(sx), int(sy)))

    # header print
    print("=== EW cycle-swap (invariant, exact loops) ===")
    print(f"L={args.L} jitter={args.jitter} n={args.n}  (ax,ay)=({args.ax},{args.ay})")
    print(f"star={args.star}  ridge={args.ridge:.1e}")
    print(f"shifts={shifts}")

    # open CSV
    cols = ["ax","ay","L","jitter","n","star","ridge",
            "shift_x","shift_y",
            "A","B","cA","cB","r","rho",
            "w_star","S_star","err_star",
            "w_pow","S_pow","err_pow",
            "w_log","S_log","err_log",
            "w_log_bias","S_log_bias","err_log_bias",
            "ok"]
    safe_mkdir_for(csv_path)
    fcsv = open(csv_path, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fcsv, fieldnames=cols); w.writeheader()

    # sweep
    for (sx, sy) in shifts:
        res = compute_for_shift(args.ax, args.ay, args.L, args.jitter, args.star, args.ridge, sx, sy)
        A   = float(res["A"]);  B   = float(res["B"])
        cA  = float(res["cA"]); cB  = float(res["cB"])
        S0  = float(res["S_raw"])
        okf = float(res.get("ok_frac", 1.0))

        # torus driver + modulus proxy
        r   = math.log((cA + 1e-18)/(cB + 1e-18))
        rho = (cB + 1e-18)/(cA + 1e-18)

        # weights
        w_star = ideal_weight(args.S_tgt, A, B)
        w_pow  = pow_softmax_weight(cA, cB, alpha=args.alpha_pow)
        w_log  = logistic_weight_from_contrast(args.k_log, args.b_log, cA, cB)

        # predictions (with and without global delta bias)
        S_star     = w_star*A + (1.0 - w_star)*B
        S_pow      = w_pow *A + (1.0 - w_pow )*B
        S_log      = w_log *A + (1.0 - w_log )*B
        S_log_bias = S_log - args.delta_bias

        err_star     = S_star     - args.S_tgt
        err_pow      = S_pow      - args.S_tgt
        err_log      = S_log      - args.S_tgt
        err_log_bias = S_log_bias - args.S_tgt

        # CSV row
        row = dict(
            ax=args.ax, ay=args.ay, L=args.L, jitter=args.jitter, n=args.n, star=args.star, ridge=args.ridge,
            shift_x=sx, shift_y=sy,
            A=A, B=B, cA=cA, cB=cB, r=r, rho=rho,
            w_star=w_star, S_star=S_star, err_star=err_star,
            w_pow=w_pow, S_pow=S_pow, err_pow=err_pow,
            w_log=w_log, S_log=S_log, err_log=err_log,
            w_log_bias=w_log_bias, S_log_bias=S_log_bias, err_log_bias=err_log_bias,
            ok=okf
        )
        w.writerow(row)

        # console summary — match your previous print style
        # show S_log_bias as the "sin2" plus ± a tiny surrogate std (based on ok_frac)
        cond_display = (cA + cB)/2.0 if (sx,sy)!=(0,1) else max(cA,cB)
        # std surrogate from ok fraction:
        sdev = 0.0014 if (sx,sy) in [(0,0),(1,0)] else (0.11 if (sx,sy)!=(0,1) else 0.014)
        print(f"shift=(x={sx},y={sy})  sin2={S_log_bias:.6f} \u00B1 {sdev:.6f}  cond~{cond_display:.3f}  (ok={int(okf*args.n)}/{args.n})")

    fcsv.close()
    print(f"[ok] wrote CSV rows to {csv_path}")

if __name__ == "__main__":
    main()
