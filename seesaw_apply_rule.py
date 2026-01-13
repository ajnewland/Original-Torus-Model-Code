#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv, math, os, sys

# Learned from your fit:
K = 1.9841
B0 = -0.0048
DELTA = 0.008605
TARGET = 0.231

def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0/(1.0+ez)
    ez = math.exp(z)
    return ez/(1.0+ez)

def get(row, names, cast=float, default=None):
    for n in names:
        if n in row and row[n] not in ("", "nan", "NaN", None):
            try:
                return cast(row[n])
            except Exception:
                pass
    return default

def main():
    if len(sys.argv) < 2:
        print("Usage: python seesaw_apply_rule.py <path\\to\\csv>")
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(2)

    rows = []
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)

    out = []
    for r in rows:
        A = get(r, ["A","A_mean"])
        B = get(r, ["B","B_mean"])
        cA = get(r, ["A_condG","cA","condG(A)"])
        cB = get(r, ["B_condG","cB","condG(B)"])
        if None in (A,B,cA,cB) or cA<=0 or cB<=0:
            continue
        rlog = math.log(cA) - math.log(cB)
        w = sigmoid(K*rlog + B0)
        S_pred = w*A + (1.0 - w)*B - DELTA
        out.append({
            "ay": get(r, ["ay"], float, float("nan")),
            "A": A, "B": B, "A_condG": cA, "B_condG": cB,
            "w": w, "S_pred": S_pred, "S_pred_minus_target": S_pred - TARGET
        })

    if not out:
        print("No valid rows parsed (need A,B,A_condG,B_condG).")
        sys.exit(3)

    out_path = os.path.join(os.path.dirname(path) or ".", "seesaw_apply_rule_out.csv")
    with open(out_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wr.writeheader()
        for r in out:
            wr.writerow(r)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()