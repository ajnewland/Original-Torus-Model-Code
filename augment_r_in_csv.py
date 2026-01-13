#!/usr/bin/env python3
# augment_r_in_csv.py — add r=ln(cA/cB) and rho=cB/cA if missing
import csv, sys, math, os

def tofloat(x):
    try: return float(x)
    except: return float("nan")

if len(sys.argv) < 3:
    print("usage: python augment_r_in_csv.py in.csv out.csv", file=sys.stderr)
    sys.exit(1)

inp, outp = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(inp, newline="", encoding="utf-8")))
if not rows:
    print("no rows", file=sys.stderr); sys.exit(2)

fieldnames = rows[0].keys()
need = {"cA","cB"}
if not need.issubset(fieldnames):
    print(f"missing columns {need - set(fieldnames)} in {inp}", file=sys.stderr); sys.exit(3)

has_r   = "r"   in fieldnames
has_rho = "rho" in fieldnames
new_fields = list(fieldnames)
if not has_r:   new_fields.append("r")
if not has_rho: new_fields.append("rho")

with open(outp, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=new_fields)
    w.writeheader()
    for row in rows:
        cA = tofloat(row["cA"]); cB = tofloat(row["cB"])
        if not has_r:
            r = math.log((cA + 1e-18)/(cB + 1e-18)) if math.isfinite(cA) and math.isfinite(cB) else ""
            row["r"] = r
        if not has_rho:
            rho = (cB + 1e-18)/(cA + 1e-18) if math.isfinite(cA) and math.isfinite(cB) else ""
            row["rho"] = rho
        w.writerow(row)

print(f"[ok] wrote {outp}")
