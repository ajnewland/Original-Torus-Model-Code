import argparse
import csv
import json
import math
import os
from typing import Optional, Dict, Any

def load_locks(path: str) -> Dict[str, Dict[str, Any]]:
    req = ["species", "m_GeV"]
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        missing = [c for c in req if c not in r.fieldnames]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        for row in r:
            sp = row["species"].strip()
            if not sp:
                continue
            rows[sp] = row
    return rows

def safe_float(d: Dict[str, Any], key: str) -> Optional[float]:
    try:
        return float(d.get(key, ""))
    except Exception:
        return None

def pick_plateau_from_seesaw(seesaw_csv: str) -> Optional[float]:
    """
    Attempt to extract the plateau S ~ sin^2(theta_W)_geo from a seesaw sweep file.
    We look for a column named like 'S', 'sin2', 'S_geo', or 'plateau'.
    Then we take a robust central tendency (median of the column).
    """
    if not seesaw_csv or not os.path.exists(seesaw_csv):
        return None

    candidates = ["S", "sin2", "S_geo", "plateau", "S_pred", "S_clean"]
    values = []
    with open(seesaw_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        picks = [c for c in candidates if c in cols]
        if not picks:
            # try to guess: any column in [0,1] with small spread might be S
            rows = list(r)
            if not rows:
                return None
            # recompute min-spread bounded-in-[0,1] column
            best_name, best_spread = None, 10.0
            for c in cols:
                try:
                    vals = [float(x[c]) for x in rows if x.get(c, "") != ""]
                    if not vals:
                        continue
                    if min(vals) >= 0.0 and max(vals) <= 1.0:
                        spread = max(vals) - min(vals)
                        if spread < best_spread:
                            best_spread = spread
                            best_name = c
                except Exception:
                    pass
            if best_name is None:
                return None
            for row in rows:
                try:
                    values.append(float(row[best_name]))
                except Exception:
                    pass
        else:
            for row in r:
                try:
                    values.append(float(row[picks[0]]))
                except Exception:
                    pass

    if not values:
        return None
    values.sort()
    mid = len(values)//2
    median = values[mid] if len(values)%2==1 else 0.5*(values[mid-1]+values[mid])
    return median

def make_report(outdir: str, data: Dict[str, Any]) -> None:
    os.makedirs(outdir, exist_ok=True)
    # CSV
    csv_path = os.path.join(outdir, "boson_unification_checks.csv")
    header = [
        "mW_GeV","mZ_GeV","mH_GeV",
        "sin2_mass","sin2_geo","delta_sin2",
        "rho_param","rho_minus_1",
        "vW_band_rel","vZ_band_rel",
        "notes"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow([
            data.get("mW"), data.get("mZ"), data.get("mH"),
            data.get("sin2_mass"), data.get("sin2_geo"),
            data.get("delta_sin2"),
            data.get("rho"), data.get("rho_minus_1"),
            data.get("vW_band_rel"), data.get("vZ_band_rel"),
            data.get("notes", "")
        ])
    # JSON (for completeness)
    with open(os.path.join(outdir, "boson_unification_checks.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Boson-sector unification checks: Weinberg, custodial rho, band-vev consistency.")
    ap.add_argument("--locks", required=True, help="Path to all_particles_locked.csv")
    ap.add_argument("--seesaw", default="", help="Optional: electroweak seesaw sweep CSV (e.g., ew_seesaw_clean.csv)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    args = ap.parse_args()

    locks = load_locks(args.locks)
    if "W" not in locks or "Z" not in locks or "H" not in locks:
        raise ValueError("Locks file must include species rows named exactly 'W','Z','H'.")

    mW = safe_float(locks["W"], "m_GeV")
    mZ = safe_float(locks["Z"], "m_GeV")
    mH = safe_float(locks["H"], "m_GeV")
    if None in (mW, mZ, mH):
        raise ValueError("m_GeV column must be numeric for W,Z,H.")

    # 1) mass-based angle
    cos2_mass = (mW/mZ)**2
    sin2_mass = 1.0 - cos2_mass

    # 2) geometric angle from seesaw plateau
    sin2_geo = pick_plateau_from_seesaw(args.seesaw)
    # if missing, fall back to mass-based as a placeholder (we flag that in notes)
    notes = []
    if sin2_geo is None:
        sin2_geo = sin2_mass
        notes.append("sin2_geo missing -> using sin2_mass as placeholder")

    delta_sin2 = sin2_geo - sin2_mass

    # 3) Custodial symmetry with geometric angle
    cos2_geo = 1.0 - sin2_geo
    rho = (mW**2) / (mZ**2 * cos2_geo) if cos2_geo > 0 else float("nan")
    rho_minus_1 = rho - 1.0 if (rho == rho) else float("nan")

    # 4) Band-vev relative consistency (geometric, scheme-free ratios)
    # We construct dimensionless, relative estimators that should be comparable:
    #   vW_band_rel ~ mW / sqrt(S*(1-S)),  vZ_band_rel ~ mZ / sqrt(1-S)
    S = sin2_geo
    if 0.0 < S < 1.0:
        try:
            vW_band_rel = mW / math.sqrt(S*(1.0-S))
            vZ_band_rel = mZ / math.sqrt(1.0-S)
        except Exception:
            vW_band_rel = float("nan")
            vZ_band_rel = float("nan")
    else:
        vW_band_rel = float("nan")
        vZ_band_rel = float("nan")
        notes.append("S outside (0,1): cannot compute relative band-vev proxies")

    report = {
        "mW": mW, "mZ": mZ, "mH": mH,
        "sin2_mass": sin2_mass,
        "sin2_geo": sin2_geo,
        "delta_sin2": delta_sin2,
        "rho": rho,
        "rho_minus_1": rho_minus_1,
        "vW_band_rel": vW_band_rel,
        "vZ_band_rel": vZ_band_rel,
        "notes": "; ".join(notes)
    }
    make_report(args.outdir, report)

    # Pretty print
    print("\n=== Boson Unification Checks ===")
    print(f"mW = {mW:.6f} GeV,  mZ = {mZ:.6f} GeV,  mH = {mH:.6f} GeV")
    print(f"sin^2(theta_W) (mass)   = {sin2_mass:.6f}")
    print(f"sin^2(theta_W) (geo)    = {sin2_geo:.6f}   (delta = {delta_sin2:+.6e})")
    print(f"rho (custodial)         = {rho:.6f}   (rho-1 = {rho_minus_1:+.3e})")
    print(f"vW_band_rel (arb units) = {vW_band_rel:.6f}")
    print(f"vZ_band_rel (arb units) = {vZ_band_rel:.6f}")
    if notes:
        print("Notes:", "; ".join(notes))
    print(f"\nWrote report to: {os.path.join(args.outdir, 'boson_unification_checks.csv')}\n")

if __name__ == "__main__":
    main()