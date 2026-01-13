#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
lock_fermion_axes.py
--------------------
Fit a quadratic z-surface over (ax, ay) from your latent table, then, for each
fermion mass, compute z_target from the isotonic table and find (ax, ay) that
best reproduces z_target inside the observed box.

Header handling is VERY tolerant:
- mass column may be named like: m_GeV, m GeV, Mass (GeV), mass_gev, mass, m
- logm column may be: logm, log_m, ln m, ln(m), lnm, logmass, lnmass, …
- z column may be: z, z_target, z iso, …
- q column may be: q, q_target, q iso, …

If logm is missing but mass is present, we compute logm = ln(mass).

Usage (example):
  python "...\lock_fermion_axes.py" ^
    --latent "...\latent_z_merged2.csv" ^
    --iso    "...\anchor_free_isotonic_masses.csv" ^
    --sin2 0.231 --alpha 0.0647 --beta 0.5529 ^
    --masses "u:0.0022,d:0.0047,s:0.095,e:0.000511,mu:0.10566,c:1.27,tau:1.77686,b:4.18,t:172.76" ^
    --out "...\fermions_locked.csv"
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- Helpers for robust headers ----------

def _norm(s: str) -> str:
    """Normalize a header: lower, strip, remove all non [a-z0-9_]."""
    s = s.lower().strip()
    out = []
    for ch in s:
        if ch.isalnum() or ch == "_":
            out.append(ch)
    return "".join(out)

def _best_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return original column name matching any normalized candidate."""
    norm_map = {_norm(c): c for c in df.columns}
    for k in candidates:
        if k in norm_map:
            return norm_map[k]
    return None


# ---------- ISO table handling ----------

def load_iso_table(path):
    """
    Load isotonic mass table.
    Returns dict with 'logm' and either 'z' or 'q' numpy arrays.
    """
    iso = pd.read_csv(path)

    # strip cell whitespace safely
    iso = iso.map(lambda x: x.strip() if isinstance(x, str) else x)

    # find logm or mass
    logm_col = _best_col(iso, (
        "logm", "log_m", "lnm", "ln_m", "lnmass", "logmass", "lnmg_ev", "logmg_ev", "lnmgev", "logmgev"
    ))
    mass_col = _best_col(iso, (
        "mgev", "massgev", "mass", "m", "mass_g_ev", "m_g_ev"
    ))

    if logm_col is None and mass_col is None:
        # try very generic: any column whose normalized name ends with 'gev'
        for c in iso.columns:
            if _norm(c).endswith("gev"):
                mass_col = c
                break

    if logm_col is None:
        if mass_col is None:
            raise ValueError("ISO table: couldn't find logm or mass (GeV) column after normalization.")
        iso["__logm__"] = pd.to_numeric(iso[mass_col], errors="coerce").map(
            lambda v: math.log(v) if pd.notna(v) else np.nan
        )
        logm_col = "__logm__"
    else:
        iso[logm_col] = pd.to_numeric(iso[logm_col], errors="coerce")

        # if a mass column also exists but logm has NaNs, fill them
        if mass_col is not None:
            mtmp = pd.to_numeric(iso[mass_col], errors="coerce")
            iso[logm_col] = iso[logm_col].fillna(mtmp.map(lambda v: math.log(v) if pd.notna(v) else np.nan))

    # z or q?
    z_col = _best_col(iso, ("z", "ztarget", "ziso"))
    q_col = _best_col(iso, ("q", "qtarget", "qiso"))

    if z_col is not None:
        iso[z_col] = pd.to_numeric(iso[z_col], errors="coerce")
        iso = iso.dropna(subset=[logm_col, z_col]).sort_values(logm_col)
        if iso.empty:
            raise ValueError("ISO table: 'z' column present but no valid numeric rows.")
        return {"logm": iso[logm_col].to_numpy(), "z": iso[z_col].to_numpy()}

    if q_col is not None:
        iso[q_col] = pd.to_numeric(iso[q_col], errors="coerce")
        iso = iso.dropna(subset=[logm_col, q_col]).sort_values(logm_col)
        if iso.empty:
            raise ValueError("ISO table: 'q' column present but no valid numeric rows.")
        return {"logm": iso[logm_col].to_numpy(), "q": iso[q_col].to_numpy()}

    # last resort: if a column literally named 'z_target' or 'q_target' existed before normalization
    for raw in iso.columns:
        r = raw.strip().lower()
        if r in ("z_target", "q_target"):
            iso[raw] = pd.to_numeric(iso[raw], errors="coerce")
            iso = iso.dropna(subset=[logm_col, raw]).sort_values(logm_col)
            key = "z" if r.startswith("z") else "q"
            return {"logm": iso[logm_col].to_numpy(), key: iso[raw].to_numpy()}

    raise ValueError("ISO table: no z or q column found (looked for z, z_target, z iso, q, q_target, q iso).")


def z_from_mass(m, iso_info):
    """
    Map mass m (GeV) to z_target using the ISO dataset by 1D interpolation over logm.
    If ISO stores q(logm), convert with the logit.
    """
    logm = math.log(m)
    if "z" in iso_info:
        z = np.interp(logm, iso_info["logm"], iso_info["z"])
        return logm, None, float(z)
    else:
        q = float(np.interp(logm, iso_info["logm"], iso_info["q"]))
        q = min(max(q, 1e-12), 1.0 - 1e-12)  # avoid 0/1
        z = math.log(q / (1.0 - q))
        return logm, q, z


# ---------- Latent surface fitting ----------

def _design(ax, ay):
    return np.stack([np.ones_like(ax), ax, ay, ax*ax, ax*ay, ay*ay], axis=1)

def fit_z_surface(df_latent):
    ax = pd.to_numeric(df_latent["ax"], errors="coerce").to_numpy()
    ay = pd.to_numeric(df_latent["ay"], errors="coerce").to_numpy()
    z  = pd.to_numeric(df_latent["z"],  errors="coerce").to_numpy()
    mask = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(z)
    ax, ay, z = ax[mask], ay[mask], z[mask]
    X = _design(ax, ay)
    coef, *_ = np.linalg.lstsq(X, z, rcond=None)
    bbox = {
        "ax_min": float(np.min(ax)), "ax_max": float(np.max(ax)),
        "ay_min": float(np.min(ay)), "ay_max": float(np.max(ay)),
    }
    return coef, bbox

def z_pred(ax, ay, coef):
    c0, c1, c2, c3, c4, c5 = coef
    return c0 + c1*ax + c2*ay + c3*ax*ax + c4*ax*ay + c5*ay*ay


# ---------- Search ----------

def _clamp(v, lo, hi): return max(lo, min(hi, v))

def refine_min(ax0, ay0, coef, z_target, bbox, steps=(0.02, 0.005, 0.001)):
    ax_min, ax_max = bbox["ax_min"], bbox["ax_max"]
    ay_min, ay_max = bbox["ay_min"], bbox["ay_max"]

    ax, ay = _clamp(ax0, ax_min, ax_max), _clamp(ay0, ay_min, ay_max)
    zh = z_pred(ax, ay, coef)
    best_err = abs(zh - z_target)
    best = (ax, ay, zh)

    for h in steps:
        improved = False
        for dax in (-h, -h/2, 0.0, h/2, h):
            for day in (-h, -h/2, 0.0, h/2, h):
                axt = _clamp(ax + dax, ax_min, ax_max)
                ayt = _clamp(ay + day, ay_min, ay_max)
                zt  = z_pred(axt, ayt, coef)
                err = abs(zt - z_target)
                if err + 1e-12 < best_err:
                    best_err = err
                    best = (axt, ayt, zt)
                    improved = True
        ax, ay = best[0], best[1]
        if not improved:
            ax = _clamp(ax + 0.25*h*(2*np.random.rand()-1), ax_min, ax_max)
            ay = _clamp(ay + 0.25*h*(2*np.random.rand()-1), ay_min, ay_max)

    note = "local_min_interior"
    eps = 1e-6
    if abs(ax - ax_min) < eps or abs(ax - ax_max) < eps or abs(ay - ay_min) < eps or abs(ay - ay_max) < eps:
        note = "on_box_edge"
    return best[0], best[1], best[2], float(best_err), note

def solve_for_mass(z_target, coef, bbox, seeds):
    best = None
    for ax0, ay0 in seeds:
        ax, ay, zh, err, note = refine_min(ax0, ay0, coef, z_target, bbox)
        cand = (err, ax, ay, zh, note)
        if best is None or cand < best:
            best = cand
    err, ax, ay, zh, note = best
    ok = (note == "local_min_interior")
    return ax, ay, zh, err, ok, note


# ---------- Utilities ----------

def parse_masses(s):
    out = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk: continue
        k, v = chunk.split(":", 1)
        out.append((k.strip(), float(v.strip())))
    return out


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", required=True)
    ap.add_argument("--iso",    required=True)
    ap.add_argument("--sin2",   type=float, default=0.231)
    ap.add_argument("--alpha",  type=float, default=0.0647)
    ap.add_argument("--beta",   type=float, default=0.5529)
    ap.add_argument("--masses", required=True)
    ap.add_argument("--out",    required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.latent)
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    for col in ("ax", "ay", "z"):
        if col not in df.columns:
            raise ValueError(f"Latent table missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    coef, bbox = fit_z_surface(df)
    print(f"[OK] fitted z-surface (deg=2) on {len(df.dropna(subset=['ax','ay','z']))} points.")
    print(f"     box: ax∈[{bbox['ax_min']:.4f},{bbox['ax_max']:.4f}], ay∈[{bbox['ay_min']:.4f},{bbox['ay_max']:.4f}]")
    print(f"     sin2={args.sin2} (α={args.alpha}, β={args.beta}) [reference only]")

    iso_info = load_iso_table(args.iso)

    latent_pts = df.dropna(subset=["ax","ay","z"])[["ax","ay","z"]].to_numpy()

    rows = []
    for species, m in parse_masses(args.masses):
        logm, q_t, z_t = z_from_mass(m, iso_info)

        dz = np.abs(latent_pts[:,2] - z_t)
        idx = np.argsort(dz)[:12]
        seeds = [(float(latent_pts[i,0]), float(latent_pts[i,1])) for i in idx]
        seeds += [
            ((bbox["ax_min"] + bbox["ax_max"])/2, (bbox["ay_min"] + bbox["ay_max"])/2),
            (bbox["ax_min"], (bbox["ay_min"] + bbox["ay_max"])/2),
            (bbox["ax_max"], (bbox["ay_min"] + bbox["ay_max"])/2),
            ((bbox["ax_min"] + bbox["ax_max"])/2, bbox["ay_min"]),
            ((bbox["ax_min"] + bbox["ax_max"])/2, bbox["ay_max"]),
        ]

        ax_b, ay_b, zhat, err, ok, note = solve_for_mass(z_t, coef, bbox, seeds)

        rows.append({
            "species": species,
            "m_GeV": m,
            "logm": logm,
            "q_target": q_t if q_t is not None else "",
            "z_target": z_t,
            "ax": ax_b,
            "ay": ay_b,
            "z_pred": zhat,
            "abs_err": err,
            "ok": bool(ok),
            "note": note,
        })

        print(f"\n[{species}] m={m} GeV -> z_target={z_t}")
        print(f"     best: ax={ax_b:.6f}  ay={ay_b:.6f}  ẑ={zhat:.9f}  |Δ|={err:.3e}  ok={ok}  ({note})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=[
        "species","m_GeV","logm","q_target","z_target","ax","ay","z_pred","abs_err","ok","note"
    ]).to_csv(out, index=False)
    print(f"\n[OK] wrote {out}")


if __name__ == "__main__":
    np.random.seed(0)
    main()