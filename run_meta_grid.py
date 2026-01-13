#!/usr/bin/env python3
"""
Run a grid of meta-geometry probes and collate results.

Prereqs:
- meta_geometry_probe.py on PATH (or adjust PROBE)
- Your base H5 (torus slice) exists.

Usage (Windows cmd example shown below).

Outputs (per run, inside outdir/<run_name>/):
- meta_summary.json
- corr_vs_distance.csv
- meta_heat_trace.csv
Plus a master: outdir/meta_all_runs.csv
"""

import argparse, itertools, json, csv, subprocess, sys, os
from pathlib import Path

PROBE = r"C:\Users\anthoni.newland\OneDrive - ZINUS\Desktop\New Scripts\meta_geometry_probe.py"  # adjust if needed

def run_probe(h5_base, outdir, N, alpha, smooth_px, sigma_graph,
              operator="spectral", canvas_factor=3.0, rot_deg=0.0,
              windowsize=32, seed=0):
    run_name = f"N{N}_op{operator}_sp{int(round(smooth_px))}_sig{int(round(sigma_graph))}_rot{int(round(rot_deg))}_s{seed}"
    run_dir = Path(outdir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, PROBE,
        "--h5_base", str(h5_base),
        "--N", str(N),
        "--alpha", str(alpha),
        "--smooth_px", str(smooth_px),
        "--operator", operator,
        "--canvas_factor", str(canvas_factor),
        "--rot_deg", str(rot_deg),
        "--seed", str(seed),
        
        "--sigma_graph", str(sigma_graph),
        "--outdir", str(run_dir),
    ]
    print(">>", " ".join(cmd))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    return run_name, run_dir

def harvest_one(run_name, run_dir):
    meta = run_dir / "meta_summary.json"
    corr = run_dir / "corr_vs_distance.csv"
    heat = run_dir / "meta_heat_trace.csv"
    row = {
        "run": run_name,
        "path": str(run_dir).replace("\\", "/"),
        "ok_meta": meta.exists(),
        "ok_corr": corr.exists(),
        "ok_heat": heat.exists(),
    }
    if meta.exists():
        try:
            j = json.loads(meta.read_text())
            # accept common key variants
            row.update({
                "N": j.get("N", j.get("n")),
                "operator": j.get("operator", j.get("laplacian")),
                "sigma_graph": j.get("sigma_graph", j.get("sigma")),
                "smooth_px": j.get("smooth_px"),
                "rot_deg": j.get("rot_deg", j.get("rotation", 0.0)),
                "alpha": j.get("alpha"),
                "xi_pixels": j.get("xi_pixels", j.get("xi", None)),
                "xi_logfit_R2": j.get("xi_logfit_R2", j.get("xi_R2", None)),
                "ds_plateau_mean": j.get("ds_plateau_mean", None),
                "ds_plateau_spread": j.get("ds_plateau_spread", None),
                "ds_max": j.get("ds_max", None),
                "ds_median": j.get("ds_median", None),
            })
        except Exception as e:
            row["meta_read_error"] = str(e)
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_base", required=True,
        help="Path to torus_rich_t0_fields.h5 (or similar).")
    ap.add_argument("--outdir", required=True,
        help="Folder to receive per-run outputs + meta_all_runs.csv.")
    ap.add_argument("--alpha", type=float, default=4.9)
    ap.add_argument("--operator", default="spectral", choices=["spectral","nine"])
    ap.add_argument("--canvas_factor", type=float, default=3.0)
    ap.add_argument("--windowsize", type=int, default=32)
    ap.add_argument("--seeds", default="0,1,2", help="Comma list, e.g. 0,1")
    ap.add_argument("--N_list", default="256,300,400,512",
        help="Comma list of ensemble sizes.")
    ap.add_argument("--smooth_list", default="72,80,88",
        help="Comma list of smooth_px values.")
    ap.add_argument("--sigma_list", default="350,400,450",
        help="Comma list of sigma_graph values.")
    ap.add_argument("--rot_list", default="0,5",
        help="Comma list of small rotation degrees.")
    args = ap.parse_args()

    h5_base = Path(args.h5_base)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    N_list       = [int(x) for x in args.N_list.split(",") if x.strip()]
    smooth_list  = [float(x) for x in args.smooth_list.split(",") if x.strip()]
    sigma_list   = [float(x) for x in args.sigma_list.split(",") if x.strip()]
    rot_list     = [float(x) for x in args.rot_list.split(",") if x.strip()]
    seeds        = [int(x) for x in args.seeds.split(",") if x.strip()]

    print("=== meta-geometry grid ===")
    print("N:", N_list)
    print("smooth_px:", smooth_list)
    print("sigma_graph:", sigma_list)
    print("rot_deg:", rot_list)
    print("seeds:", seeds)
    print("operator:", args.operator, "alpha:", args.alpha)

    runs = []
    for (N, spx, sig, rot, seed) in itertools.product(N_list, smooth_list, sigma_list, rot_list, seeds):
        rn, rd = run_probe(
            h5_base=h5_base,
            outdir=outdir,
            N=N, alpha=args.alpha,
            smooth_px=spx, sigma_graph=sig,
            operator=args.operator,
            canvas_factor=args.canvas_factor,
            rot_deg=rot,
            windowsize=args.windowsize,
            seed=seed
        )
        runs.append((rn, rd))

    # Harvest
    rows = [harvest_one(rn, rd) for rn, rd in runs]
    csv_path = outdir / "meta_all_runs.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved:", csv_path)

if __name__ == "__main__":
    main()