import argparse, subprocess, sys, math, os

EV_TO_GEV = 1e-9

def nh_masses(m_light, dm21, dm31):
    # Normal hierarchy: m1 = m_light
    m1 = m_light
    m2 = math.sqrt(m1*m1 + dm21)
    m3 = math.sqrt(m1*m1 + dm31)
    return m1, m2, m3

def ih_masses(m_light, dm21, dm31_abs):
    # Inverted hierarchy: m3 = m_light
    m3 = m_light
    m1 = math.sqrt(m3*m3 + dm31_abs)       # since Δm^2_31 ≈ m3^2 - m1^2 ≈ -|dm31|
    m2 = math.sqrt(m1*m1 + dm21)
    return m1, m2, m3

def fmt(val):
    # keep enough precision for tiny neutrino masses
    return f"{val:.12g}"

def main():
    ap = argparse.ArgumentParser(description="Compute neutrino masses from Δm^2 and call predict_ax_ay_for_mass.py")
    ap.add_argument("--hierarchy", choices=["NH","IH"], default="NH")
    ap.add_argument("--m_light_eV", type=float, default=0.001, help="Lightest mass (eV). NH: m1, IH: m3")
    ap.add_argument("--dm21_eV2", type=float, default=7.42e-5, help="Solar Δm^2_21 (eV^2)")
    ap.add_argument("--dm31_eV2", type=float, default=2.517e-3, help="Atmospheric |Δm^2_31| (eV^2); sign from hierarchy")
    ap.add_argument("--latent", required=True)
    ap.add_argument("--iso", required=True)
    ap.add_argument("--sin2", type=float, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--predict_script", required=True,
                    help="Path to your predict_ax_ay_for_mass.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.hierarchy == "NH":
        m1_eV, m2_eV, m3_eV = nh_masses(args.m_light_eV, args.dm21_eV2, args.dm31_eV2)
        labels = ["nu1","nu2","nu3"]
        masses_GeV = [m1_eV*EV_TO_GEV, m2_eV*EV_TO_GEV, m3_eV*EV_TO_GEV]
    else:
        m1_eV, m2_eV, m3_eV = ih_masses(args.m_light_eV, args.dm21_eV2, args.dm31_eV2)
        labels = ["nu1","nu2","nu3"]  # naming stays ν1<ν2 by mass; ν3 is lightest in IH here
        masses_GeV = [m1_eV*EV_TO_GEV, m2_eV*EV_TO_GEV, m3_eV*EV_TO_GEV]

    masses_arg = ",".join(f"{lab}:{fmt(m)}" for lab,m in zip(labels, masses_GeV))
    print("[neutrino masses]")
    print(f" hierarchy={args.hierarchy}")
    print(f" m_light = {args.m_light_eV} eV")
    print(f" m({labels[0]}) = {masses_GeV[0]/EV_TO_GEV:.6g} eV")
    print(f" m({labels[1]}) = {masses_GeV[1]/EV_TO_GEV:.6g} eV")
    print(f" m({labels[2]}) = {masses_GeV[2]/EV_TO_GEV:.6g} eV")
    print(f"--masses \"{masses_arg}\"")

    # Call your existing predictor
    cmd = [
        sys.executable, args.predict_script,
        "--latent", args.latent,
        "--iso", args.iso,
        "--sin2", str(args.sin2),
        "--alpha", str(args.alpha),
        "--beta", str(args.beta),
        "--masses", masses_arg,
        "--out", args.out
    ]
    print("\n[calling]", " ".join(cmd))
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()