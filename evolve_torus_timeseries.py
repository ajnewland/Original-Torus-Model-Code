
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evolve a real torus time-series from an existing t0 slice (stable scaling).
Key fixes:
  • Spectral Laplacian uses unit-box frequencies (d=1.0), independent of x,y span.
  • lnΩ tightly clipped; Ω floored and sanitized before divisions.
"""

import os, sys, argparse, numpy as np, h5py, matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------- numerics ----------
def laplacian_spectral_unit(F):
    """Periodic spectral Laplacian on a unit box (grid step = 1.0)."""
    Ny, Nx = F.shape
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=1.0)  # unit spacing
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=1.0)
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX*KX + KY*KY
    Fhat = np.fft.fft2(F)
    lap_hat = -k2 * Fhat
    lap_hat[0,0] = 0.0
    return np.fft.ifft2(lap_hat).real

def central_grad_unit(F):
    """Central differences with periodic wrap; unit spacing."""
    dFx = (np.roll(F, -1, 1) - np.roll(F, 1, 1)) * 0.5
    dFy = (np.roll(F, -1, 0) - np.roll(F, 1, 0)) * 0.5
    return dFx, dFy

def gaussian1d_kernel(sigma_px, radius=None):
    if sigma_px <= 0: return None
    if radius is None: radius = int(max(3, round(3*sigma_px)))
    v = np.arange(-radius, radius+1, dtype=float)
    k = np.exp(-0.5*(v/sigma_px)**2); k /= k.sum()
    return k

def gaussian_smooth2d(F, sigma_px):
    k = gaussian1d_kernel(sigma_px)
    if k is None: return F
    pad = len(k)//2
    G = F.copy()
    # rows
    for j in range(F.shape[0]):
        row = np.r_[F[j, -pad:], F[j], F[j, :pad]]
        conv = np.convolve(row, k, mode="same")
        G[j, :] = conv[pad:pad+F.shape[1]]
    # cols
    H = G.copy()
    for i in range(F.shape[1]):
        col = np.r_[G[-pad:, i], G[:, i], G[:pad, i]]
        conv = np.convolve(col, k, mode="same")
        H[:, i] = conv[pad:pad+F.shape[0]]
    return H

def sanitize(A, fill=0.0):
    B = A.copy()
    bad = ~np.isfinite(B)
    if bad.any(): B[bad] = fill
    return B

def safe_lnOmega(T, alpha, smooth_px, lnom_clip):
    lnOm = alpha*T
    if smooth_px > 0: lnOm = gaussian_smooth2d(lnOm, smooth_px)
    lnOm = lnOm - float(np.mean(lnOm))
    if lnom_clip and lnom_clip > 0:
        lnOm = np.clip(lnOm, -lnom_clip, lnom_clip)
    return lnOm

def lnOmega_to_Omega(lnOm, omega_floor):
    Om = np.exp(lnOm)
    m = Om.mean()
    if not np.isfinite(m) or m <= 0:
        lnOm = np.clip(lnOm - float(np.mean(lnOm)), -8.0, 8.0)
        Om = np.exp(lnOm); m = Om.mean() if np.isfinite(Om.mean()) and Om.mean() > 0 else 1.0
    Om = Om / m
    Om = np.maximum(Om, omega_floor)
    return sanitize(Om, fill=1.0)

def recompute_fields(T, rho_m, alpha, smooth_px, lnom_clip, omega_floor, report=False, tag=""):
    lnOm = safe_lnOmega(T, alpha, smooth_px, lnom_clip)
    Om   = lnOmega_to_Omega(lnOm, omega_floor)
    lap_ln = laplacian_spectral_unit(lnOm)
    Om2 = Om*Om
    R2 = -2.0 * lap_ln / Om2
    dTdx, dTdy = central_grad_unit(T)
    rho_phi = 0.5 * (dTdx*dTdx + dTdy*dTdy) / Om2
    R2 = sanitize(R2); rho_phi = sanitize(rho_phi)
    if report:
        print(f"[{tag}] lnΩ min/max={lnOm.min():.3f}/{lnOm.max():.3f} | Ω min/max={Om.min():.3e}/{Om.max():.3e}")
        print(f"[{tag}] R2 rms={np.sqrt(np.mean(R2**2)):.3g} | rho_phi rms={np.sqrt(np.mean(rho_phi**2)):.3g}")
    return dict(lnOm=lnOm, Om=Om, R2=R2, rho_phi=rho_phi)

def evolve_step(T, mode, dt, mu, advec_angle_deg, advec_speed):
    if mode == "gradflow":
        lapT = laplacian_spectral_unit(T)
        return T + dt*mu*lapT
    elif mode == "advection":
        theta = np.deg2rad(advec_angle_deg)
        vx, vy = advec_speed*np.cos(theta), advec_speed*np.sin(theta)
        sx = int(np.round(vx*dt))  # unit spacing
        sy = int(np.round(vy*dt))
        return np.roll(np.roll(T, sy, axis=0), sx, axis=1)
    else:
        raise ValueError("mode must be 'gradflow' or 'advection'")

def maybe_rescale_T(T, cap_std):
    if cap_std is None or cap_std <= 0: return T
    m = float(np.mean(T)); s = float(np.std(T))
    if s > cap_std:
        T = (T - m) * (cap_std / s) + m
    return T

def save_h5(path, T, rho_m, fields, meta):
    with h5py.File(path, "w") as f:
        f.create_dataset("Teff", data=T)
        f.create_dataset("rho_m", data=rho_m)
        f.create_dataset("lnOmega", data=fields["lnOm"])
        f.create_dataset("Omega", data=fields["Om"])
        f.create_dataset("R2", data=fields["R2"])
        f.create_dataset("rho_phi", data=fields["rho_phi"])
        for k, v in meta.items():
            f.attrs[k] = str(v)

def quick_png(path, arr, title):
    plt.figure(figsize=(6.0,4.8))
    plt.imshow(arr, origin="lower")
    plt.colorbar()
    plt.title(title); plt.xlabel("a_x (pix)"); plt.ylabel("a_y (pix)")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# ---------- main ----------
def main():
    import argparse, h5py
    ap = argparse.ArgumentParser("Evolve real torus time-series (stable scaling)")
    ap.add_argument("--h5_t0", required=True)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--mode", choices=["gradflow","advection"], default="gradflow")
    ap.add_argument("--mu", type=float, default=0.003)
    ap.add_argument("--advec_angle", type=float, default=0.0)
    ap.add_argument("--advec_speed", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=4.9)
    ap.add_argument("--smooth_px", type=float, default=1.6)
    ap.add_argument("--lnom_clip", type=float, default=3.0, help="clip |lnΩ-mean| (tighter is safer)")
    ap.add_argument("--omega_floor", type=float, default=1e-3, help="minimum Ω to avoid blow-ups")
    ap.add_argument("--rescale_T", type=float, default=0.5, help="cap std(Teff) per step; 0=off")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--outprefix", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.outprefix), exist_ok=True)

    # Load minimal fields (only those we need)
    with h5py.File(args.h5_t0, "r") as f0:
        T  = f0["Teff"][:]
        rho_m = f0["rho_m"][:]

    # Optional initial rescale
    T = maybe_rescale_T(T, args.rescale_T)

    # Save t0 via this pipeline
    f0_fields = recompute_fields(T, rho_m, args.alpha, args.smooth_px,
                                 args.lnom_clip, args.omega_floor, report=args.report, tag="t0")
    save_h5(f"{args.outprefix}_t0_fields.h5", T, rho_m, f0_fields,
            meta=dict(step=0, dt=args.dt, mode=args.mode, mu=args.mu,
                      advec_angle=args.advec_angle, advec_speed=args.advec_speed,
                      alpha=args.alpha, smooth_px=args.smooth_px,
                      lnom_clip=args.lnom_clip, omega_floor=args.omega_floor,
                      rescale_T=args.rescale_T))
    quick_png(f"{args.outprefix}_t0_R2.png", f0_fields["R2"], "R2 (t0)")
    quick_png(f"{args.outprefix}_t0_Teff.png", T, "Teff (t0)")

    # Evolve
    Tn = T.copy()
    for k in range(1, args.steps+1):
        Tn = evolve_step(Tn, args.mode, args.dt, args.mu, args.advec_angle, args.advec_speed)
        Tn = maybe_rescale_T(Tn, args.rescale_T)
        fields = recompute_fields(Tn, rho_m, args.alpha, args.smooth_px,
                                  args.lnom_clip, args.omega_floor, report=args.report, tag=f"t{k}")
        save_h5(f"{args.outprefix}_t{k}_fields.h5", Tn, rho_m, fields,
                meta=dict(step=k, dt=args.dt, mode=args.mode, mu=args.mu,
                          advec_angle=args.advec_angle, advec_speed=args.advec_speed,
                          alpha=args.alpha, smooth_px=args.smooth_px,
                          lnom_clip=args.lnom_clip, omega_floor=args.omega_floor,
                          rescale_T=args.rescale_T))
        quick_png(f"{args.outprefix}_t{k}_R2.png", fields["R2"], f"R2 (t{k})")
        quick_png(f"{args.outprefix}_t{k}_Teff.png", Tn, f"Teff (t{k})")

    print(f"Done. Wrote t0..t{args.steps} to prefix: {args.outprefix}")

if __name__ == "__main__":
    main()