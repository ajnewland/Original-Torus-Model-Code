#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_unified_torus.py
---------------------
Base visualization for the unified torus:
- Torus parameterized by (a_x, a_y) ≡ (u, v) in [0, 2π)^2
- Latent surface z(u,v) (synthetic but shaped for concave/convex regions)
- Color map = z(u,v)
- Equilibrium ridge (A=B) drawn as a guide curve
- Placeholders for particle groups and gauge rings
- Inset spectral-dimension flow panel (synthetic tri-plateau curve)
Outputs: PNG + SVG in ./fig_unified_torus_{png,svg}

Usage:
  python make_unified_torus.py \
      --R 2.0 --r 0.85 --nu 280 --nv 360 \
      --out-prefix fig_unified_torus \
      --show
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def latent_z(u, v, *,
             phi=0.7, amp1=1.0, amp2=0.55, amp3=0.25,
             skew=0.22, bias=0.0):
    """
    Synthetic latent surface z(u,v).
    Tuned to give:
      - alternating concave/convex regions,
      - a soft ridge where A≈B (u≈v),
      - smooth plateaus for visual clarity.

    u, v: radians on [0, 2π)
    returns z of shape u.shape
    """
    # Base torus “texture”
    base = amp1*np.cos(u) + amp2*np.cos(2*v + phi) + amp3*np.cos(u - v)
    # Gentle cross-term to form a diagonal equilibrium valley/ridge
    cross = 0.35*np.cos((u - v))
    # Slight skew to break perfect symmetry (more realistic)
    skew_term = skew*(np.sin(2*u)*np.sin(v))
    return base + cross + skew_term + bias

def torus_xyz(u, v, R=2.0, r=0.85):
    """
    Standard torus embedding in R^3.
    u,v in radians on [0, 2π)
    R = major radius, r = minor radius
    """
    x = (R + r*np.cos(v)) * np.cos(u)
    y = (R + r*np.cos(v)) * np.sin(u)
    z =  r*np.sin(v)
    return x, y, z

def normalize01(a):
    amin, amax = np.min(a), np.max(a)
    if amax == amin:
        return np.zeros_like(a)
    return (a - amin) / (amax - amin)

def synthetic_ds_curve(n=300, s1=1e-3, s2=1e-1, s3=10.0,
                       d_small=1.26, d_mid=2.60, d_large=4.00):
    """
    Smooth tri-plateau spectral-dimension curve vs 'sigma' (log-spaced).
    """
    sigma = np.logspace(-4, 4, n)
    # Two logistic ramps
    w1 = 1.0 / (1.0 + np.exp(- (np.log10(sigma) - np.log10(s1)) / 0.35))
    w2 = 1.0 / (1.0 + np.exp(- (np.log10(sigma) - np.log10(s3)) / 0.45))
    ds = d_small*(1-w1) + d_mid*w1*(1-w2) + d_large*w2
    return sigma, ds

def add_equilibrium_ridge(ax3d, R, r, n=500):
    """
    Draw a guide curve for A≈B. Here we use u≈v (diagonal) as a reasonable proxy.
    """
    t = np.linspace(0, 2*np.pi, n)
    u = t
    v = t
    X, Y, Z = torus_xyz(u, v, R=R, r=r)
    ax3d.plot(X, Y, Z, lw=2.0, alpha=0.9)

def place_text_on_surface(ax3d, label, u, v, R=2.0, r=0.85, zoff=0.0,
                          fontsize=9, ha='center', va='center'):
    """
    Project a label on/near the surface at (u,v).
    """
    x, y, z = torus_xyz(np.array([u]), np.array([v]), R=R, r=r)
    ax3d.text(float(x), float(y), float(z + zoff), label,
              fontsize=fontsize, ha=ha, va=va)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--R', type=float, default=2.0, help='Major radius')
    p.add_argument('--r', type=float, default=0.85, help='Minor radius')
    p.add_argument('--nu', type=int, default=280, help='Grid along u (a_x)')
    p.add_argument('--nv', type=int, default=360, help='Grid along v (a_y)')
    p.add_argument('--out-prefix', type=str, default='fig_unified_torus',
                   help='Output file prefix (no extension)')
    p.add_argument('--show', action='store_true', help='Show window')
    args = p.parse_args()

    # Parameter grids (a_x, a_y) ≡ (u, v)
    u = np.linspace(0, 2*np.pi, args.nu, endpoint=True)
    v = np.linspace(0, 2*np.pi, args.nv, endpoint=True)
    U, V = np.meshgrid(u, v, indexing='ij')

    # Latent surface & torus embedding
    Zlatent = latent_z(U, V)
    X, Y, Z = torus_xyz(U, V, R=args.R, r=args.r)
    C = normalize01(Zlatent)  # color field

    # Figure
    fig = plt.figure(figsize=(11.5, 8.5), dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Plot surface
    surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2,
                           facecolors=cm.viridis(C),
                           linewidth=0, antialiased=True, shade=False)

    # Lighting hint (optional): turn off default shading to keep color = data
    ax.set_box_aspect((1, 1, 0.6))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$'); ax.set_zlabel(r'$z$')
    ax.view_init(elev=25, azim=35)

    # Colorbar for z(a_x,a_y)
    mappable = cm.ScalarMappable(cmap=cm.viridis)
    mappable.set_array(C)
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
    cbar.set_label(r'latent surface $z(a_x,a_y)$', rotation=90)

    # Draw equilibrium ridge (A≈B)
    add_equilibrium_ridge(ax, R=args.R, r=args.r, n=600)

    # Gauge “rings” (labels near the principal cycles)
    place_text_on_surface(ax, r'$\mathrm{U}(1)$', u=0.06*np.pi, v=np.pi,
                          R=args.R, r=args.r, zoff=0.08, fontsize=12)
    place_text_on_surface(ax, r'$\mathrm{SU}(2)$', u=np.pi, v=0.02*np.pi,
                          R=args.R, r=args.r, zoff=0.10, fontsize=12)
    place_text_on_surface(ax, r'$\mathrm{SU}(3)$', u=1.62*np.pi, v=1.12*np.pi,
                          R=args.R, r=args.r, zoff=0.10, fontsize=12)

    # Sector placeholders (you can nudge u,v later for aesthetics)
    # Charged leptons (concave wells)
    place_text_on_surface(ax, 'e',  u=0.35*np.pi, v=1.65*np.pi, R=args.R, r=args.r, zoff=0.02)
    place_text_on_surface(ax, r'$\mu$', u=0.52*np.pi, v=1.40*np.pi, R=args.R, r=args.r, zoff=0.02)
    place_text_on_surface(ax, r'$\tau$', u=0.70*np.pi, v=1.18*np.pi, R=args.R, r=args.r, zoff=0.02)

    # Up-type quarks (convex ridge, + charge)
    place_text_on_surface(ax, 'u',  u=1.05*np.pi, v=0.30*np.pi, R=args.R, r=args.r, zoff=0.02)
    place_text_on_surface(ax, 'c',  u=1.20*np.pi, v=0.52*np.pi, R=args.R, r=args.r, zoff=0.02)
    place_text_on_surface(ax, 't',  u=1.38*np.pi, v=0.76*np.pi, R=args.R, r=args.r, zoff=0.02)

    # Down-type quarks (negative curvature side)
    place_text_on_surface(ax, 'd',  u=1.05*np.pi, v=1.70*np.pi, R=args.R, r=args.r, zoff=0.02)
    place_text_on_surface(ax, 's',  u=1.20*np.pi, v=1.45*np.pi, R=args.R, r=args.r, zoff=0.02)
    place_text_on_surface(ax, 'b',  u=1.38*np.pi, v=1.20*np.pi, R=args.R, r=args.r, zoff=0.02)

    # Neutrinos (near-flat basins)
    place_text_on_surface(ax, r'$\nu_e$',  u=0.12*np.pi, v=0.18*np.pi, R=args.R, r=args.r, zoff=0.02)
    place_text_on_surface(ax, r'$\nu_\mu$', u=0.25*np.pi, v=0.30*np.pi, R=args.R, r=args.r, zoff=0.02)
    place_text_on_surface(ax, r'$\nu_\tau$',u=0.40*np.pi, v=0.42*np.pi, R=args.R, r=args.r, zoff=0.02)

    # Bosons (convex ridges)
    place_text_on_surface(ax, r'$\gamma$',  u=1.70*np.pi, v=0.10*np.pi, R=args.R, r=args.r, zoff=0.02)
    place_text_on_surface(ax, r'$W^\pm$',   u=1.85*np.pi, v=0.35*np.pi, R=args.R, r=args.r, zoff=0.02)
    place_text_on_surface(ax, r'$Z^0$',     u=1.95*np.pi, v=0.55*np.pi, R=args.R, r=args.r, zoff=0.02)
    place_text_on_surface(ax, r'$g$',       u=1.60*np.pi, v=0.80*np.pi, R=args.R, r=args.r, zoff=0.02)
    place_text_on_surface(ax, r'$H$',       u=0.95*np.pi, v=0.98*np.pi, R=args.R, r=args.r, zoff=0.02)

    # Key equations as floating annotations (you can reposition later)
    ax.text2D(0.02, 0.96, r'$\log m=\alpha z+\beta$', transform=ax.transAxes, fontsize=10)
    ax.text2D(0.02, 0.91, r'$\sin^2\theta_W\simeq 0.231$', transform=ax.transAxes, fontsize=10)
    ax.text2D(0.02, 0.86, r'$Q\sim \mathrm{sgn}[\det H(z)]$', transform=ax.transAxes, fontsize=10)

    # Inset: spectral dimension flow
    axins = inset_axes(ax, width="28%", height="28%", loc="upper right",
                       borderpad=0.8)
    sigma, ds = synthetic_ds_curve()
    axins.plot(sigma, ds, lw=1.8)
    axins.set_xscale('log')
    axins.set_xlabel(r'$\sigma$', fontsize=8)
    axins.set_ylabel(r'$d_s(\sigma)$', fontsize=8)
    axins.set_title('Spectral flow', fontsize=9)
    axins.grid(True, alpha=0.3)

    fig.suptitle("Unified Torus: geometry, fields, and relations (base render)", fontsize=14, y=0.98)

    # Save
    png_path = f"{args.out_prefix}.png"
    svg_path = f"{args.out_prefix}.svg"
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(svg_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {png_path}")
    print(f"[OK] Saved: {svg_path}")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()