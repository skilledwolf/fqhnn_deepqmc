# scripts/precompute_disk_potential.py
"""
Precompute the confining disk potential Vc(r) using the standard
elliptic-integral representation:

  Vc(r) = -N/(π a^2) ∫_0^a dr' r' [ 4 / √(d² + (r+r')²) ] K(m),

with m = 4 r r' / (d² + (r + r')²).

The resulting table Vc(r) has no λ factor; λ is applied later to both
electron-electron and electron-background terms in the Hamiltonian.

Usage:
  python scripts/precompute_disk_potential.py --N 6 --nu 0.333333 --B 1.0 --d 0.1 \
      --n_points 10000 --out data/disk_N6.npz
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy import integrate, special


def Vc_r(r: float, N: int, a: float, d: float) -> float:
    """Evaluate V_c(r) via a robust 1D integral with elliptic K."""

    def integrand(rp):
        rp = float(rp)
        d2 = d * d
        r2 = r * r
        rp2 = rp * rp

        C = d2 + r2 + rp2
        D = 2.0 * r * rp
        # parameter for K: 0 <= m < 1
        m = 2.0 * D / (C + D)  # = 4 r rp / (d² + (r+rp)²)
        # Guard against tiny overshoots due to rounding
        m = np.clip(m, 0.0, 1.0 - 1e-12)

        prefactor = 4.0 / np.sqrt(C + D)
        return rp * prefactor * special.ellipk(m)

    val, _ = integrate.quad(
        integrand,
        0.0,
        a,
        epsabs=1e-9,
        epsrel=1e-7,
        limit=200,
    )
    return -N / (np.pi * a * a) * val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--nu", type=float, default=1.0 / 3.0)
    parser.add_argument("--B", type=float, default=1.0)
    parser.add_argument("--d", type=float, default=0.1)
    parser.add_argument("--n_points", type=int, default=10_000)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    N = args.N
    nu = args.nu
    B = args.B
    d = args.d
    n_points = args.n_points

    # Disk radius from flux condition (in l_B=1 units):
    #    B a^2 / 2 = N/ν  ⇒  a^2 = 2N/(ν B).
    a = float(np.sqrt(2.0 * N / (nu * B)))

    r_max = 15.0 * a
    r_grid = np.linspace(0.0, r_max, n_points)
    V_vals = np.empty_like(r_grid)

    print(f"Precomputing Vc(r) for N={N}, nu={nu}, d={d}, a={a:.6f}")
    for i, r in enumerate(r_grid):
        V_vals[i] = Vc_r(r, N=N, a=a, d=d)
        if (i + 1) % max(1, n_points // 20) == 0:
            print(f"{i+1}/{n_points} points")

    # Sanity: for a correct integrand, V_vals should be finite everywhere.
    nonfinite = ~np.isfinite(V_vals)
    if np.any(nonfinite):
        n_bad = int(np.sum(nonfinite))
        raise RuntimeError(
            f"Found {n_bad} non-finite Vc values; investigate instead of interpolating."
        )

    # Quick analytic sanity checks.
    V0_expected = -(2.0 * N / (a * a)) * (np.sqrt(d * d + a * a) - d)
    V0_err = abs(V_vals[0] - V0_expected)
    if V0_err > 1e-5:
        raise RuntimeError(
            f"V_c(0) mismatch: computed {V_vals[0]:.8f}, expected {V0_expected:.8f} "
            f"(|Δ|={V0_err:.2e})."
        )

    tail_target = -N / np.sqrt(d * d + r_max * r_max)
    tail_rel_err = abs(V_vals[-1] - tail_target) / max(abs(tail_target), 1e-12)
    if tail_rel_err > 1e-3:
        raise RuntimeError(
            f"Tail mismatch at r={r_max:.3f}: computed {V_vals[-1]:.8f}, "
            f"target {tail_target:.8f} (rel err {tail_rel_err:.2e})."
        )

    np.savez(
        args.out,
        r_grid=r_grid,
        V_grid=V_vals,
        a=a,
        d=d,
        N=N,
        nu=nu,
        B=B,
    )
    print(f"Saved table to {args.out}")


if __name__ == "__main__":
    main()
