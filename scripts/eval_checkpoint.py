from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from fqhnn_deepqmc import (
    FQHDiskHamiltonian,
    FQHPsiformerAnsatz,
    disk_radius_for_filling,
    load_disk_potential,
)
from fqhnn_deepqmc.vmc import make_sampler


def radial_density(samples: jnp.ndarray, r_max: float, n_bins: int = 50):
    r = jnp.sqrt(jnp.sum(samples**2, axis=-1)).reshape(-1)
    bins = jnp.linspace(0.0, r_max, n_bins + 1)
    hist, _ = jnp.histogram(r, bins=bins)
    dr = r_max / n_bins
    r_centers = (bins[:-1] + bins[1:]) / 2.0
    area_shell = 2.0 * jnp.pi * r_centers * dr
    rho = hist / (area_shell * (samples.shape[0]))
    # normalize so 2π ∫ ρ(r) r dr = N
    norm = jnp.sum(rho * area_shell)
    rho = rho / (norm / samples.shape[1])
    return r_centers, rho


def structure_factor(samples: jnp.ndarray, k_vals):
    """
    S(k) = 1/N < | Σ_j exp(i k·r_j) |^2 >, averaged over angles of k.
    samples: [S, N, 2]
    k_vals: 1D array of |k|
    """
    S, N, _ = samples.shape
    thetas = jnp.linspace(0, 2 * jnp.pi, 32, endpoint=False)
    k_dirs = jnp.stack([jnp.cos(thetas), jnp.sin(thetas)], axis=-1)  # [A, 2]

    def s_of_k(k_mag):
        k_vecs = k_mag * k_dirs  # [A, 2]
        phase = jnp.einsum("a2,sn2->san", k_vecs, samples)  # [S, A, N]
        exp_phase = jnp.exp(1j * phase)
        rho_k = jnp.sum(exp_phase, axis=-1)  # [S, A]
        s_val = jnp.mean(jnp.abs(rho_k) ** 2 / N, axis=(0, 1))
        return s_val.real

    return jnp.stack([s_of_k(k) for k in k_vals])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/n6_lambda1over3.pkl")
    p.add_argument("--steps", type=int, default=200, help="MCMC samples to collect")
    p.add_argument("--burn", type=int, default=50, help="Burn-in steps")
    p.add_argument("--walkers", type=int, default=256)
    p.add_argument("--outdir", type=str, default="plots")
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    with open(ckpt, "rb") as f:
        params = pickle.load(f)["params"]

    # Hard-coded system (matches checkpoint)
    N = 6
    nu = 1.0 / 3.0
    lam = 1.0 / 3.0
    B = 1.0
    a = disk_radius_for_filling(N, filling=nu, B=B)
    table = load_disk_potential("data/disk_N6.npz", expected_nu=nu, expected_B=B)

    hamil = FQHDiskHamiltonian(N, lam, B, a, table)
    ansatz = FQHPsiformerAnsatz(
        n_electrons=N,
        n_dets=16,
        n_layers=2,
        n_heads=4,
        head_dim=64,
        hidden_dim=256,
    )

    rng = jax.random.PRNGKey(0)
    sampler = make_sampler(hamil, ansatz.apply, step_size=0.01)
    rng, init_rng = jax.random.split(rng)
    state = sampler.init(init_rng, params, args.walkers, hamil.background_R)
    le_fn = hamil.local_energy(ansatz.apply)

    samples = []
    energies = []

    for i in range(args.burn + args.steps):
        rng, sk = jax.random.split(rng)
        state, phys_conf, _ = sampler.sample(sk, state, params, hamil.background_R)
        if i >= args.burn:
            samples.append(phys_conf.r)
            E, _ = le_fn(None, params, phys_conf)
            energies.append(E)

    samples = jnp.concatenate(samples, axis=0)  # [S, N, 2]
    energies = jnp.concatenate([e.reshape(-1) for e in energies])

    print(f"Collected {samples.shape[0]} configs; energy mean={float(energies.mean()):.3f} ± {float(energies.std()):.3f}")

    r_centers, rho = radial_density(samples, r_max=10.0, n_bins=80)
    k_vals = jnp.linspace(0.1, 3.0, 60)
    S_k = structure_factor(samples, k_vals)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(4, 3))
    plt.plot(r_centers, rho)
    plt.xlabel("r")
    plt.ylabel("rho(r)")
    plt.tight_layout()
    plt.savefig(outdir / "radial_density.png", dpi=200)

    plt.figure(figsize=(4, 3))
    plt.plot(k_vals, S_k)
    plt.xlabel("|k|")
    plt.ylabel("S(k)")
    plt.tight_layout()
    plt.savefig(outdir / "structure_factor.png", dpi=200)

    print(f"Saved plots to {outdir}")


if __name__ == "__main__":
    main()
