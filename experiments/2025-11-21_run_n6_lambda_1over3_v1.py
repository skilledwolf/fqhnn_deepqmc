from __future__ import annotations

import os
import pickle
from pathlib import Path
import argparse

import deepqmc
import jax
import jax.numpy as jnp
from deepqmc.types import PhysicalConfiguration

from fqhnn_deepqmc import (
    FQHDiskHamiltonian,
    FQHPsiformerAnsatz,
    disk_radius_for_filling,
    load_disk_potential,
)
from fqhnn_deepqmc.geometry import init_electron_configs
from fqhnn_deepqmc.vmc import train_vmc
from fqhnn_deepqmc.vmc_kfac import train_vmc_kfac


def parse_args():
    p = argparse.ArgumentParser(description="FQH Psiformer VMC (N=6, nu=1/3).")
    p.add_argument("--steps", type=int, default=800, help="Optimizer steps.")
    p.add_argument("--walkers", type=int, default=4096, help="Metropolis walkers.")
    p.add_argument("--mcmc-per-step", type=int, default=20, help="MCMC refreshes per opt step.")
    p.add_argument("--step-size", type=float, default=0.01, help="Metropolis proposal scale.")
    p.add_argument("--lr", type=float, default=2e-3, help="Optax Adam learning rate.")
    p.add_argument("--optimizer", choices=["adam", "kfac"], default="adam", help="Optimizer (adam|kfac).")
    p.add_argument("--kfac-damping", type=float, default=1e-3, help="KFAC damping (Tikhonov).")
    p.add_argument("--kfac-norm", type=float, default=1e-3, help="KFAC norm constraint.")
    p.add_argument("--kfac-momentum", type=float, default=0.9, help="KFAC momentum.")
    p.add_argument("--log-interval", type=int, default=20, help="Print every n steps.")
    p.add_argument("--logdir", type=str, default=None, help="TensorBoard log directory (optional).")
    p.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    return p.parse_args()


def main():
    args = parse_args()

    N = 6
    nu = 1.0 / 3.0
    lam = 1.0 / 3.0
    d = 0.1
    B = 1.0

    a = disk_radius_for_filling(N, filling=nu, B=B)
    print(f"N={N}, nu={nu}, lambda={lam}, a={a:.6f}, d={d}")

    table_path = Path("data") / f"disk_N{N}.npz"
    if not table_path.exists():
        raise FileNotFoundError(f"{table_path} missing. Run scripts/precompute_disk_potential.py first.")
    table = load_disk_potential(str(table_path), expected_nu=nu, expected_B=B)

    hamil = FQHDiskHamiltonian(n_electrons=N, lam=lam, B=B, a=a, table=table)
    ansatz = FQHPsiformerAnsatz(
        n_electrons=N,
        n_dets=16,
        n_layers=2,
        n_heads=4,
        head_dim=64,
        hidden_dim=256,
    )

    key = jax.random.PRNGKey(args.seed)
    key_init, key_train = jax.random.split(key)
    key_conf, key_params = jax.random.split(key_init)

    dummy_conf = hamil.init_sample(key_conf, hamil.background_R, 1)[0]
    params = ansatz.init(key_params, dummy_conf)
    print("Initialized ansatz parameters.")

    n_walkers = args.walkers
    if args.optimizer == "adam":
        params, sampler_state = train_vmc(
            key_train,
            params,
            ansatz.apply,
            hamil,
            n_steps=args.steps,
            n_walkers=n_walkers,
            mcmc_steps_per_iter=args.mcmc_per_step,
            step_size=args.step_size,
            opt_lr=args.lr,
            log_interval=args.log_interval,
            logdir=args.logdir,
        )
    else:
        params, sampler_state = train_vmc_kfac(
            key_train,
            params,
            ansatz.apply,
            hamil,
            n_steps=args.steps,
            n_walkers=n_walkers,
            mcmc_steps_per_iter=args.mcmc_per_step,
            step_size=args.step_size,
            opt_lr=args.lr,
            damping=args.kfac_damping,
            norm_constraint=args.kfac_norm,
            momentum=args.kfac_momentum,
            log_interval=args.log_interval,
            logdir=args.logdir,
        )

    os.makedirs("checkpoints", exist_ok=True)
    out_path = Path("checkpoints") / "n6_lambda1over3.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"params": params}, f)
    print(f"Saved params to {out_path}")


if __name__ == "__main__":
    main()
