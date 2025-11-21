#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fqhnn_deepqmc import (
    FQHDiskHamiltonian,
    FQHPsiformerAnsatz,
    disk_radius_for_filling,
    load_disk_potential,
    train_vmc,
    train_vmc_kfac,
)


def main():
    p = argparse.ArgumentParser(
        description="Train Psiformer+Jastrow FQH ansatz for N=6, nu=1/3 on a disk."
    )

    # System / physics
    p.add_argument("--N", type=int, default=6, help="Number of electrons")
    p.add_argument("--nu", type=float, default=1.0 / 3.0, help="Filling fraction")
    p.add_argument("--B", type=float, default=1.0, help="Magnetic field (in l_B units)")
    p.add_argument("--lam", type=float, default=1.0 / 3.0, help="Coulomb coupling λ")
    p.add_argument(
        "--potential-table",
        type=str,
        default="data/disk_N6.npz",
        help="Path to precomputed confining potential table (.npz).",
    )

    # Ansatz hyperparameters
    p.add_argument("--n-dets", type=int, default=16)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=256)

    # VMC / optimizer
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n-steps", type=int, default=20000, help="Optimizer steps")
    p.add_argument("--n-walkers", type=int, default=512)
    p.add_argument("--mcmc-steps-per-iter", type=int, default=10)
    p.add_argument("--step-size", type=float, default=0.02, help="Metropolis tau")

    p.add_argument(
        "--optimizer",
        choices=["adam", "kfac"],
        default="kfac",
        help="Which optimizer to use.",
    )
    p.add_argument("--opt-lr", type=float, default=1e-3, help="Learning rate")

    # KFAC-specific
    p.add_argument("--kfac-damping", type=float, default=1e-3)
    p.add_argument("--kfac-norm-constraint", type=float, default=1e-3)
    p.add_argument("--kfac-momentum", type=float, default=0.9)
    p.add_argument("--kfac-curvature-ema", type=float, default=0.95)
    p.add_argument("--kfac-inverse-update-period", type=int, default=10)

    # Logging / checkpoints
    p.add_argument(
        "--log-interval", type=int, default=50, help="Print/log every this many steps."
    )
    p.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="TensorBoard logdir (requires tensorboardX) or omit to disable.",
    )
    p.add_argument(
        "--ckpt-out",
        type=str,
        default="checkpoints/n6_lambda1over3.pkl",
        help="Where to save final checkpoint (pickle).",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional checkpoint to resume from (same pickle format).",
    )
    p.add_argument(
        "--init-walkers",
        type=int,
        default=64,
        help="Walkers used only for initializing network parameters.",
    )

    args = p.parse_args()

    # -------------------------------------------------------------------------
    # 1. RNG
    # -------------------------------------------------------------------------
    key = jax.random.PRNGKey(args.seed)

    # -------------------------------------------------------------------------
    # 2. Load confining potential and check metadata
    # -------------------------------------------------------------------------
    pot_path = Path(args.potential_table)
    if not pot_path.exists():
        raise FileNotFoundError(
            f"{pot_path} not found. Generate it via scripts/precompute_disk_potential.py "
            f"(e.g. --N {args.N} --nu {args.nu} --B {args.B} --d 0.1 --out {pot_path})."
        )

    print(f"Loading disk potential from {pot_path}")
    table = load_disk_potential(
        str(pot_path),
        expected_nu=args.nu,
        expected_B=args.B,
    )

    # Consistency check: a from (N, nu, B) vs a in table
    a_expected = disk_radius_for_filling(args.N, filling=args.nu, B=args.B)
    if not np.isclose(a_expected, float(table.a), rtol=1e-5, atol=1e-8):
        print(
            f"WARNING: table radius a={float(table.a):.6f} "
            f"!= computed a={a_expected:.6f} from (N={args.N}, nu={args.nu}, B={args.B})"
        )
    a = float(table.a)

    # -------------------------------------------------------------------------
    # 3. Build ansatz and Hamiltonian
    # -------------------------------------------------------------------------
    ansatz = FQHPsiformerAnsatz(
        n_electrons=args.N,
        n_dets=args.n_dets,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        head_dim=args.head_dim,
        hidden_dim=args.hidden_dim,
    )

    def wf_apply(params, phys_conf):
        return ansatz.apply(params, phys_conf)

    wf_apply = jax.jit(wf_apply)

    hamil = FQHDiskHamiltonian(
        n_electrons=args.N,
        lam=args.lam,
        B=args.B,
        a=a,
        table=table,
    )

    # -------------------------------------------------------------------------
    # 4. Initialize or resume parameters
    # -------------------------------------------------------------------------
    if args.resume is not None:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        with open(ckpt_path, "rb") as f:
            params = pickle.load(f)["params"]
        print(f"Resumed parameters from {ckpt_path}")
    else:
        key, init_key = jax.random.split(key)
        # Use Hamiltonian's init_sample to get a PhysicalConfiguration
        phys0 = hamil.init_sample(init_key, hamil.background_R, n=args.init_walkers)
        params = ansatz.init(init_key, phys0)
        print("Initialized network parameters from scratch.")

    # -------------------------------------------------------------------------
    # 5. Run VMC training
    # -------------------------------------------------------------------------
    if args.optimizer == "adam":
        print("Using Adam VMC optimizer.")
        params, sampler_state = train_vmc(
            key=key,
            params=params,
            wf_apply=wf_apply,
            hamil=hamil,
            n_steps=args.n_steps,
            n_walkers=args.n_walkers,
            mcmc_steps_per_iter=args.mcmc_steps_per_iter,
            step_size=args.step_size,
            opt_lr=args.opt_lr,
            log_interval=args.log_interval,
            logdir=args.logdir,
        )
    else:
        print("Using KFAC VMC optimizer.")
        params, sampler_state = train_vmc_kfac(
            key=key,
            params=params,
            wf_apply=wf_apply,
            hamil=hamil,
            n_steps=args.n_steps,
            n_walkers=args.n_walkers,
            mcmc_steps_per_iter=args.mcmc_steps_per_iter,
            step_size=args.step_size,
            opt_lr=args.opt_lr,
            damping=args.kfac_damping,
            norm_constraint=args.kfac_norm_constraint,
            momentum=args.kfac_momentum,
            curvature_ema=args.kfac_curvature_ema,
            inverse_update_period=args.kfac_inverse_update_period,
            log_interval=args.log_interval,
            logdir=args.logdir,
        )

    # -------------------------------------------------------------------------
    # 6. Save checkpoint (compatible with scripts/eval_checkpoint.py)
    # -------------------------------------------------------------------------
    ckpt_out = Path(args.ckpt_out)
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_out, "wb") as f:
        pickle.dump({"params": params}, f)

    print(f"Saved checkpoint to {ckpt_out}")


if __name__ == "__main__":
    main()
