from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import optax
from deepqmc.sampling.electron_samplers import MetropolisSampler
from deepqmc.types import PhysicalConfiguration


def build_phys_conf(
    r_batch: jnp.ndarray,
    R_template: jnp.ndarray,
) -> PhysicalConfiguration:
    R_tiled = jnp.tile(R_template[None, ...], (r_batch.shape[0], 1, 1))
    return PhysicalConfiguration(
        R=R_tiled,
        r=r_batch,
        mol_idx=jnp.zeros(r_batch.shape[0], dtype=jnp.int32),
    )


def make_sampler(
    hamil,
    wf_apply: Callable[[dict, PhysicalConfiguration], object],
    step_size: float,
) -> MetropolisSampler:
    return MetropolisSampler(
        hamil,
        wf_apply,
        tau=step_size,
        target_acceptance=0.57,
    )


def train_vmc(
    key: jax.Array,
    params: dict,
    wf_apply: Callable[[dict, PhysicalConfiguration], object],
    hamil,
    n_steps: int,
    n_walkers: int,
    mcmc_steps_per_iter: int = 10,
    step_size: float = 0.01,
    opt_lr: float = 3e-3,
    log_interval: int = 50,
    logdir: str | None = None,
):
    """
    Simple Adam-based VMC training loop.

    Logs (every log_interval steps):
      - raw energy ⟨E⟩ and std,
      - energy per particle in units of 1/(ε ℓ_B) = lam * sqrt(B),
      - reduced energy per particle (E/N - ω_c/2) in the same units,
      - total angular momentum Lz and Lz/N.

    For paper comparison:
      λ_paper = lam / sqrt(B),
      energy unit 1/(ε ℓ_B) = lam * sqrt(B).
    """
    sampler = make_sampler(hamil, wf_apply, step_size=step_size)
    key, init_key = jax.random.split(key)
    sampler_state = sampler.init(init_key, params, n_walkers, hamil.background_R)

    local_energy_fn = hamil.local_energy(wf_apply)
    Lz_fn = hamil.angular_momentum(wf_apply)

    opt = optax.adam(opt_lr)
    opt_state = opt.init(params)

    writer = None
    if logdir is not None:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(logdir)

    @jax.jit
    def mcmc_once(state, rng, params):
        state, phys_conf, smpl_stats = sampler.sample(
            rng, state, params, hamil.background_R
        )
        return state, phys_conf, smpl_stats

    def loss_fn(p, r_batch):
        phys_conf = build_phys_conf(r_batch, hamil.background_R)
        energies, _ = local_energy_fn(None, p, phys_conf)
        return jnp.mean(energies), energies

    loss_and_grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    N = hamil.n_electrons
    lam = jnp.asarray(hamil.lam)
    B = jnp.asarray(hamil.B)
    E_unit = lam * jnp.sqrt(B)  # 1 / (ε ℓ_B)

    for step in range(n_steps):
        # Refresh walkers with MCMC
        for _ in range(mcmc_steps_per_iter):
            key, sk = jax.random.split(key)
            sampler_state, phys_conf, smpl_stats = mcmc_once(
                sampler_state,
                sk,
                params,
            )

        # Compute loss and gradients on the current batch of electron positions
        (loss, energies), grads = loss_and_grads(params, phys_conf.r)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if step % log_interval == 0:
            # Energies
            E_mean = jnp.mean(energies)
            E_std = jnp.std(energies)
            E_pp = E_mean / N  # per particle (raw units)

            # Convert to paper units: 1/(ε ℓ_B) = lam * sqrt(B)
            E_pp_paper = E_pp / E_unit
            # Reduced energy per particle: (E/N - ω_c/2) / (1/(ε ℓ_B))
            E_pp_reduced = (E_pp - 0.5 * B) / E_unit

            # Angular momentum Lz (canonical)
            phys_eval = build_phys_conf(phys_conf.r, hamil.background_R)
            Lz_vals, _ = Lz_fn(None, params, phys_eval)
            Lz_mean = jnp.mean(Lz_vals)
            Lz_std = jnp.std(Lz_vals)
            Lz_pp = Lz_mean / N

            acc = smpl_stats["sampling/acceptance"]

            # Convert to Python floats for printing / logging
            E_mean_f = float(E_mean)
            E_std_f = float(E_std)
            E_pp_paper_f = float(E_pp_paper)
            E_pp_reduced_f = float(E_pp_reduced)
            acc_f = float(acc)
            Lz_mean_f = float(Lz_mean)
            Lz_std_f = float(Lz_std)
            Lz_pp_f = float(Lz_pp)

            print(
                f"step {step:6d}  "
                f"E = {E_mean_f:+.6f}  std = {E_std_f:.6f}  "
                f"E_pp[1/(εℓ_B)] = {E_pp_paper_f:+.6f}  "
                f"E_red_pp[1/(εℓ_B)] = {E_pp_reduced_f:+.6f}  "
                f"Lz = {Lz_mean_f:.3f}  Lz/N = {Lz_pp_f:.3f}  "
                f"acc = {acc_f:.3f}"
            )

            if writer is not None:
                writer.add_scalar("energy/mean", E_mean_f, step)
                writer.add_scalar("energy/std", E_std_f, step)
                writer.add_scalar("energy_pp_paper/mean", E_pp_paper_f, step)
                writer.add_scalar("energy_pp_reduced/mean", E_pp_reduced_f, step)
                writer.add_scalar("sampling/acceptance", acc_f, step)
                writer.add_scalar("sampling/tau", float(sampler_state["tau"]), step)
                writer.add_scalar("Lz/mean", Lz_mean_f, step)
                writer.add_scalar("Lz/std", Lz_std_f, step)
                writer.add_scalar("Lz_per_particle/mean", Lz_pp_f, step)

    if writer is not None:
        writer.close()

    return params, sampler_state
