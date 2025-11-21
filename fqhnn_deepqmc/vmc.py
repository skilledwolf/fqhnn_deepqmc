from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from deepqmc.sampling.electron_samplers import MetropolisSampler
from deepqmc.types import PhysicalConfiguration


def build_phys_conf(r_batch: jnp.ndarray, R_template: jnp.ndarray) -> PhysicalConfiguration:
    R_tiled = jnp.tile(R_template[None, ...], (r_batch.shape[0], 1, 1))
    return PhysicalConfiguration(R=R_tiled, r=r_batch, mol_idx=jnp.zeros(r_batch.shape[0], dtype=jnp.int32))


def make_sampler(hamil, wf_apply: Callable[[dict, PhysicalConfiguration], object], step_size: float) -> MetropolisSampler:
    return MetropolisSampler(hamil, wf_apply, tau=step_size, target_acceptance=0.57)


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
    sampler = make_sampler(hamil, wf_apply, step_size=step_size)
    key, init_key = jax.random.split(key)
    sampler_state = sampler.init(init_key, params, n_walkers, hamil.background_R)
    local_energy_fn = hamil.local_energy(wf_apply)

    opt = optax.adam(opt_lr)
    opt_state = opt.init(params)

    writer = None
    if logdir is not None:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(logdir)

    @jax.jit
    def mcmc_once(state, rng, params):
        state, phys_conf, smpl_stats = sampler.sample(rng, state, params, hamil.background_R)
        return state, phys_conf, smpl_stats

    def loss_fn(p, r_batch):
        phys_conf = build_phys_conf(r_batch, hamil.background_R)
        energies, _ = local_energy_fn(None, p, phys_conf)
        return jnp.mean(energies), energies

    loss_and_grads = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    for step in range(n_steps):
        for _ in range(mcmc_steps_per_iter):
            key, sk = jax.random.split(key)
            sampler_state, phys_conf, smpl_stats = mcmc_once(sampler_state, sk, params)

        (loss, energies), grads = loss_and_grads(params, phys_conf.r)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if step % log_interval == 0:
            print(
                f"step {step:6d}  E = {float(jnp.mean(energies)):+.6f}  "
                f"std = {float(jnp.std(energies)):.6f}  acc = {float(smpl_stats['sampling/acceptance']):.3f}"
            )
            if writer is not None:
                writer.add_scalar("energy/mean", float(jnp.mean(energies)), step)
                writer.add_scalar("energy/std", float(jnp.std(energies)), step)
                writer.add_scalar("sampling/acceptance", float(smpl_stats["sampling/acceptance"]), step)
                writer.add_scalar("sampling/tau", float(sampler_state["tau"]), step)

    if writer is not None:
        writer.close()

    return params, sampler_state
