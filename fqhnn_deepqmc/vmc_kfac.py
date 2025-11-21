from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from deepqmc.types import PhysicalConfiguration
from kfac_jax import optimizer as kfac_optim
from kfac_jax._src import loss_functions

from .vmc import build_phys_conf, make_sampler


def train_vmc_kfac(
    key: jax.Array,
    params: dict,
    wf_apply: Callable[[dict, PhysicalConfiguration], object],
    hamil,
    n_steps: int,
    n_walkers: int,
    mcmc_steps_per_iter: int = 10,
    step_size: float = 0.01,
    opt_lr: float = 1e-3,
    damping: float = 1e-3,
    norm_constraint: float = 1e-3,
    momentum: float = 0.9,
    curvature_ema: float = 0.95,
    inverse_update_period: int = 10,
    log_interval: int = 50,
    logdir: str | None = None,
):
    """
    KFAC-based VMC training loop (aligned with the paper setup).

    Parameters
    ----------
    key : PRNGKey
        JAX RNG for sampling and optimizer randomness.
    params : dict
        Initial network parameters.
    wf_apply : callable
        Wavefunction apply(params, PhysicalConfiguration) -> Psi.
    hamil : FQHDiskHamiltonian
    n_steps : int
        Optimizer steps.
    n_walkers : int
        Number of Metropolis walkers.
    mcmc_steps_per_iter : int
        Number of MCMC refreshes per optimizer step.
    step_size : float
        Metropolis proposal scale (tau).
    opt_lr : float
        KFAC learning rate.
    damping : float
        KFAC damping (Tikhonov) parameter.
    norm_constraint : float
        Trust-region constraint on the update norm.
    momentum : float
        Momentum used by KFAC.
    curvature_ema : float
        EMA factor for curvature estimates.
    inverse_update_period : int
        How often to refresh the inverse curvature blocks.
    """
    sampler = make_sampler(hamil, wf_apply, step_size=step_size)
    key, init_key = jax.random.split(key)
    sampler_state = sampler.init(init_key, params, n_walkers, hamil.background_R)
    local_energy_fn = hamil.local_energy(wf_apply)

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
        mean_e = jnp.mean(energies)
        aux = {
            "energy_mean": mean_e,
            "energy_std": jnp.std(energies),
        }
        # Register a loss tag so KFAC's tracer sees a valid loss.  We use a
        # squared-error loss on the per-walker energies against zero; this
        # supplies curvature information without altering the returned loss.
        loss_functions.register_squared_error_loss(
            prediction=energies,
            targets=jnp.zeros_like(energies),
            weight=0.5,
        )
        return mean_e, aux

    loss_and_grads = jax.value_and_grad(loss_fn, has_aux=True)

    def value_and_grad_func(p, batch):
        return loss_and_grads(p, batch)

    lr_schedule = lambda step: opt_lr
    damping_schedule = lambda step: damping
    momentum_schedule = lambda step: momentum

    # KFAC handles compilation internally; keep the value_and_grad_fn pure (unjitted).
    optimizer = kfac_optim.Optimizer(
        value_and_grad_func=value_and_grad_func,
        l2_reg=0.0,
        value_func_has_aux=True,
        value_func_has_rng=False,
        learning_rate_schedule=lr_schedule,
        momentum_schedule=momentum_schedule,
        damping_schedule=damping_schedule,
        initial_damping=damping,
        min_damping=damping,
        norm_constraint=norm_constraint,
        estimation_mode="fisher_gradients",
        curvature_ema=curvature_ema,
        curvature_update_period=1,
        inverse_update_period=inverse_update_period,
        num_burnin_steps=0,  # avoid data_iterator requirement
        multi_device=False,
    )

    # Need an initial batch for optimizer state.
    key, sample_key = jax.random.split(key)
    sampler_state, phys_conf, smpl_stats = mcmc_once(sampler_state, sample_key, params)
    initial_batch = phys_conf.r

    key, opt_rng = jax.random.split(key)
    opt_state = optimizer.init(params, opt_rng, initial_batch)

    def log(step, energies, smpl_stats, stats_dict):
        mean_e = float(jnp.mean(energies))
        std_e = float(jnp.std(energies))
        acc = float(smpl_stats["sampling/acceptance"])
        print(
            f"step {step:6d}  E = {mean_e:+.6f}  std = {std_e:.6f}  acc = {acc:.3f}"
        )
        if writer is not None:
            writer.add_scalar("energy/mean", mean_e, step)
            writer.add_scalar("energy/std", std_e, step)
            writer.add_scalar("sampling/acceptance", acc, step)
            writer.add_scalar("sampling/tau", float(sampler_state["tau"]), step)
            if "qmodel_change" in stats_dict:
                writer.add_scalar("kfac/qmodel_change", float(stats_dict["qmodel_change"]), step)
            if "gradient_norm" in stats_dict:
                writer.add_scalar("kfac/grad_norm", float(stats_dict["gradient_norm"]), step)

    for step in range(n_steps):
        for _ in range(mcmc_steps_per_iter):
            key, sk = jax.random.split(key)
            sampler_state, phys_conf, smpl_stats = mcmc_once(sampler_state, sk, params)

        batch = phys_conf.r
        key, opt_key = jax.random.split(key)

        step_out = optimizer.step(params, opt_state, rng=opt_key, batch=batch)
        if len(step_out) == 4:
            params, opt_state, _, stats = step_out
        else:
            params, opt_state, stats = step_out

        if step % log_interval == 0:
            energies, _ = local_energy_fn(None, params, build_phys_conf(batch, hamil.background_R))
            log(step, energies, smpl_stats, stats)

    if writer is not None:
        writer.close()

    return params, sampler_state
