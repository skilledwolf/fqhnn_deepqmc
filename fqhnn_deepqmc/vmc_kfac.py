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
    KFAC-based VMC training loop.

    Logs (every log_interval steps):
      - raw energy ⟨E⟩ and std,
      - energy per particle in units of 1/(ε ℓ_B) = lam * sqrt(B),
      - reduced energy per particle (E/N - ω_c/2) in the same units,
      - total angular momentum Lz and Lz/N.

    For comparison with the paper:
      λ_paper = lam / sqrt(B),
      energy unit 1/(ε ℓ_B) = lam * sqrt(B).
    """
    sampler = make_sampler(hamil, wf_apply, step_size=step_size)
    key, init_key = jax.random.split(key)
    sampler_state = sampler.init(init_key, params, n_walkers, hamil.background_R)

    local_energy_fn = hamil.local_energy(wf_apply)
    Lz_fn = hamil.angular_momentum(wf_apply)

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
        mean_e = jnp.mean(energies)
        aux = {
            "energy_mean": mean_e,
            "energy_std": jnp.std(energies),
        }
        # Register a loss tag so KFAC's tracer sees a valid loss; this does not
        # change the returned loss, but supplies curvature information.
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

    # Initial batch for optimizer state
    key, sample_key = jax.random.split(key)
    sampler_state, phys_conf, smpl_stats = mcmc_once(sampler_state, sample_key, params)
    initial_batch = phys_conf.r

    key, opt_rng = jax.random.split(key)
    opt_state = optimizer.init(params, opt_rng, initial_batch)

    N = hamil.n_electrons
    lam = jnp.asarray(hamil.lam)
    B = jnp.asarray(hamil.B)
    E_unit = lam * jnp.sqrt(B)  # 1 / (ε ℓ_B)

    for step in range(n_steps):
        # Refresh walkers
        for _ in range(mcmc_steps_per_iter):
            key, sk = jax.random.split(key)
            sampler_state, phys_conf, smpl_stats = mcmc_once(
                sampler_state,
                sk,
                params,
            )

        batch = phys_conf.r
        key, opt_key = jax.random.split(key)

        step_out = optimizer.step(params, opt_state, rng=opt_key, batch=batch)
        if len(step_out) == 4:
            params, opt_state, _, stats = step_out
        else:
            params, opt_state, stats = step_out

        if step % log_interval == 0:
            phys_eval = build_phys_conf(batch, hamil.background_R)
            energies, _ = local_energy_fn(None, params, phys_eval)
            E_mean = jnp.mean(energies)
            E_std = jnp.std(energies)
            E_pp = E_mean / N

            E_pp_paper = E_pp / E_unit
            E_pp_reduced = (E_pp - 0.5 * B) / E_unit

            # Angular momentum
            Lz_vals, _ = Lz_fn(None, params, phys_eval)
            Lz_mean = jnp.mean(Lz_vals)
            Lz_std = jnp.std(Lz_vals)
            Lz_pp = Lz_mean / N

            acc = smpl_stats["sampling/acceptance"]

            E_mean_f = float(E_mean)
            E_std_f = float(E_std)
            E_pp_paper_f = float(E_pp_paper)
            E_pp_reduced_f = float(E_pp_reduced)
            acc_f = float(acc)
            Lz_mean_f = float(Lz_mean)
            Lz_std_f = float(Lz_std)
            Lz_pp_f = float(Lz_pp)

            print(
                f"[KFAC] step {step:6d}  "
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
                if "qmodel_change" in stats:
                    writer.add_scalar(
                        "kfac/qmodel_change", float(stats["qmodel_change"]), step
                    )
                if "gradient_norm" in stats:
                    writer.add_scalar(
                        "kfac/grad_norm", float(stats["gradient_norm"]), step
                    )

    if writer is not None:
        writer.close()

    return params, sampler_state
