from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def disk_radius_for_filling(
    n_electrons: int, filling: float = 1.0 / 3.0, B: float = 1.0
) -> float:
    """Disk radius a (l_B units) from the flux condition."""
    return float(np.sqrt(2.0 * n_electrons / (filling * B)))


def init_electron_configs(
    key: jax.Array,
    n_walkers: int,
    n_electrons: int,
    a: float,
    scale: float = 0.5,
) -> jax.Array:
    """Sample electron positions uniformly in a disk of radius ``a``."""
    key_r, key_theta = jax.random.split(key)
    r = (scale * a) * jnp.sqrt(jax.random.uniform(key_r, (n_walkers, n_electrons, 1)))
    theta = 2.0 * jnp.pi * jax.random.uniform(key_theta, (n_walkers, n_electrons, 1))
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    return jnp.concatenate([x, y], axis=-1)
