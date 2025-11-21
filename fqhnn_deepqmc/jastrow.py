from __future__ import annotations

import jax.numpy as jnp


def pairwise_distances(R: jnp.ndarray) -> jnp.ndarray:
    """Pairwise distances r_ij for a batch of configurations."""
    diff = R[:, :, None, :] - R[:, None, :, :]
    r2 = jnp.sum(diff**2, axis=-1)
    return jnp.sqrt(jnp.maximum(r2, 0.0))


def jastrow_log(R: jnp.ndarray, alpha: float, beta: float = 0.25) -> jnp.ndarray:
    r_ij = pairwise_distances(R)
    _, N, _ = r_ij.shape
    mask = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
    denom = alpha + r_ij + 1e-12
    term = -beta * alpha**2 / denom
    term = jnp.where(mask[None, :, :], term, 0.0)  # zero out diagonal/self-terms
    return jnp.sum(term, axis=(1, 2))
