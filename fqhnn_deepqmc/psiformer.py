from __future__ import annotations

from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp

# Avoid truncated normal (uses erf not supported on Metal); use normal init instead.
DEFAULT_W_INIT = hk.initializers.VarianceScaling(
    scale=1.0,
    mode="fan_in",
    distribution="normal",
)


def one_electron_features(R: jnp.ndarray) -> jnp.ndarray:
    """Simple one-electron features: (x, y, r, r^2)."""
    r2 = jnp.sum(R**2, axis=-1, keepdims=True)
    r = jnp.sqrt(r2 + 1e-12)
    return jnp.concatenate([R, r, r2], axis=-1)


class SelfAttentionBlock(hk.Module):
    """Multi-head self-attention + MLP with residual (no layernorm)."""

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        mlp_hidden_dim: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.mlp_hidden_dim = mlp_hidden_dim

    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        B, N, D = h.shape
        H = self.n_heads
        Hd = self.head_dim
        total_dim = H * Hd

        q = hk.Linear(total_dim, name="q", w_init=DEFAULT_W_INIT)(h)
        k = hk.Linear(total_dim, name="k", w_init=DEFAULT_W_INIT)(h)
        v = hk.Linear(total_dim, name="v", w_init=DEFAULT_W_INIT)(h)

        def split_heads(x):
            return x.reshape(B, N, H, Hd)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scale = 1.0 / jnp.sqrt(Hd)
        scores = jnp.einsum("bnhd,bmhd->bhnm", q, k) * scale
        attn = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhnm,bmhd->bnhd", attn, v)
        context = context.reshape(B, N, total_dim)

        out = hk.Linear(D, name="out", w_init=DEFAULT_W_INIT)(context)
        h = h + out

        mlp = hk.Sequential(
            [
                hk.Linear(self.mlp_hidden_dim, w_init=DEFAULT_W_INIT),
                jax.nn.gelu,
                hk.Linear(D, w_init=DEFAULT_W_INIT),
            ]
        )
        return h + mlp(h)


class PsiformerCore(hk.Module):
    """
    Psiformer core: attention layers taking one-electron features to
    per-electron hidden states, then to complex orbitals with Gaussian envelopes.
    """

    def __init__(
        self,
        n_electrons: int,
        n_layers: int = 2,
        n_heads: int = 4,
        head_dim: int = 64,
        hidden_dim: int = 256,
        n_dets: int = 16,
        n_orbitals: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.n_electrons = n_electrons
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.n_dets = n_dets
        self.n_orbitals = n_orbitals or n_electrons

    def __call__(self, R: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        R : [B, N, 2] or [N, 2]

        Returns
        -------
        orbitals : [B, n_dets, N, n_orbitals] complex
        """
        if R.ndim == 2:
            R = R[None, ...]
        B, N, _ = R.shape
        assert N == self.n_electrons

        feats = one_electron_features(R)
        h = hk.Linear(self.hidden_dim, w_init=DEFAULT_W_INIT)(feats)

        for _ in range(self.n_layers):
            block = SelfAttentionBlock(
                n_heads=self.n_heads,
                head_dim=self.head_dim,
                mlp_hidden_dim=self.hidden_dim,
            )
            h = block(h)

        n_orb = self.n_orbitals

        real_head = hk.Linear(
            self.n_dets * n_orb,
            name="real_orbitals",
            w_init=DEFAULT_W_INIT,
        )
        imag_head = hk.Linear(
            self.n_dets * n_orb,
            name="imag_orbitals",
            w_init=DEFAULT_W_INIT,
        )

        orb_r = real_head(h).reshape(B, N, self.n_dets, n_orb)
        orb_i = imag_head(h).reshape(B, N, self.n_dets, n_orb)

        log_sigma = hk.get_parameter("log_sigma", shape=[n_orb], init=jnp.zeros)
        sigma = jnp.exp(log_sigma)

        r2 = jnp.sum(R**2, axis=-1, keepdims=True)[..., None]
        envelope = jnp.exp(-0.5 * r2 * sigma[None, None, None, :])

        phi = (orb_r + 1j * orb_i) * envelope

        return jnp.swapaxes(phi, 1, 2)
