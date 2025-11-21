from __future__ import annotations

from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp
from deepqmc.types import Ansatz, PhysicalConfiguration, Psi

from .jastrow import jastrow_log
from .psiformer import PsiformerCore


class FQHPsiformerAnsatz(Ansatz):
    """Psiformer + Jastrow ansatz packaged for deepqmc."""

    def __init__(
        self,
        n_electrons: int,
        n_dets: int = 16,
        n_layers: int = 2,
        n_heads: int = 4,
        head_dim: int = 64,
        hidden_dim: int = 256,
    ):
        self.n_electrons = n_electrons
        self.n_dets = n_dets
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim

        def forward(R: jnp.ndarray) -> jnp.ndarray:
            """
            R: [B, N, 2] (or [N,2]); returns log_psi per batch element.
            """
            # ensure batch dim
            if R.ndim == 2:
                R = R[None, ...]

            core = PsiformerCore(
                n_electrons=self.n_electrons,
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                head_dim=self.head_dim,
                hidden_dim=self.hidden_dim,
                n_dets=self.n_dets,
                n_orbitals=self.n_electrons,
            )

            phi = core(R)  # [B, n_dets, N, N]
            eps = jnp.asarray(1e-6, dtype=phi.dtype)
            phi = phi + eps * jnp.eye(self.n_electrons, dtype=phi.dtype)[None, None, :, :]

            sign_dets, log_dets = jnp.linalg.slogdet(phi)  # [B, n_dets]

            log_w = hk.get_parameter("log_det_weights", shape=[self.n_dets], init=jnp.zeros)
            L = log_w[None, :] + log_dets  # [B, n_dets]
            L_real = jnp.real(L)
            L_max = jnp.max(L_real, axis=-1, keepdims=True)  # [B,1]
            L_max_safe = jnp.where(jnp.isneginf(L_max), jnp.zeros_like(L_max), L_max)
            scaled_terms = sign_dets * jnp.exp(L - L_max_safe)
            sum_scaled = jnp.sum(scaled_terms, axis=-1)  # [B]
            sum_safe = jnp.where(jnp.abs(sum_scaled) > 0.0, sum_scaled, 1e-30 + 0j)
            log_psi_det = L_max_safe.squeeze(-1) + jnp.log(sum_safe)  # [B]

            raw_alpha = hk.get_parameter("jastrow_raw_alpha", shape=[], init=jnp.ones)
            alpha = jax.nn.softplus(raw_alpha) + 1e-3
            log_J = jastrow_log(R, alpha=alpha, beta=0.25)  # [B]

            return log_J + log_psi_det  # [B]

        self._hk = hk.without_apply_rng(hk.transform(forward))

    def init(self, rng: jax.Array, phys_conf: PhysicalConfiguration):
        R = phys_conf.r[..., :2]
        if R.ndim == 2:
            R = R[None, ...]
        return self._hk.init(rng, R)

    def apply(
        self,
        params: dict,
        phys_conf: PhysicalConfiguration,
        return_mos: bool = False,  # kept for Ansatz API compatibility
    ) -> Psi:
        R = phys_conf.r[..., :2]
        if R.ndim == 2:
            R = R[None, ...]
        log_psi = self._hk.apply(params, R)  # [B]
        log_abs = jnp.real(log_psi)
        phase = jnp.imag(log_psi)
        # deepqmc expects Psi.sign to track phase; use unit-modulus complex phase
        sign = jnp.exp(1j * phase)
        # when batching, Psi.log must match Psi.sign shape; deepqmc vmap handles leading dims
        return Psi(sign=sign, log=log_abs)
