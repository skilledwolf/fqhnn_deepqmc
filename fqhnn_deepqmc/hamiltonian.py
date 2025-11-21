from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from deepqmc.types import ParametrizedWaveFunction, PhysicalConfiguration, Psi

from .geometry import init_electron_configs
from .potential import (
    DiskPotentialTable,
    background_self_energy,
    disk_potential_energy,
    electron_electron_potential,
)


def vector_potential(R: jnp.ndarray, B: float = 1.0) -> jnp.ndarray:
    """Symmetric gauge A = (-By/2, Bx/2)."""
    x = R[..., 0]
    y = R[..., 1]
    Ax = -0.5 * B * y
    Ay = 0.5 * B * x
    return jnp.stack([Ax, Ay], axis=-1)


class FQHDiskHamiltonian:
    """Fractional quantum Hall disk Hamiltonian wired into deepqmc protocols."""

    def __init__(
        self,
        n_electrons: int,
        lam: float,
        B: float,
        a: float,
        table: DiskPotentialTable,
    ):
        self.n_electrons = n_electrons
        self.lam = lam
        self.B = B
        self.a = a
        self.table = table
        # dummy "molecule" for samplers expecting coords
        self.mol = SimpleNamespace(coords=jnp.zeros((1, 2)))

    @property
    def background_R(self) -> jnp.ndarray:
        return self.mol.coords

    def init_sample(self, rng: jax.Array, R: jnp.ndarray, n: int) -> PhysicalConfiguration:
        r0 = init_electron_configs(rng, n, self.n_electrons, self.a, scale=0.5)
        R_tiled = jnp.tile(self.background_R[None, ...], (n, 1, 1))
        return PhysicalConfiguration(R=R_tiled, r=r0, mol_idx=jnp.zeros(n, dtype=jnp.int32))

    def _phys_from_r(self, r_single: jnp.ndarray) -> PhysicalConfiguration:
        return PhysicalConfiguration(R=self.background_R, r=r_single, mol_idx=jnp.array(0))

    def local_energy(
        self, wf_apply: ParametrizedWaveFunction
    ) -> Callable[[Optional[jax.Array], dict, PhysicalConfiguration], tuple[jnp.ndarray, dict]]:
        V_b = background_self_energy(self.n_electrons, self.a, self.lam)

        def log_psi_complex(params, r_single):
            psi: Psi = wf_apply(params, self._phys_from_r(r_single))
            phase = jnp.angle(psi.sign)
            return (psi.log + 1j * phase).squeeze()

        def local_energy_single(params, r_single):
            """
            r_single: [N, 2]
            """
            log_psi = log_psi_complex(params, r_single)
            u = jnp.real(log_psi)
            v = jnp.imag(log_psi)

            X = r_single.reshape(-1)

            def u_fn(X_flat):
                r_flat = X_flat.reshape(r_single.shape)
                return jnp.real(log_psi_complex(params, r_flat))

            def v_fn(X_flat):
                r_flat = X_flat.reshape(r_single.shape)
                return jnp.imag(log_psi_complex(params, r_flat))

            grad_u = jax.grad(u_fn)(X)
            grad_v = jax.grad(v_fn)(X)
            hess_u = jax.jacfwd(jax.grad(u_fn))(X)
            lap_u = jnp.trace(hess_u)

            norm_grad_u2 = jnp.dot(grad_u, grad_u)
            norm_grad_v2 = jnp.dot(grad_v, grad_v)

            grad_v_e = grad_v.reshape(r_single.shape)
            A = vector_potential(r_single[None, ...], B=self.B)[0]
            A2 = jnp.sum(A**2)
            A_dot_gradv = jnp.sum(A * grad_v_e)

            # For H = 0.5 * (-i∇ + A)^2 the real part carries a -2 A·∇v cross term.
            T_loc = 0.5 * (-lap_u - norm_grad_u2 + norm_grad_v2 + A2 - 2.0 * A_dot_gradv)

            R_batch = r_single[None, ...]
            V_ee = electron_electron_potential(R_batch, lam=self.lam)[0]
            V_c = disk_potential_energy(R_batch, self.table, lam=self.lam)[0]

            return T_loc + V_ee + V_c + V_b

        def local_energy_batch(rng_key, params, phys_conf: PhysicalConfiguration):
            energies = jax.vmap(lambda r: local_energy_single(params, r))(phys_conf.r)
            stats = {
                "local_energy/mean": jnp.mean(energies),
                "local_energy/std": jnp.std(energies),
            }
            return energies, stats

        return local_energy_batch
