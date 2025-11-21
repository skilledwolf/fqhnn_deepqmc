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
    """
    Fractional quantum Hall disk Hamiltonian wired into deepqmc protocols.

    Parameters
    ----------
    n_electrons : int
        Number of electrons N.
    lam : float
        Coulomb prefactor λ_C = 1/ε (paper’s ε).
    B : float
        Magnetic field (cyclotron frequency ω_c = B in these units).
    a : float
        Disk radius (in ℓ_B units if you choose B=1 and use disk_radius_for_filling).
    table : DiskPotentialTable
        Precomputed confining potential V_c(r) table (without λ/ε factor).
    """

    def __init__(
        self,
        n_electrons: int,
        lam: float,
        B: float,
        a: float,
        table: DiskPotentialTable,
    ):
        self.n_electrons = n_electrons
        self.lam = float(lam)
        self.B = float(B)
        self.a = float(a)
        self.table = table
        # dummy "molecule" for samplers expecting coords
        self.mol = SimpleNamespace(coords=jnp.zeros((1, 2)))

    @property
    def background_R(self) -> jnp.ndarray:
        return self.mol.coords

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def init_sample(
        self,
        rng: jax.Array,
        R: jnp.ndarray,
        n: int,
    ) -> PhysicalConfiguration:
        r0 = init_electron_configs(rng, n, self.n_electrons, self.a, scale=0.5)
        R_tiled = jnp.tile(self.background_R[None, ...], (n, 1, 1))
        return PhysicalConfiguration(
            R=R_tiled,
            r=r0,
            mol_idx=jnp.zeros(n, dtype=jnp.int32),
        )

    def _phys_from_r(self, r_single: jnp.ndarray) -> PhysicalConfiguration:
        return PhysicalConfiguration(
            R=self.background_R,
            r=r_single,
            mol_idx=jnp.array(0),
        )

    def _make_log_psi_complex(
        self, wf_apply: ParametrizedWaveFunction
    ) -> Callable[[dict, jnp.ndarray], jnp.ndarray]:
        """
        Build a closure returning log ψ (complex) for a single configuration r_single.

        Returns
        -------
        log_psi_complex(params, r_single) -> complex scalar
        """

        def log_psi_complex(params: dict, r_single: jnp.ndarray) -> jnp.ndarray:
            psi: Psi = wf_apply(params, self._phys_from_r(r_single))
            # psi.sign is unit-modulus complex phase, psi.log = log|ψ|
            phase = jnp.angle(psi.sign)
            return (psi.log + 1j * phase).squeeze()

        return log_psi_complex

    # -------------------------------------------------------------------------
    # Local energy
    # -------------------------------------------------------------------------

    def local_energy(
        self, wf_apply: ParametrizedWaveFunction
    ) -> Callable[
        [Optional[jax.Array], dict, PhysicalConfiguration],
        tuple[jnp.ndarray, dict],
    ]:
        """
        Return a batched local energy function:

            local_energy_fn(rng_key, params, phys_conf) -> (energies, stats)

        where energies is [n_walkers] and stats includes mean/std of E.
        """
        log_psi_complex = self._make_log_psi_complex(wf_apply)
        V_b = background_self_energy(self.n_electrons, self.a, self.lam)

        def local_energy_single(params: dict, r_single: jnp.ndarray) -> jnp.ndarray:
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

            # Kinetic energy T_loc = 1/2 Re[ψ^{-1} (-i∇ + A)^2 ψ]
            T_loc = 0.5 * (
                -lap_u
                - norm_grad_u2
                + norm_grad_v2
                + A2
                - 2.0 * A_dot_gradv
            )

            R_batch = r_single[None, ...]
            V_ee = electron_electron_potential(R_batch, lam=self.lam)[0]
            V_c = disk_potential_energy(R_batch, self.table, lam=self.lam)[0]

            return T_loc + V_ee + V_c + V_b

        def local_energy_batch(
            rng_key,
            params: dict,
            phys_conf: PhysicalConfiguration,
        ):
            energies = jax.vmap(lambda r: local_energy_single(params, r))(phys_conf.r)
            stats = {
                "local_energy/mean": jnp.mean(energies),
                "local_energy/std": jnp.std(energies),
            }
            return energies, stats

        return local_energy_batch

    # -------------------------------------------------------------------------
    # Angular momentum Lz (canonical)
    # -------------------------------------------------------------------------

    def angular_momentum(
        self, wf_apply: ParametrizedWaveFunction
    ) -> Callable[
        [Optional[jax.Array], dict, PhysicalConfiguration],
        tuple[jnp.ndarray, dict],
    ]:
        """
        Return a batched local Lz function:

            Lz_fn(rng_key, params, phys_conf) -> (Lz_values, stats)

        We measure the canonical orbital angular momentum

            L_z = sum_j (x_j p_{y,j} - y_j p_{x,j}),
            p = -i ∇

        in units where ħ = 1 (atomic units).  On the level of a local estimator:

            Lz_loc(R) = sum_j (x_j ∂_{y_j} v - y_j ∂_{x_j} v),

        where v = Im[log ψ(R)] is the phase of the wavefunction.  This is the
        real part of (Lz ψ)/ψ, and its Monte Carlo average yields ⟨L_z⟩.
        """
        log_psi_complex = self._make_log_psi_complex(wf_apply)

        def Lz_single(params: dict, r_single: jnp.ndarray) -> jnp.ndarray:
            """
            r_single: [N, 2]
            """
            X = r_single.reshape(-1)

            def v_fn(X_flat):
                r_flat = X_flat.reshape(r_single.shape)
                return jnp.imag(log_psi_complex(params, r_flat))

            grad_v = jax.grad(v_fn)(X).reshape(r_single.shape)  # [N, 2]

            x = r_single[:, 0]
            y = r_single[:, 1]
            dv_dx = grad_v[:, 0]
            dv_dy = grad_v[:, 1]

            # Lz_loc = ∑ (x dv_dy - y dv_dx)
            return jnp.sum(x * dv_dy - y * dv_dx)

        def Lz_batch(
            rng_key,
            params: dict,
            phys_conf: PhysicalConfiguration,
        ):
            Lz_vals = jax.vmap(lambda r: Lz_single(params, r))(phys_conf.r)
            stats = {
                "Lz/mean": jnp.mean(Lz_vals),
                "Lz/std": jnp.std(Lz_vals),
            }
            return Lz_vals, stats

        return Lz_batch
