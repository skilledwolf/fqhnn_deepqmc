# fqhnn_deepqmc/potential.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import jax.numpy as jnp

from .geometry import disk_radius_for_filling


@dataclass
class DiskPotentialTable:
    r_grid: jnp.ndarray  # [M]
    V_grid: jnp.ndarray  # [M]  (bare potential, no λ)
    a: float
    d: float
    N: int
    nu: float | None = None
    B: float | None = None


def load_disk_potential(
    path: str,
    expected_nu: float | None = None,
    expected_B: float | None = None,
    allow_repair: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> DiskPotentialTable:
    """
    Load a disk potential table and sanity-check metadata.

    Parameters
    ----------
    path : str
        .npz file produced by scripts/precompute_disk_potential.py.
    expected_nu, expected_B : float, optional
        If provided, assert the table matches these values (and recompute
        ``a = sqrt(2N/(nu B))`` to catch radius mismatches).
    allow_repair : bool
        If True, non-finite V entries are linearly interpolated; otherwise an
        error is raised. Defaults to False (fail fast).
    """
    data = np.load(path)

    def _get(name, default=None):
        return data[name] if name in data.files else default

    r_grid_np = _get("r_grid")
    V_grid_np = _get("V_grid")
    if r_grid_np is None or V_grid_np is None:
        raise ValueError(f"{path} missing r_grid/V_grid")

    finite = np.isfinite(V_grid_np)
    if not np.all(finite):
        if not allow_repair:
            raise RuntimeError("Non-finite entries in disk potential table — check precompute step.")
        V_grid_np = np.interp(r_grid_np, r_grid_np[finite], V_grid_np[finite])

    a = float(_get("a"))
    d = float(_get("d"))
    N = int(_get("N"))
    nu = _get("nu", None)
    nu = float(nu) if nu is not None else None
    B = _get("B", None)
    B = float(B) if B is not None else None

    if expected_nu is not None:
        if nu is None or not np.isclose(nu, expected_nu, rtol=rtol, atol=atol):
            raise ValueError(f"Table ν={nu} does not match expected ν={expected_nu}.")
    if expected_B is not None:
        if B is None or not np.isclose(B, expected_B, rtol=rtol, atol=atol):
            raise ValueError(f"Table B={B} does not match expected B={expected_B}.")

    # Recompute a from (N, nu, B) if provided to ensure consistency.
    if expected_nu is not None and expected_B is not None:
        a_expected = disk_radius_for_filling(N, filling=expected_nu, B=expected_B)
        if not np.isclose(a, a_expected, rtol=rtol, atol=atol):
            raise ValueError(
                f"Table radius a={a} inconsistent with recomputed a={a_expected} "
                f"from (N={N}, nu={expected_nu}, B={expected_B})."
            )

    r_grid = jnp.array(r_grid_np)
    V_grid = jnp.array(V_grid_np)
    return DiskPotentialTable(r_grid=r_grid, V_grid=V_grid, a=a, d=d, N=N, nu=nu, B=B)


def disk_potential_single(r: jnp.ndarray, table: DiskPotentialTable) -> jnp.ndarray:
    """Scalar V_c(r) for one electron, using interpolation + point-charge tail for r>15a."""
    r_abs = jnp.abs(r)
    a = table.a
    r_max = 15.0 * a

    V_interp = jnp.interp(r_abs, table.r_grid, table.V_grid)

    # Point-charge tail: -N / sqrt(d^2 + r^2)
    tail = -float(table.N) / jnp.sqrt(table.d**2 + r_abs**2 + 1e-12)
    return jnp.where(r_abs > r_max, tail, V_interp)


def disk_potential_energy(R: jnp.ndarray, table: DiskPotentialTable, lam: float) -> jnp.ndarray:
    """Electron–background energy: V_c = λ Σ_j V_c(|r_j|)."""
    r = jnp.sqrt(jnp.sum(R**2, axis=-1))          # [B, N]
    V_each = disk_potential_single(r, table)      # [B, N]
    return lam * jnp.sum(V_each, axis=-1)



def pairwise_distances(R: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    R: [B, N, 2]

    Returns
    -------
    r_ij : [B, N, N]
    diff : [B, N, N, 2]
    """
    diff = R[:, :, None, :] - R[:, None, :, :]
    r2 = jnp.sum(diff**2, axis=-1)
    r = jnp.sqrt(jnp.maximum(r2, 0.0))
    return r, diff


def electron_electron_potential(R: jnp.ndarray, lam: float) -> jnp.ndarray:
    """V_ee = λ * Σ_{i<j} 1/|r_ij|."""
    r, _ = pairwise_distances(R)
    _, N, _ = r.shape
    mask = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
    inv_r = jnp.where(mask[None, :, :], 1.0 / (r + 1e-12), 0.0)
    return lam * jnp.sum(inv_r, axis=(1, 2))



def background_self_energy(N: int, a: float, lam: float, epsilon: float = 1.0) -> float:
    """
    Background self-interaction constant (Eq. (7)):
      V_b ≈ λ * 8 N^2 / (3 π a)
    """
    return float(lam * 8.0 * N * N / (3.0 * np.pi * a))
