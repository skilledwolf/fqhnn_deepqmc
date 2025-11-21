Project: fqhnn-deepqmc — Psiformer + Jastrow VMC for FQH disk systems (JAX/Haiku, DeepQMC).

Please review the following conceptual issues I found. Confirm, correct, or expand; suggest minimal fixes.

1) Magnetic kinetic-energy cross term sign
- File: fqhnn_deepqmc/hamiltonian.py:95-101
```python
grad_v_e = grad_v.reshape(r_single.shape)
A = vector_potential(r_single[None, ...], B=self.B)[0]
A2 = jnp.sum(A**2)
A_dot_gradv = jnp.sum(A * grad_v_e)
T_loc = 0.5 * (-lap_u - norm_grad_u2 + norm_grad_v2 + A2 - 2.0 * A_dot_gradv)
```
For H = ½(-i∇ + A)^2 the real local energy should contain +2 A·∇v, not -2 A·∇v. Current sign penalizes correct chirality; optimizer drifts to unphysical states.

2) Jastrow masking bug introduces O(N^2) constant term
- File: fqhnn_deepqmc/jastrow.py:16-19
```python
mask = jnp.triu(jnp.ones((N, N)), k=1)
r_ij_masked = r_ij * mask[None, :, :]
term = -beta * alpha**2 / (alpha + r_ij_masked + 1e-12)
return jnp.sum(term, axis=(1, 2))
```
Mask is applied only to r_ij; masked entries become zero so every diagonal/lower-tri element contributes constant -beta*alpha. Should mask the “term” or index i<j before summation.

3) Disk potential table and Hamiltonian disagree on magnetic field
- Table generation (scripts/precompute_disk_potential.py:129): a = sqrt(2*N/nu) assumes B=1.
- Runtime (fqhnn_deepqmc/geometry.py:8-12): a = sqrt(2*N/(nu*B)).
Changing B during training uses Vc(r) computed for a different a, breaking neutrality and background self-energy consistency. Need table keyed by B or enforce same B everywhere.

4) No LLL projection / holomorphic constraint
- Hamiltonian keeps full kinetic term with vector potential; ansatz is generic complex transformer + Gaussian envelope, not constrained to the Lowest Landau Level. Intended FQH problem is usually LLL-projected (kinetic is constant; only interactions matter). Current setup mixes Landau levels and targets a different Hamiltonian than Laughlin physics.

5) Checkpoint evaluation ansatz shape mismatch
- Training (experiments/run_n6_lambda_1over3.py): n_dets=16.
- Evaluation (scripts/eval_checkpoint.py:79-87): n_dets=12 hard-coded.
Loading saved params into different ansatz size will fail or force re-training; pipeline inconsistent.

If you see further subtle or physics-level issues, please add them.
