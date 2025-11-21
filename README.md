# fqhnn-deepqmc

Reimplementation of the original Psiformer + Jastrow fractional quantum Hall VMC using the `deepqmc` primitives (Ansatz/Psi objects, PhysicalConfiguration, Metropolis samplers).

## Environment

- New conda env: `conda create -n fqhnn_dqmc python=3.11`
- Install deps: `conda run -n fqhnn_dqmc pip install -U deepqmc jax-metal scipy matplotlib tqdm` then `conda run -n fqhnn_dqmc pip install -e .`
- On Apple Silicon, run with `ENABLE_PJRT_COMPATIBILITY=1` (Metal). If Metal hits the current `default_memory_space` JAX/Metal bug, fall back with `JAX_PLATFORMS=cpu`.

Quick sanity check:

```bash
ENABLE_PJRT_COMPATIBILITY=1 conda run -n fqhnn_dqmc python -c "import jax; print(jax.devices())"
```

You should see a Metal device (experimental warnings are expected).

## Data prep

Precompute the confining disk potential (table stores N, ν, B, d and radius a must match training):

```bash
conda run -n fqhnn_dqmc python scripts/precompute_disk_potential.py --N 6 --nu 0.333333 --B 1.0 --d 0.1 --n_points 10000 --out data/disk_N6.npz
```

## Example run (N=6, λ=1/3)

```bash
ENABLE_PJRT_COMPATIBILITY=1 conda run -n fqhnn_dqmc python experiments/run_n6_lambda_1over3.py
```

This runs a short VMC (Metropolis + optax Adam) and writes `checkpoints/n6_lambda1over3.pkl`.

To match the paper’s optimizer, switch to KFAC:

```bash
ENABLE_PJRT_COMPATIBILITY=1 conda run -n fqhnn_dqmc python experiments/run_n6_lambda_1over3.py \
  --optimizer kfac --lr 1e-3 --kfac-damping 1e-3 --kfac-norm 1e-3 --kfac-momentum 0.9
```

Heavy defaults (close to max for an M4 48 GB, still feasible on CPU if patient):

```bash
JAX_PLATFORMS=cpu conda run -n fqhnn_dqmc python experiments/run_n6_lambda_1over3.py \
  --steps 800 --walkers 4096 --mcmc-per-step 20 --step-size 0.01 --lr 2e-3
```

If Metal works on your JAX build, swap `JAX_PLATFORMS=cpu` for `ENABLE_PJRT_COMPATIBILITY=1` to hit the GPU.

## Production (server GPU) run

Tested params that fit comfortably on A100/4090-class cards:

```bash
CUDA_VISIBLE_DEVICES=0 \
JAX_PLATFORMS=gpu \
python experiments/run_n6_lambda_1over3.py \
  --steps 6000 \
  --walkers 16384 \
  --mcmc-per-step 20 \
  --step-size 0.01 \
  --lr 1e-3 \
  --log-interval 50 \
  --logdir runs/fqhnn_prod
```

Adjust `--walkers` down to 8192 if memory is tight. Keep acceptance ~0.55–0.6 by nudging `--step-size`. For longer runs increase `--steps`; for faster logging lower `--log-interval`.

## Tracking progress

- The script prints energy mean/std and acceptance every `--log-interval` steps; redirect to a log file if running long jobs: `... > logs/run.log 2>&1`.
- You can intermittently checkpoint and evaluate with `scripts/eval_checkpoint.py --steps 200 --burn 100 --walkers 512` to check radial density and S(k) without stopping training.
- TensorBoard: pass `--logdir runs/<name>` to the experiment script. Start a board with `tensorboard --logdir runs` and watch `energy/mean`, `energy/std`, `sampling/acceptance`, `sampling/tau`. CSV logging is also easy: redirect stdout to a file and plot with pandas.

## Layout

- `fqhnn_deepqmc/` core modules: Psiformer ansatz, FQH Hamiltonian, VMC loop
- `scripts/precompute_disk_potential.py` SciPy table generator for Vc(r); now hard-fails on any non-finite entries and checks Vc(0) + tail analytically
- `experiments/run_n6_lambda_1over3.py` example training script

## Modeling notes

- No LLL projection: the Hamiltonian keeps the full kinetic term with vector potential, so the wavefunction may mix Landau levels (Psiformer-style LL mixing).
- The confining potential table must be generated with the same `(N, ν, B, d)` used at runtime; `load_disk_potential` will sanity-check and raise on mismatches (no silent interpolation).
