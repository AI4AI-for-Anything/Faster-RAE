# Fixed-global-batch accumulation validation — 2026-10-15

This supersedes the batch-selection behavior in the [earlier capacity report](h200-capacity-2026-10-15.md); its historical measurements remain unchanged. [Aggregate evidence](h200-accumulation-2026-10-15.json) is included. Raw logs and local assets remain on the compute host.

## Configuration and scope

- Source fingerprint (all `src/**/*.py` and `tools/*.py`, paths and bytes): `dbf85452ec2a70b1f1e5aecb50ebc32ebcc88c03695916d989cc64ae3c1d74f1`.
- Host called H200; actual device reports NVIDIA L20X, SM90, 139.81 GiB. One reserved idle GPU used per run.
- Target deployment: **8 GPUs, upstream global batch 512**. Executed probe: **one rank, local effective batch 64**. No eight-GPU CUDA/NCCL capacity claim.
- Same existing isolated Python 3.10 / torch 2.8.0+cu128 environment and prepared local assets as the earlier report; preflight imports, compiler/backward and attention checks passed again.
- Historical optimization environment preset, cuDNN attention, `--precision fp32` with preset local BF16/TF32 operations. No model/resolution/loss/LR changes.
- Each successful phase: 8 effective updates, first 3 warmup; profiling disabled. Reconstruction+LPIPS, D training, and GAN generator training each start in a fresh process.
- Requested budgets 80 and 48 GiB; allocator headroom 4 GiB. Whole-device memory sampled about every 0.5 s, including setup/compilation. This is not hardware partitioning or a physical 80/48 GiB GPU measurement.

## Results

| Budget GiB | Microbatch | Accumulation | Phase | Result | Allocated / reserved peak GiB | Sampled device peak GiB |
|---:|---:|---:|---|---|---|---:|
| 80 | 64 | 1 | recon | PASS | 41.10 / 43.29 | 44.58 |
| 80 | 64 | 1 | disc | PASS | 39.63 / 42.02 | 43.32 |
| 80 | 64 | 1 | gan | PASS | 47.96 / 52.43 | 53.73 |
| 48 | 64 | 1 | recon | PASS | 41.10 / 43.29 | 44.58 |
| 48 | 64 | 1 | disc | PASS | 39.63 / 42.02 | 43.32 |
| 48 | 64 | 1 | gan | OOM | allocation rejected at 44 GiB cap | 45.09 |
| 48 | 32 | 2 | recon | PASS | 27.58 / 29.12 | 30.42 |
| 48 | 32 | 2 | disc | PASS | 26.12 / 27.66 | 28.96 |
| 48 | 32 | 2 | gan | PASS | 28.32 / 30.28 | 31.58 |

The selected plans are **8 × 64 × 1 = 512** at 80 GiB and **8 × 32 × 2 = 512** at 48 GiB. At 48 GiB the 64-image GAN attempt hit the 44 GiB allocator limit; the smaller observed device total at failure does not mean the next allocation would fit that cap. The launcher retained the failure and retried all three phases with microbatch 32 and accumulation 2.

All successful phases had finite losses and final decoder gradients/parameters. Active D phases also had finite D gradients/parameters. Optimizer post-step hooks recorded 8 G updates per phase, 8 D updates in active D phases, and 0 D updates in reconstruction-only phases. Accumulation runs recorded two microbatches per effective update. No new compiled graphs appeared after the three warmup steps. Reported timings in the JSON are short capacity diagnostics, not speedup evidence.

Both generated training configurations retain global batch 512, the source phase schedule, optimizer/LR settings and eight-rank expectation. The 48 GiB config specifies microbatch 32 and accumulation 2. The probe-only rank simulation option cannot be used for normal training; an actual world-size mismatch is rejected.

## Correctness checks and limits

`python -m unittest discover -s tests -v`: **8 tests passed** in the remote isolated environment. Checks cover batch arithmetic for 1/2/4/8/16 ranks, invalid divisibility and launch mismatch, source-config preservation, accumulated gradients/adaptive weight against a full local batch on a deterministic toy network, RNG replay, two-rank Gloo gradient/parameter agreement over multiple updates, and supervisor timeout/budget termination behavior. The repo-local skill validator and Python syntax checks passed.

G backward averages all microbatches before one optimizer, clipping, scheduler and EMA update. D then accumulates fresh reconstructions from the updated G, and uses its DDP wrapper to synchronize trainable gradients. GAN accumulation computes one rank-local adaptive weight from aggregated last-layer gradients, with a separate RNG-replayed backward pass. CPU DDP tests exercise synchronization; the GPU probes exercise real compiled models and optimizer cadence on one rank.

These checks do not establish exact full-batch numerical equivalence or convergence. The default BatchNormLocal uses per-sample groups without running statistics, but spectral-normalization buffers advance for each microbatch forward; random-window cropping and floating-point/stochastic paths can differ. Full eight-rank GPU execution, full training, checkpoint/sample/evaluation memory, and other precision/backends remain unvalidated. The research harness's historical correctness/speedup verdict does not extend automatically to these changes.
