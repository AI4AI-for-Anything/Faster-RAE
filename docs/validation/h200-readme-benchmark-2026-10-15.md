# Fresh Stage-1 preset benchmark — 2026-10-15

The optimized preset measured **5.41×** faster than the same trainer with optimization switches disabled: **1547.90 ms** versus **8381.01 ms** per weighted effective iteration. [Aggregate records](h200-readme-benchmark-2026-10-15.json) include both ranks' timing windows and optimizer counts.

## Version and comparison

- Code: `2c61b63ba941a81d86393d20a7e3d01de482561d`; runtime/tool source fingerprint `dbf85452ec2a70b1f1e5aecb50ebc32ebcc88c03695916d989cc64ae3c1d74f1`. The measured checkout stayed frozen throughout all runs.
- Both arms use that commit's actual `src/train_stage1.py`, including its fixed-global-batch accumulation and G/D gradient synchronization. The baseline clears acceleration switches. The faster arm loads the exact `.harness/config.json` environment preset. This comparison isolates the preset within the current trainer; it is distinct from the historical reference benchmark of `fd7800d0633a`.
- Hardware: 2 NVIDIA L20X GPUs on the host called H200, with physical SM90 devices; driver `570.148.08`. No physical H20 was measured. The GPUs have roughly 140 GiB each; this benchmark did not impose the adaptation workflow's 80 GiB cap.
- Stack: Python 3.10.22, torch 2.8.0+cu128, torchvision 0.23.0+cu128, Triton 3.4.0, transformers 4.56.2. Whole-step precision is fp32; the faster preset enables its documented local BF16/TF32 paths.
- Resolution 256×256, global batch **512**, actual world size **2**, microbatch **64**, accumulation **4**. Each effective update consumes 512 images across the two physical ranks. These are measured iteration times for this configuration, not extrapolated eight-GPU numbers.

## Timing protocol

Order: baseline A → faster → baseline B. Each run starts in a new process with separate cold compiler caches and the same prepared local ImageNet fixture and weights. GPU 0/1 were checked idle and reserved with the existing cooperative lock; unrelated jobs were left running on other GPUs.

The input fixture contains 7,680 images, giving 15 effective updates per epoch at global batch 512. The temporary timing configuration compresses the phases into 3 epochs: reconstruction+LPIPS, D updates, then adversarial G+D. It disables checkpoint/sample/evaluation output, retains optimizer/LR settings, and uses 4 loader workers per rank. Each phase has 5 warmup updates followed by two windows of 5 updates.

The independent harness's optimizer-hook timer is used unchanged, SHA-256 `c2dea443a0c9b9db2bbe4bad033dc095979f77d02f16afec74f9b47082abfe0c`. It synchronizes CUDA only at window edges. A window spans consecutive G optimizer post-step boundaries, so steady-state intervals include the preceding EMA/D work and the next G update. It runs the real entrypoint without modifying training code. This is a timing run, not the harness's numerical-equivalence gate.

For each window, use the slower rank's elapsed time divided by its 5 effective steps; then average the two windows per phase. Iter-time is `(6 × recon + 2 × disc + 8 × GAN) / 16`, reflecting the upstream 16-epoch phase proportions. Baseline iter-time averages baseline A and B; speedup is that mean divided by the faster run. Startup, cold compilation, warmup, epoch boundaries, checkpoints and evaluation are outside the reported timing windows.

## Results

| Run | Recon ms | Disc ms | GAN ms | Weighted ms/iter | Sampled device peak GiB |
|---|---:|---:|---:|---:|---:|
| baseline_a | 4770.16 | 7401.87 | 11332.06 | 8380.07 | 93.74 |
| faster | 814.79 | 1316.04 | 2155.70 | 1547.90 | 52.91 |
| baseline_b | 4774.05 | 7402.85 | 11332.65 | 8381.95 | 93.74 |

Baseline A/B relative spread: **0.02%**. Both ranks in all three runs reported 45 observed G updates and all six timing windows. Every timed window contained zero newly compiled graphs, zero D optimizer steps in reconstruction, and exactly five D optimizer steps in the active D/GAN phases. No checkpoint or full training run was produced by this timing job.

## PSNR and rFID

No completed-training checkpoint with matching provenance for commit `2c61b63` was available for this benchmark, so its quality results remain **Not evaluated**. The H200 evidence available at measurement time only validated short trajectories of `fd7800d0633a`; older eight-GPU records on that host did not establish completed training for this revision.

Follow-up: the user subsequently supplied a separate H20 host containing a completed 16-epoch run and 50k evaluation of the original `fd7800d0633a` snapshot. Its [quality report](h20-full-training-quality-2026-10-10.md) documents the exact source match and revision boundary. It does not supply completed-training validation for the later `2c61b63` trainer changes.

To populate these columns, evaluate the baseline and optimized final checkpoints with the same validation split, resolution/preprocessing, model-versus-EMA selection and rFID reference statistics. Record checkpoint provenance and completed training schedule alongside the result. This timing result does not establish full-training quality equivalence.
