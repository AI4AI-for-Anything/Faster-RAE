# Fresh eight-H20 preset benchmark — 2026-10-15

The optimized preset measured **3.2174×** the preset-off reference: **1,049.22 ms** versus **3,375.80 ms** per effective training iteration, with global batch **512** in both arms. This is a short steady-state timing measurement, not end-to-end 16-epoch training wall time.

## Hardware, source and preparation

- Eight physical NVIDIA H20 GPUs, CUDA-visible capacity approximately 95.00 GiB each, driver `535.183.06`. All selected devices were idle before launch; the existing shared GPU lock was honored. No 80 GiB memory cap was imposed.
- Fresh isolated Python 3.10 virtual environment, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, Triton 3.4.0, transformers 4.56.2 and cuDNN 91002. The earlier temporary environment was missing and was rebuilt. `pip check`, actual-trainer imports, asset checks, CUDA compile/backward and cuDNN/Flash attention checks passed. Complete package versions are in the JSON record.
- Source commit: `6f0df056db8d9164cda0c32804e347e783e9ad16`. Its training runtime is unchanged from `2c61b63`; both timing arms use this same accumulation-capable trainer, including discriminator DDP. The reference is **optimization switches off**, not an untouched upstream trainer.
- Runtime/tools SHA-256: `dbf85452ec2a70b1f1e5aecb50ebc32ebcc88c03695916d989cc64ae3c1d74f1` (sorted `src/**/*.py` and `tools/*.py`, each relative path + NUL + file bytes).
- Independent timing harness SHA-256: `c2dea443a0c9b9db2bbe4bad033dc095979f77d02f16afec74f9b47082abfe0c`. It executes the unchanged real `src/train_stage1.py` entrypoint and uses optimizer post-step hooks.
- Benchmark supervisor SHA-256: `d3543630c8fe89dbaadd33e8fdf9771d45a91c32e9bff0b3378eff90b6d870a6`.
- Existing read-only encoder, discriminator, VGG16 and LPIPS weights were reused offline. The timing fixture contains 7,680 real ImageNet training images extracted locally from the retained public ModelScope shards 0 and 1. No dataset or checkpoint is uploaded with this report.

## Fixed workload and timing protocol

DINOv2-with-registers base encoder and ViT-XL decoder; 256×256 images; `--precision fp32`; **8 GPUs × microbatch 64 × accumulation 1 = global batch 512**. The preset enables regional compilation, TF32 and selected local BF16 operations. The full preset and timing YAML are recorded in [the JSON](h20-readme-benchmark-2026-10-15.json).

The benchmark compresses the three phase starts to epochs 0 / 1 / 2, with 15 effective updates per epoch. Each phase has five warmup updates followed by two windows of five updates. Checkpointing, sample generation and evaluation are disabled. Baseline A, optimized, then baseline B run sequentially with fresh compiler caches; source, data, batch plan and timing configuration are fixed.

Window edges synchronize CUDA. For each window, use the slowest rank's elapsed time divided by five, then average the two windows per phase. An interval between generator optimizer updates includes one complete cycle of generator work, EMA and, when active, discriminator work. All active-D windows contain exactly five discriminator updates. The timer stops at the final generator boundary, after the last measured window, before that boundary's subsequent D/EMA work.

Iter-time is `(6 × reconstruction + 2 × discriminator + 8 × GAN) / 16`, matching the upstream phase proportions. The reference is the average of baseline A and B, and speedup is that reference divided by the optimized result. Startup, compilation, warmup, epoch transitions, checkpointing and evaluation are excluded; the value is not a measured full-training epoch average.

## Results

| Run | Reconstruction + LPIPS (ms) | D-update phase (ms) | GAN phase (ms) | Weighted iter-time (ms) | Max sampled device peak (GiB) |
|---|---:|---:|---:|---:|---:|
| baseline_a | 2,260.80 | 3,467.17 | 4,182.74 | 3,372.56 | 93.78 |
| faster | 689.96 | 1,064.36 | 1,314.88 | 1,049.22 | 52.96 |
| baseline_b | 2,277.76 | 3,469.87 | 4,182.28 | 3,379.03 | 93.78 |

Baseline A/B relative spread: **0.192%**. Every rank in all three runs completed 45 observed generator updates and six timing windows. All 144 rank-windows reported **zero newly compiled graphs**. Reconstruction windows had zero D updates; D/GAN windows had exactly five. No recompile-limit or skipped-CUDA-graph fallback signature was found. Peak memory is sampled whole-device usage; allocator peaks are retained separately in the JSON.

The optimized short run's approximately 52.96 GiB per-device peak is evidence for this eight-H20 workload. It is not a test under an 80 GiB allocator cap and does not validate checkpoint/evaluation memory or all later-training behavior. The separate [H200 capacity report](h200-accumulation-2026-10-15.md) covers capped adaptation probes.

## Quality and revision boundary

The README combines these fresh timing columns with the separately documented [completed-training quality](h20-full-training-quality-2026-10-10.md). Reference quality comes from the public upstream decoder. Optimized quality comes from `fd7800d0633a` trained for 16 epochs, not from this 45-update benchmark. That completed run exactly matches the original imported snapshot, but predates the later accumulation / discriminator-DDP changes. The current trainer's full-training PSNR/rFID remains unvalidated. The benchmark makes no new convergence or numerical-equivalence claim.
