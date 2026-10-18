---
name: adapt-rae-gpu
description: Prepare Faster-RAE and fit a per-GPU VRAM budget while preserving upstream global batch through microbatching and gradient accumulation. Use for first-run setup, 80 GiB deployment, or OOM adaptation; not full training or numerical-equivalence research.
---

# Adapt Faster-RAE to a GPU budget

Read [environment preparation](../../docs/agent-environment.md) first, then [the adaptation guide](../../docs/gpu-adaptation.md).

Determine the intended total GPU count; ask when useful, but **default to 8 if unspecified**. Use `--world-size` to record it. Preserve the source config's global batch (512 in the default config). Compute `accumulation = global_batch / (GPU_count × microbatch)` with exact integer divisibility. Never choose a smaller global batch to solve OOM. Default to 80 GiB per card with 4 GiB allocator headroom. Record actual hardware, package versions and preset; a capped H200 is not an A100/H100 test.

Run preflight and `tools/adapt_gpu.py` in a fresh directory on one reserved idle GPU, using existing assets. It simulates one target rank, including its entire accumulation cycle. Default microbatch candidates are divisors of the per-rank effective batch, descending from 64. Only OOM warrants automatic microbatch reduction and increased accumulation. Diagnose other failures before retrying in a fresh directory. Try `--attention-backend flash` only for actual cuDNN support failures, report the numerical-path change, and rerun all phases.

Inspect `summary.json`, `environment.json`, and each phase's `metrics.json`. Require all three phases, full effective steps, expected optimizer update counts, finite losses/gradients/parameters, and peaks within budget. Report both device and allocator peaks. Compiles after warmup and profiler timings are diagnostic.

Deliver **global batch, target GPU count, microbatch, accumulation**, peaks, preset, `training.yaml`, `launch.sh`, and limitations. The training config restores the upstream schedule and checks the actual world size. Single-rank probing does not validate target-size DDP. Spectral-normalization update counts and stochastic crop paths remain microbatch-dependent (default BatchNormLocal has per-sample groups and no running statistics); do not claim accumulation gives exact numerical equivalence. GAN accumulation uses an extra forward/last-layer-gradient pass with RNG replay to calculate one rank-local adaptive weight.

If no candidate fits, report the failure. Do not silently change architecture, resolution, loss, or precision. A full research optimization loop belongs in the separately released harness.
