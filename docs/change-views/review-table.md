# What changed in Faster-RAE

Three views of the same pinned Git evidence: **[review table](review-table.md)** · [module map](module-map.svg) · [browsable cards](index.html). The cards are a standalone HTML file: download or open them locally; GitHub displays the source.

Current code, presets, tests and documentation from two historical Git ranges. Retired reports, the source manifest and this presentation are excluded.

**Attribution:** No per-file causal speedups are claimed. An optimization setting can span multiple files; measurements apply to the benchmarked configuration as a whole.

**Selected vs available:** Optional encoder-linear BF16, VGG-convolution BF16, foreach EMA and foreach AdamW hooks exist in source but are not selected by the imported preset.

**Adaptation:** Preserving effective global batch preserves sample/update cadence, not exact numerical equivalence. A single-rank capacity PASS does not validate full distributed convergence.

## Initial optimization snapshot

**15 changed paths · +400 / −38 lines.** Line counts describe the diff, not runtime benefit.

| Module | File | Change | Purpose |
|---|---|---:|---|
| Preset | [.harness/config.json](../../.harness/config.json) | A +23 / −0 | Records the selected optimization switches as an environment mapping; this small preset is not the independent research harness. |
| Documentation | [README.md](../../README.md) | M +33 / −0 | Documents the optimized snapshot and usage; the later phase adds measured benchmarks, reconstruction quality and adaptation instructions. |
| Preset | [scripts/fd7800d0633a.env.sh](../../scripts/fd7800d0633a.env.sh) | A +21 / −0 | Shell version of the selected preset: TF32/cuBLAS, block compilation, LPIPS layout and local BF16 paths. |
| Loss & discriminator | [src/disc/dinodisc.py](../../src/disc/dinodisc.py) | M +23 / −5 | Adds local BF16 attention, projection and MLP paths; compiles discriminator transformer blocks and the normalization tensor region. |
| Loss & discriminator | [src/disc/lpips.py](../../src/disc/lpips.py) | M +33 / −6 | Adds channels-last VGG/comparison layouts, a compiled LPIPS comparison region and optional convolution hooks. |
| Decoder | [src/stage1/decoders/decoder.py](../../src/stage1/decoders/decoder.py) | M +34 / −21 | Routes attention through local BF16 SDPA and projection/MLP through scoped BF16 linear helpers; compiles decoder blocks; fixes the SDPA dropout attribute. |
| Compute controls | [src/stage1/encoders/dinov2.py](../../src/stage1/encoders/dinov2.py) | M +10 / −2 | Installs opt-in EncoderLinear hooks and an attention-backend context. Encoder-linear BF16 is OFF in the selected preset. |
| Compute controls | [src/stage1/rae.py](../../src/stage1/rae.py) | M +3 / −1 | Applies the requested process-wide matrix multiplication settings when constructing RAE. |
| Compute controls | [src/utils/compute_backends.py](../../src/utils/compute_backends.py) | A +39 / −0 | Controls TF32, BLAS and SDPA selection; offers optional nondeterministic cuDNN autotuning. |
| Compute controls | [src/utils/fusion_backends.py](../../src/utils/fusion_backends.py) | A +80 / −0 | Centralizes opt-in torch.compile regions, ATen GEMM selection, repeated-backward buffer compatibility and call diagnostics. |
| Decoder | [src/utils/local_attention.py](../../src/utils/local_attention.py) | A +19 / −0 | Casts Q/K/V operands to BF16 for a selected native SDPA backend, then restores the incoming output dtype. |
| Compute controls | [src/utils/local_auxiliary.py](../../src/utils/local_auxiliary.py) | A +27 / −0 | Implements scoped BF16 discriminator linear operations plus optional encoder-linear and VGG-convolution hooks. The latter two are OFF in the selected preset. |
| Decoder | [src/utils/local_linear.py](../../src/utils/local_linear.py) | A +18 / −0 | Casts decoder linear operands locally while keeping original parameter ownership and gradients; restores the activation dtype. |
| Compute controls | [src/utils/optim_utils.py](../../src/utils/optim_utils.py) | M +20 / −2 | Adds an optional foreach AdamW route and resume-state handling. The selected preset retains upstream fused AdamW. |
| Compute controls | [src/utils/train_utils.py](../../src/utils/train_utils.py) | M +17 / −1 | Adds optional grouped foreach EMA updates and diagnostics. Foreach EMA is OFF in the selected preset. |

## Later GPU adaptation

**17 changed paths · +1088 / −84 lines.** Line counts describe the diff, not runtime benefit.

| Module | File | Change | Purpose |
|---|---|---:|---|
| Documentation | [.gitignore](../../.gitignore) | M +5 / −1 | Excludes local adaptation environments and generated run artifacts. |
| Environment & workflow | [AGENTS.md](../../AGENTS.md) | A +15 / −0 | Defines the agent workflow: default 8 GPUs, fixed upstream global batch, bounded memory probing and release boundaries. |
| Documentation | [README.md](../../README.md) | M +25 / −3 | Documents the optimized snapshot and usage; the later phase adds measured benchmarks, reconstruction quality and adaptation instructions. |
| Environment & workflow | [docs/agent-environment.md](../../docs/agent-environment.md) | A +76 / −0 | Documents isolated environment setup, dependency/toolchain checks, local assets and troubleshooting before GPU probing. |
| Environment & workflow | [docs/gpu-adaptation.md](../../docs/gpu-adaptation.md) | A +73 / −0 | Explains the fixed-global-batch plan, three probe phases, outputs, launch commands and limits of single-rank capacity evidence. |
| Documentation | [docs/release-boundary.md](../../docs/release-boundary.md) | A +9 / −0 | Separates the repository adaptation entrypoint from the independently versioned research harness and its experiment ledger. |
| Environment & workflow | [requirements-adapt.txt](../../requirements-adapt.txt) | A +7 / −0 | Lists adaptation dependencies alongside the repository environment requirements. |
| Environment & workflow | [skills/adapt-rae-gpu/SKILL.md](../../skills/adapt-rae-gpu/SKILL.md) | A +18 / −0 | Gives agents a first-run GPU adaptation checklist while preserving model, loss, precision and global batch. |
| Environment & workflow | [src/disc/lpips_utils.py](../../src/disc/lpips_utils.py) | M +5 / −0 | Accepts an explicit local LPIPS checkpoint path and verifies its expected checksum before use. |
| Trainer & accumulation | [src/train_stage1.py](../../src/train_stage1.py) | M +100 / −80 | Adds bounded probes; splits an effective CPU batch into GPU microbatches; accumulates G/D gradients with DDP synchronization; advances G/EMA/schedules per effective update. |
| Trainer & accumulation | [src/utils/accumulation.py](../../src/utils/accumulation.py) | A +66 / −0 | Implements mean-gradient accumulation, final-microbatch DDP synchronization and RNG replay for rank-local GAN adaptive weighting. |
| Trainer & accumulation | [src/utils/batch_plan.py](../../src/utils/batch_plan.py) | A +39 / −0 | Enforces global batch = world size × microbatch × accumulation and rejects invalid divisors or mismatched deployment sizes. |
| Capacity validation | [src/utils/capacity_probe.py](../../src/utils/capacity_probe.py) | A +107 / −0 | Bounds the real trainer run; applies allocator limits; records warmup-adjusted timing, memory and finite/update checks. |
| Capacity validation | [tests/test_accumulation.py](../../tests/test_accumulation.py) | A +123 / −0 | Checks mean-gradient/adaptive-weight behavior on toy models and distributed synchronization; does not establish full-run convergence. |
| Capacity validation | [tests/test_gpu_adaptation.py](../../tests/test_gpu_adaptation.py) | A +53 / −0 | Checks batch arithmetic and configuration constraints for GPU adaptation. |
| Capacity validation | [tools/adapt_gpu.py](../../tools/adapt_gpu.py) | A +254 / −0 | Runs environment preflight and three real-trainer phases; lowers microbatch only on OOM; emits a fixed-global-batch training config and launch script. |
| Environment & workflow | [tools/check_environment.py](../../tools/check_environment.py) | A +113 / −0 | Checks actual trainer imports, local assets, compiler forward/backward and available SDPA backends before probing. |

## Reproduce and review

```bash
python3 docs/change-views/generate.py
```

The generator requires both pinned commits in local Git history and uses Python’s standard library. `changes.json` stores per-phase paths, status, added/deleted lines, commit IDs and Git blob IDs. The curated descriptions live in `generate.py`; regenerating the views never runs GPU work.

A = added; M = modified. A file can appear in both phases (for example README). Runtime comparisons belong to the [H20 results](../h20-results.md), not this line-count inventory.
