# GPU memory adaptation with a fixed global batch

Start with [environment preparation](agent-environment.md). This is a bounded operational smoke test of the actual Stage-1 trainer. It is not the independent optimization harness or a convergence test.

## Batch planning

Keep the upstream global batch fixed: **512** for the default configuration. Ask for the intended total GPU count when needed; otherwise use **8 GPUs**. `--world-size` is the target deployment size, while `--gpu` selects one idle device for capacity testing.

`global batch = GPUs × microbatch per GPU × gradient accumulation steps`

| Target GPUs | Microbatch | Accumulation | Global batch |
|---|---:|---:|---:|
| 8 (default) | 64 | 1 | 512 |
| 8 | 32 | 2 | 512 |
| 8 | 16 | 4 | 512 |
| 4 | 32 | 4 | 512 |
| 1 | 32 | 16 | 512 |

Only exact divisors are accepted. The default ladder tries divisors from 64 down to 1. Explicit invalid candidates fail before GPU work. Reducing microbatch never changes global batch or learning rate. The probe simulates one target rank: for 8 GPUs, every G update consumes 64 images on the tested device regardless of microbatch. Its output separately records the executed single-rank batch and planned global batch; this is not an executed 512-image distributed update.

If memory is insufficient, first reduce the microbatch and increase gradient accumulation while keeping the global batch fixed. If this still does not fit, consider gradient checkpointing and tune the number of checkpointed decoder layers. **Agent hint:** check implementation and configuration support first, then validate correctness, peak memory, and step time across all three training phases.

## First run: 80 GiB

```bash
python tools/adapt_gpu.py --gpu 0 --budget-gib 80 --reserve-gib 4 \
  --world-size 8 --batches 64 32 16 8 --steps 8 --warmup 3 \
  --data-path /path/to/imagefolder \
  --disc-checkpoint /path/to/dino_vit_small_patch8_224.pth \
  --out runs/adapt-80g
```

`--gpu` identifies a physical GPU index or UUID. The launcher exposes just that card to a one-rank `torchrun`. Use your cluster's scheduler/reservation rules; on the test host, runs also used its existing cooperative GPU lock. The launcher refuses a card with more than 1 GiB already occupied and never kills unrelated processes. Another job starting on the same card invalidates the test; reserve the GPU rather than relying only on the initial check.

For each microbatch and its accumulation count, three fresh processes run reconstruction+LPIPS, discriminator updates, and full adversarial-generator updates. Each probe step is one effective G update (all accumulated microbatches), followed by the configured number of D updates and one EMA update in the real training loop. The generator GAN case includes adaptive-weight autograd and the fresh reconstruction after the G update. A microbatch/accumulation plan is selected only when all three phases finish with finite losses and final decoder gradients/parameters, plus discriminator gradients/parameters in phases that update it. Reconstruction has no discriminator gradients by design. Compile/setup memory is included in the peak. Non-OOM failures stop the search; OOM moves to the next microbatch and increases accumulation to keep global batch fixed. Each phase has a 20-minute default timeout, including compilation.

The phases temporarily start at step zero for capacity coverage. Probe output disables checkpoint/EMA-image/FID generation to bound time and storage. Consequently, their memory and runtime are not validated. The generated `training.yaml` restores the source training schedule, checkpointing and sampling settings; perform a separate output/evaluation smoke check before a full training job. Short probes can underestimate later allocator/shape behavior.

## What the budget means

Units are GiB (2^30 bytes). The effective budget is the smaller of the requested value and the GPU's reported total memory. The default 80 GiB request limits the PyTorch allocator to 76 GiB and leaves 4 GiB for CUDA context, libraries and other allocations. [PyTorch's memory fraction API](https://docs.pytorch.org/docs/2.8/generated/torch.cuda.memory.set_per_process_memory_fraction.html) limits its caching allocator, not all device allocations. The launcher additionally samples whole-device memory every approximately 0.5 seconds and stops its child process group if the sampled total exceeds the budget. Brief inter-sample excursions are possible; this is not hardware partitioning or a hard system-wide VRAM cap.

An H200 with that limit remains an H200. The procedure does not emulate an A100's SM architecture, bandwidth, kernel support, or compiler choices. Run it on each actual target GPU. The default preset forces cuDNN SDPA; if unsupported, explicitly try `--attention-backend flash` in a new output directory. That changes the numerical path and requires its own correctness validation for scientific comparisons.

## Outputs and normal training

- `environment.json`: actual import, toolchain, hardware and asset checks.
- `summary.json`: all attempts, OOMs, selected batch and measured peaks.
- `b<batch>-<phase>/metrics.json`: step completion, loss scalars, final finite checks, compilation count, allocator peaks.
- `b<batch>-<phase>/run.log`: full diagnostics, retained locally.
- `training.yaml`: unchanged upstream global batch and schedule, plus microbatch, accumulation, and expected world size.
- `launch.sh`: single-node launch example for the selected target GPU count (not automatically executed).
- `preset.env.sh` / `environment-preset.json`: exact optimization switches.

After reviewing the result, a normal training launch is:

```bash
source runs/adapt-80g/preset.env.sh
export EXPERIMENT_NAME=my-eight-gpu-run
torchrun --standalone --nproc_per_node=8 src/train_stage1.py \
  --config runs/adapt-80g/training.yaml \
  --data-path /path/to/imagefolder --results-dir results --precision fp32
```

This launches full training and is intentionally not done by the adaptation tool. Keep the asset/cache environment from preflight. Match the target GPU count; a mismatch fails before model initialization. For multiple nodes, use the same total world size and configure torchrun rendezvous/node arguments instead of the single-node example. Re-run adaptation for a changed GPU count.

## Accumulation semantics

The DataLoader supplies the effective per-rank batch on CPU and drops incomplete tails exactly as a non-accumulated local batch would. Only one microbatch at a time is transferred to CUDA. Each mean loss is divided by the accumulation count. G gradients synchronize on the final microbatch, followed by one optimizer/clip/scheduler/EMA update. D then regenerates reconstructions with the updated G and accumulates over the same CPU images; each configured D update has its own optimizer/scheduler step. D uses its DDP wrapper for training so its gradients synchronize too. [PyTorch requires both forward and backward inside `no_sync`](https://docs.pytorch.org/docs/2.8/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync).

For GAN accumulation, the trainer first sums reconstruction and GAN last-layer gradients across the rank's microbatches, computes one adaptive weight from their mean gradients, and then replays the microbatches for weighted backward. Torch CPU/CUDA RNG states are restored so replay uses the same augmentation/noise draws. This costs an extra forward and last-layer gradient pass but avoids retaining all activation graphs. The weight remains rank-local, matching upstream's scope. D parameters are frozen only during G backward, while gradients through D to G remain enabled.

The epoch length, GAN phase starts, logging, sampling, and G scheduler are measured in effective updates. LR and EMA decay stay unchanged. This preserves sample-count and update cadence, **not exact numerical equivalence**: D spectral-normalization buffers update for every microbatch forward, and random-window crops, floating-point reductions and stochastic kernel paths can differ. The default `BatchNormLocal(virtual_bs=1)` normalizes each sample separately with no running statistics; it does not have standard BatchNorm running-state behavior. Different normalization configurations need their own assessment. Accumulation tests establish mean-gradient behavior for batch-independent toy models and DDP synchronization; convergence and full target-size GPU DDP still require separate validation. This workflow does not change image size, architecture, loss coefficients, or whole-step precision.

For another capacity, use a fresh output path, e.g. `--budget-gib 48 --out runs/adapt-48g`. For a diagnostic PyTorch trace, repeat one chosen batch with `--profile` and another output path. This writes a trace and operator table for the first post-warmup step of each phase. Do not compare its reported timing with unprofiled runs or the historical 5.32x claim.
