# TPU v6e results (experimental)

The Stage-1 recipe step at global batch 512 ran on a **64-chip TPU v6e slice** in **363.9 ms**, against **4,889 ms** for the upstream code run unchanged under torch_xla. That is **13.4×** for the 6:2:8-weighted step on synthetic images, measured with baseline-candidate-baseline (A/B/A) timing. The statistical gate on short trajectories **did not certify** the adopted configuration. It was adopted with a resolution statement, quoted below. No full training run or downstream evaluation was done on TPU.

The harness and commands are in [`tpu_smoke/`](../tpu_smoke/README.md).

## Hardware and software

- TPU v6e, 64 chips on 16 hosts (4 chips per host), one slice.
- torch 2.9.0 (CPU wheel), torch_xla 2.9.0, libtpu 0.0.21, transformers 4.56.2.
- SPMD with one process per host. The 64 chips form a 1-D `data` mesh: the batch is sharded over it and the parameters are replicated.

## Workload

- DINOv2-with-registers base encoder (frozen, pretrained) and ViT-XL decoder (trainable, randomly initialized). Images are 256×256.
- Global batch **512** = 64 chips × 8 images, no gradient accumulation.
- All three Stage-1 phases of `src/train_stage1.py`, each run as a separate process starting at its first epoch:
  - phase 1, epochs 0–5: reconstruction (L1) + LPIPS;
  - phase 2, epochs 6–7: phase 1 + discriminator update;
  - phase 3, epochs 8–15: phase 2 + adversarial generator loss with adaptive weight.

  The recipe step time is `(6·p1 + 2·p2 + 8·p3) / 16`.
- The upstream RAE, discriminator, LPIPS, `generator_backward`, `update_ema`, optimizer and scheduler are imported unchanged. The DINO ViT-S/8 discriminator backbone, LPIPS and VGG16 use the public pretrained weights.
- Both arms use the precision switches of `scripts/fd7800d0633a.env.sh`: decoder and discriminator attention, MLP and projection in BF16, with `RAE_LOCAL_ATTN_BACKEND=math`. That preset's layout and compile switches are CUDA-only and are off. AdamW uses `RAE_ADAMW_IMPL=foreach` in both arms, because the default `fused` is a CUDA kernel.
- Timing uses a fixed synthetic batch of uniform-noise images that stays on the device. Data loading and host-to-device copies are not measured.

## Timing method

- The same command runs on all 16 hosts. After warmup (8 steps in the first experiment, 6 in the host-side experiments), each arm times 3 windows of 10 steps. Steps are pipelined: the host traces step *n+1* while the device runs step *n*, and the device is drained at the end of each window. An arm's figure is the median window.
- A window that triggered a recompilation is flagged as tainted.
- One additional blocking window per arm (each step starts from an idle device) splits the step into host trace, submit and device time.
- Arms run as baseline, candidate, baseline. A speedup is the mean of the two baselines divided by the candidate. The acceptance threshold is 1.02×, raised when the two baselines drift apart (to 1.035× when they differed by 1.7%).

## Results

### 1. CPU fallbacks and DiffAug translation (2026-11-06)

Running the unchanged code exposed two TPU-specific costs:

- **Two CPU fallbacks per step.** Two bicubic resizes fell back to the CPU each step: the frozen DINOv2 position table (37×37 interpolated to 16×16) and the 256→224 input resize at `src/stage1/rae.py:84`. Each fallback made the device return data mid-step, so the step needed two executions instead of one. `--no-fallbacks` computes the constant table once and moves the image resize to the host. It uses the same functions on the same inputs.
- **956 all-gathers per step.** DiffAug translation (`src/disc/diffaug.py:77`) uses an advanced-index gather that the SPMD partitioner lowered to 956 all-gathers of the whole padded batch per step. `--diffaug-rows` rewrites it as two batch-aligned gathers, and the all-gather count in the HLO drops from 956 to 0.

A is upstream; B is `--no-fallbacks --diffaug-rows`. Pipelined ms/step on 64 chips:

| Phase | A | B | A2 | Baseline drift | Speedup |
|---|---:|---:|---:|---:|---:|
| 1 Reconstruction + LPIPS | 529.78 | **304.99** | 530.05 | 0.05% | **1.74×** |
| 2 + discriminator update | 5,430.07 | **443.56** | 5,435.41 | 0.10% | **12.2×** |
| 3 + adversarial generator | 8,025.89 | **518.20** | 8,020.58 | 0.07% | **15.5×** |
| 6:2:8 weighted | 4,889 | **429** | | | **11.4×** |

Phase 1 has no discriminator, so its gain comes from `--no-fallbacks` alone. No window recompiled.

After these fixes, the step is host-bound (blocking split, ms):

| Phase | Pipelined | Host trace | Submit | Device |
|---|---:|---:|---:|---:|
| 1 | 291.5 | 242.6 | 42.7 | 56.1 |
| 2 | 444.9 | 371.0 | 59.5 | 79.0 |
| 3 | 520.1 | 429.2 | 77.9 | 100.8 |

### 2. Host-side candidates, phase 1 (2026-11-07)

Every arm also has `--no-fallbacks --diffaug-rows`. Runs are listed in order.

| Arm | Pipelined ms | Trace | Device | Speedup | Final loss |
|---|---:|---:|---:|---:|---|
| Baseline | 290.72 | 244.5 | 54.2 | | 0.5826840400695801 |
| `RAE_EMA_FOREACH=1` | 278.12 | 233.3 | 52.6 | 1.054× | 0.5826840400695801 (bitwise) |
| `--dynamo` (decoder forward only) | 253.97 | 204.9 | 20.2 | 1.154× | 0.583113431930542 |
| Baseline (repeat) | 295.75 | 243.7 | 55.8 | | 0.5826840400695801 |

The baselines differed by 1.7%, so the threshold was 1.035×. Both candidates passed. `--dynamo-all` additionally compiles the frozen encoder and LPIPS forwards and keeps the encoder mean/std on the device. In exploratory phase-1 runs without a bracketing baseline, `--dynamo-all` with EMA foreach measured 231.11 ms and 239.66 ms in two runs. Those runs are not used for a speedup claim; the combined stack is measured in the next table.

### 3. Adopted configuration, all phases (2026-11-07)

The adopted configuration is `--no-fallbacks --diffaug-rows --dynamo-all --gc-freeze` with `RAE_EMA_FOREACH=1`. The baseline is the first change set (`--no-fallbacks --diffaug-rows`), measured twice per phase.

| Phase | Baseline (two runs) | Candidate | Candidate windows | Speedup | Candidate trace / submit / device |
|---|---:|---:|---|---:|---|
| 1 | 290.72 / 295.75 | **229.30** | 228.8, 229.3, 229.9 | 1.28× | 175.8 / 37.2 / 19.8 |
| 2 | 445.97 / 442.90 | **371.52** | 370.5, 371.8, 371.5 | 1.20× | 298.0 / 53.5 / 43.0 |
| 3 | 519.08 / 520.76 | **462.97** | 463.8, 462.4, 463.0 | 1.12× | 382.9 / 57.0 / 41.1 |
| 6:2:8 weighted | 425.5 | **363.9** | | **1.17×** | |

Against the upstream arms of Section 1, the recipe step goes from 4,889 to 363.9 ms, **13.4×**. This figure chains two A/B/A experiments. The intermediate configuration measured 429 ms (weighted) in the first experiment and 425.5 ms in the second.

`--gc-freeze` runs `gc.freeze()` after warmup and raises the GC threshold. Without it, one window spent 41.9 ms/step in Python GC, caused by the objects dynamo creates. Values are unchanged.

### Rejected or not pursued

| Item | Result |
|---|---|
| `--adamw-single` (AdamW per-tensor path) | Slower: 284.42 ms against 231.11 ms in the same phase-1 configuration. Rejected. |
| `--compile-opt` (`torch.compile(opt.step)`) | Does not run: LambdaLR requires `step` to be a bound method, and the compiled version is not. |
| Whole-step graph capture (dynamo over forward, backward and optimizer) | Failed on torch_xla 2.9 and on a 2.10 dev build. |
| torch_xla 2.10 dev build (4-chip, global 256 diagnostic) | Eager 336 ms against 321 ms on 2.9. `--dynamo` ran out of device memory at runtime, as on 2.9. Not pursued. |

### Where the time goes now

With the adopted configuration, device time is 20–43 ms per phase and host tracing is 176–383 ms, which is **77–83%** of the step. A step now executes 7–10 graphs; the baseline executes 1. The remaining cost is torch_xla's per-step tracing. Removing it needs the whole step captured in one graph, and both routes to that failed (see above). Under torch_xla 2.9 there is essentially no headroom left within the constraints of [AGENTS.md](../AGENTS.md).

## Numerical tiers

Tiers used here: **N0** means bitwise-identical results. **N1** means the same function and dtypes, with differences from fusion or reduction order.

| Change | Tier | Evidence |
|---|---|---|
| `--no-fallbacks` | N0 | CPU: `torch.equal` to upstream (`tpu_smoke/test_rae_fallbacks.py`). A mutation to bilinear is caught. TPU: phase-1 loss identical across A, B and A2 (0.5896322727203369). On 4 chips, phase 2 with only this flag is bitwise equal to upstream. |
| `--diffaug-rows` | N0 arithmetic; random draws differ | CPU: `torch.equal` to upstream for two policies × 8 seeds, with an off-by-one mutation caught (`tpu_smoke/test_rae_diffaug.py`). On device: equal on 1 chip and on 4-chip SPMD over 3 seeds, max difference 0 (`tpu_smoke/test_rae_diffaug_tpu.py`). In the full step, phase-2/3 losses differ from step 0 (G loss 0.7773 → 0.7772 on 4 chips), and that loss does not pass through the augmentation. The difference therefore comes from XLA RNG draws changing with the compiled graph. AGENTS.md notes that random crops can differ across plans. |
| `RAE_EMA_FOREACH=1` | N0 | Upstream switch. Final loss bitwise equal. |
| `--dynamo`, `--dynamo-all` | N1 | Different fusion and random draws. Phase-1 final loss 0.5826840400695801 → 0.583113431930542 (`--dynamo`). In phase 3 the GAN dynamics amplify the difference: step-0 loss 5.99 → 6.14. |
| `--gc-freeze` | N0 by construction (not measured separately) | Changes Python GC scheduling only; no tensor values are involved. |

## Statistical gate

An N1 change needs a statistical gate. Paired-seed trajectories start from initialization, and baseline and candidate share the seed: it fixes model init, XLA RNG and data. `tpu_smoke/gate_eval.py` takes 5 endpoints. The threshold δ is 2 × the baseline's standard deviation across seeds, frozen before the candidate is examined. The verdict per arm is:

- **PASS:** the mean paired difference is within δ at every endpoint.
- **INCORRECT:** some endpoint exceeds δ with paired t-test p < 0.05.
- **UNRESOLVED:** otherwise.

Each version included a sensitivity control, upstream with the learning rate × 1.02, which must come out INCORRECT for the gate to be trusted. Versions 4 and 5 added a × 1.05 control. All versions ran phase 1 at global batch 512 on 64 chips.

| Version | Data | Steps | LR position | Seeds | Endpoint | δ | Candidate | lr × 1.02 | lr × 1.05 |
|---|---|---:|---|---|---|---|---|---|---|
| 1 | Synthetic | 60 | Epoch 0 warmup (about 2% of peak at step 60) | 0–2 | Training-loss windows | 0.0046–0.0061 | PASS | PASS (should fail) | — |
| 2 | Synthetic | 60 | `--sched-epoch 3`, about 96% of peak | 0–5 (candidate: 5; one run ended early without an error) | Training-loss windows | 0.016–0.026 | PASS | PASS (should fail) | — |
| 3 | Imagenette | 60 | Same as 2 | 0–2 | Training-loss windows | 0.018–0.035 | PASS | PASS (should fail) | — |
| 4 | Imagenette | 120 | Same as 2 | 0–2 | Training-loss windows | 0.013–0.026 | PASS | PASS (should fail) | PASS (should fail) |
| 5 | Imagenette | 120 | Same as 2 | 0–2 | Fixed held-out eval loss | 0.012–0.054 | PASS | PASS (should fail) | UNRESOLVED |

Imagenette is the public fast.ai ImageNet subset (`imagenette2-320.tgz`). The pool is the first 4,096 training images in sorted path order, bicubic-resized to 384 on the short side and center-cropped to 384×384, and it was byte-identical on all 16 hosts. Each step draws 512 images and random 256 crops from (seed, step). The version-5 endpoint is the reconstruction loss (L1 + LPIPS) in eval mode, which has no `noise_tau` noise. It is computed on the pool's last 512 images, center-cropped, after steps 32, 54, 76, 98 and 120.

Version 5, mean paired differences per endpoint:

| Arm | Verdict | e0 | e1 | e2 | e3 | e4 |
|---|---|---:|---:|---:|---:|---:|
| δ | | 0.012 | 0.013 | 0.054 | 0.019 | 0.036 |
| Candidate | PASS | +0.0026 | −0.0051 | +0.0209 | +0.0061 | +0.0172 |
| lr × 1.02 | PASS | +0.0106 | +0.0113 | −0.0044 | +0.0085 | −0.0018 |
| lr × 1.05 | UNRESOLVED (e0, e1 exceed δ; p = 0.22, 0.55) | +0.0213 | +0.0133 | +0.0014 | +0.0009 | +0.0058 |

**The gate could not certify the candidate.** Neither sensitivity control was resolved, even at learning rate × 1.05, so a PASS for the candidate carries no weight. The cause is the dynamics, not the candidate:

- **Divergence near peak LR.** With the schedule near its peak, any small perturbation makes paired trajectories diverge within tens of steps. Version 1 avoided this only by sitting in the warmup, where the learning rate was too small for differences to show. In version 3, paired p-values stayed above 0.5, so pairing by seed did not cancel the noise. In version 4, the three non-baseline arms had paired differences of the same magnitude with no consistent sign. On the fixed held-out set, one trajectory's loss moved by 0.03–0.05 between adjacent endpoints (upstream seed 0: 0.9445 → 0.9043). The oscillation therefore comes from the parameters themselves, not from data sampling.
- **Eval-forward insertion.** Version 5 reran version 4's arms with the periodic eval forward inserted. The first training-loss window (steps 10–32), before any eval, reproduced version 4 bitwise (−0.00582, +0.00014, +0.01248 for candidate, × 1.02 and × 1.05). After the first eval at step 32, every arm's later endpoints differed from version 4; for example, the × 1.02 arm's e1 moved from +0.0051 to +0.0095. A no-grad forward does not change the training arithmetic, only the compiled graph, yet it moved the trajectory by 0.005–0.01. That is the same order as the effect of learning rate × 1.05. A short-trajectory gate in this setting cannot distinguish a bitwise-level change from a 5% learning-rate change.

Making the gate resolve a 5% learning-rate change would need hundreds of steps or more than ten seeds per arm. That was not run.

**Adoption.** On 2026-11-07 the project owner adopted the candidate with the following resolution statement:

> The candidate's two fixes keep upstream's arithmetic (N0); only the way random samples are drawn differs, which AGENTS.md allows. The dynamo compilation is N1: same dtypes, different fusion, not bitwise equal to eager execution. With real images, 120 steps and 3 seeds, the candidate and upstream show no measurable difference on either endpoint, training loss or fixed held-out evaluation loss. This gate cannot resolve learning-rate perturbations of 5% or less, so the conclusion only rules out deviations larger than that magnitude. Formal acceptance rests on downstream evaluation of a real long training run.

Formal acceptance, meaning a real long training run on TPU followed by downstream evaluation like the one in [H20 results](h20-results.md), has not been done.

## Batch fairness and comparability

The baseline and candidate use the same global batch of 512 on the same 64 chips (8 images per chip), with the same data-parallel layout, model, resolution, losses, learning-rate schedule and precision switches. The 13.4× ratio is therefore like-for-like on TPU.

It is **not** comparable with [H20 results](h20-results.md), which used 8 H20 GPUs × 64 images per GPU, with CUDA and the optimization preset switched off and on. The TPU baseline is the upstream code running unchanged under torch_xla. Most of the TPU gain removes costs that only exist there: two CPU fallbacks, 956 all-gathers per step, and per-step host tracing. This document makes no TPU-versus-GPU speed claim.

## Limitations

- **Timing on synthetic data.** The batch is fixed, uniform noise, and resident on the device. The decoder is randomly initialized. Data loading, checkpointing, sampling and evaluation are not covered.
- **The gate covers phase 1 only.** Phases 2 and 3 (discriminator and adversarial training) were not gated. In phase 3, dynamo changed the step-0 loss from 5.99 to 6.14.
- **No certification.** The gate could not resolve learning rate × 1.05. Adoption rests on the resolution statement. No long training or downstream evaluation was run on TPU.
- **Adaptive GAN weight.** Upstream computes it per rank from 64 images. One SPMD program has no ranks, so the harness computes it once from the whole batch's last-layer gradients. Per-chip work matches (two `autograd.grad` calls), so timing is representative, but numerics are not bitwise equal to an 8-GPU run. Both arms are affected equally.
- **Accumulation in phase 3 is not supported.** `generator_backward` replays CPU/CUDA RNG states but not the XLA RNG, so the harness raises an error instead of running with mismatched noise.
- **SPMD random draws.** In a different model on this torch_xla version, random tensors on a multi-dimensional mesh were affected by sharding propagation. A probe on 1-D and 2-D meshes did not reproduce that problem. Whether RAE's random draws (DiffAug, encoder noise) on its 1-D mesh are affected is unverified.
- **Image pool.** The 384×384 center crop drops a strip of the long side that upstream's `RandomCrop` could reach.
- **One software stack.** All results are on torch_xla 2.9.0 with libtpu 0.0.21.
