# tpu_smoke: RAE Stage-1 on TPU v6e (torch_xla SPMD)

Experimental. A timing and gate harness that runs the full Stage-1 step from `src/train_stage1.py` on a TPU slice with torch_xla SPMD. It imports the upstream modules unchanged; the TPU changes are applied from outside at runtime and switched on by flags. Results and their conditions are in [docs/tpu-v6e-results.md](../docs/tpu-v6e-results.md).

It is not a training script. There is no data loader, checkpointing, sampling or evaluation suite. Timing uses a fixed synthetic batch, and the statistical-gate mode uses short trajectories.

## Files

| File | Purpose |
|---|---|
| `rae_recipe_step.py` | Entry point. One process per worker, one phase per run (`--phase 1/2/3` = epochs 0, 6 and 8 of the recipe). Prints `SUMMARY {json}` in timing mode or `TRAJ {json}` in gate mode (`--traj N`). |
| `rae_fallbacks.py` | `--no-fallbacks`: caches the frozen DINOv2 position-encoding table and moves the 256→224 bicubic resize to the host. Together these remove two per-step CPU fallbacks. Bitwise equal to upstream. |
| `rae_diffaug.py` | `--diffaug-rows`: rewrites DiffAug's translation gather (`src/disc/diffaug.py:77`) as two batch-aligned `torch.gather` calls. Bitwise equal to upstream, and it removes 956 whole-batch all-gathers per step. |
| `step_counters.py` | Per-step torch_xla counter deltas (CPU fallbacks, executions, transfers) and Python GC time for each timing window. |
| `build_image_pool.py` | Builds the fixed uint8 Imagenette pool (`[4096, 384, 384, 3]`) used by `--data-pool` in gate mode. |
| `gate_eval.py` | Statistical gate over paired-seed trajectories. Prints PASS, INCORRECT or UNRESOLVED per arm. |
| `test_rae_diffaug.py` | CPU test: `--diffaug-rows` is `torch.equal` to upstream DiffAug for two policies × 8 seeds, and an off-by-one mutation is caught. |
| `test_rae_fallbacks.py` | CPU test: the cached position table and host resize are `torch.equal` to upstream, and the cache refuses other input sizes. Needs the DINOv2-with-registers base weights in the HF cache. |
| `test_rae_diffaug_tpu.py` | TPU test: same comparison on device, unsharded and with `--spmd`. Imports torch_xla. |
| `test_gate_eval.py` | CPU test of `gate_eval.py` on synthetic trajectories: a near-copy gives PASS and a +0.05 shift gives INCORRECT, for both endpoints. |

## CPU tests

Run these from the repository root. A CPU PyTorch environment with the repository requirements is enough; torch_xla is not needed.

```bash
python -m tpu_smoke.test_rae_diffaug      # prints bitwise=True ... mutation_caught=True, PASS
python -m tpu_smoke.test_rae_fallbacks    # needs facebook/dinov2-with-registers-base in HF_HOME
python -m tpu_smoke.test_gate_eval        # no torch needed
```

The test files are scripts with a `main()`. The directory has no `__init__.py`, so `python -m unittest discover` does not collect them. Pytest would, and `test_rae_diffaug_tpu.py` fails to import without torch_xla.

## Environment

The measured stack was Python 3.11, `torch==2.9.0` (CPU wheel), `torch-xla==2.9.0`, `libtpu==0.0.21` and `transformers==4.56.2`, with `omegaconf==2.3.0`, `einops` and `torchvision==0.24.0` (CPU wheel). Install it on every worker:

```bash
python3.11 -m venv <VENV> && . <VENV>/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0 torchvision==0.24.0
pip install torch-xla==2.9.0 libtpu==0.0.21 transformers==4.56.2 omegaconf==2.3.0 einops
```

The assets are the same public files listed in [docs/agent-environment.md](../docs/agent-environment.md). Put them on every worker:

```bash
mkdir -p <ASSETS>/torch/hub/checkpoints
curl -L -o <ASSETS>/vgg.pth "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"   # MD5 d507d7349b931f0638a25a48a722f98a
curl -L -o <ASSETS>/torch/hub/checkpoints/vgg16-397923af.pth https://download.pytorch.org/models/vgg16-397923af.pth
curl -L -o <ASSETS>/dino_vit_small_patch8_224.pth https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth
HF_HOME=<HF_CACHE> huggingface-cli download facebook/dinov2-with-registers-base
```

## Launch

Every worker of the slice runs the same command at the same time from the repository root, for example through `gcloud compute tpus tpu-vm ssh <TPU_NAME> --zone=<ZONE> --worker=all --command='...'`. The 64 chips form a 1-D `data` mesh. The batch is sharded over it and the parameters are replicated. At global batch 512 that is 8 images per chip. Every worker prints the same summary, so read worker 0's.

```bash
export PJRT_DEVICE=TPU ALLOW_MULTIPLE_LIBTPU_LOAD=1
export HF_HOME=<HF_CACHE> TORCH_HOME=<ASSETS>/torch RAE_LPIPS_CHECKPOINT=<ASSETS>/vgg.pth
COMMON="--batch 512 --recipe-bf16 --disc-ckpt <ASSETS>/dino_vit_small_patch8_224.pth --window 10 --windows 3"
```

`--recipe-bf16` turns on the precision switches of `scripts/fd7800d0633a.env.sh`, which are the decoder and discriminator attention/MLP/projection BF16 settings, with `RAE_LOCAL_ATTN_BACKEND=math`. Both arms use it. Both arms also use `RAE_ADAMW_IMPL=foreach`, the script's default: upstream's default `fused` is a CUDA kernel and rejects XLA tensors.

Timing, for each `P` in 1 2 3:

```bash
# Baseline: upstream code as is
TORCHDYNAMO_DISABLE=1 python -m tpu_smoke.rae_recipe_step --phase P $COMMON --warmup 8
# First change set (both bitwise fixes)
TORCHDYNAMO_DISABLE=1 python -m tpu_smoke.rae_recipe_step --phase P $COMMON --warmup 8 --no-fallbacks --diffaug-rows
# Adopted configuration
TORCHDYNAMO_DISABLE=0 RAE_EMA_FOREACH=1 python -m tpu_smoke.rae_recipe_step --phase P $COMMON --warmup 6 \
  --no-fallbacks --diffaug-rows --dynamo-all --gc-freeze
```

The first A/B/A (upstream against the first change set) used the default `--warmup 8`. The host-side runs used `--warmup 6`. Run the arms in A, B, A order and report the recipe step as `(6*p1 + 2*p2 + 8*p3) / 16`. In `SUMMARY`, `pipelined_ms` is the median of the windows. `tainted` flags a window that recompiled. `blocking` gives the trace/submit/device split from one extra blocking window. `last_total_dloss` holds the last losses for the bitwise checks.

`--adamw-single` and `--compile-opt` were measured and rejected. `--dynamo`, which compiles the decoder only, passed but is superseded by `--dynamo-all`. `--host-profile` prints a cProfile of 3 steps.

## Statistical gate (phase 1)

Build the image pool on every worker. It must be byte-identical across workers, so compare `sha256sum` on each.

```bash
curl -L -o <DATA>/imagenette2-320.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar xzf <DATA>/imagenette2-320.tgz -C <DATA>
python -m tpu_smoke.build_image_pool <DATA>/imagenette2-320/train <DATA>/pool384.npy --n 4096
```

The gate has four arms on paired seeds. `up` is upstream. `cand` is the adopted configuration. `sens` and `sens5` are upstream with the learning rate × 1.02 and × 1.05, and they are sensitivity controls that must come out INCORRECT for the gate to count. Run on every worker:

```bash
mkdir -p logs
for seed in 0 1 2; do for arm in up cand sens sens5; do
  case $arm in
    up)    envs="TORCHDYNAMO_DISABLE=1"; x="";;
    cand)  envs="TORCHDYNAMO_DISABLE=0 RAE_EMA_FOREACH=1"; x="--no-fallbacks --diffaug-rows --dynamo-all --gc-freeze";;
    sens)  envs="TORCHDYNAMO_DISABLE=1"; x="--lr-scale 1.02";;
    sens5) envs="TORCHDYNAMO_DISABLE=1"; x="--lr-scale 1.05";;
  esac
  env $envs python -m tpu_smoke.rae_recipe_step --phase 1 --batch 512 --recipe-bf16 \
    --disc-ckpt <ASSETS>/dino_vit_small_patch8_224.pth \
    --traj 120 --seed $seed --sched-epoch 3 --data-pool <DATA>/pool384.npy \
    --eval-every 22 --eval-start 10 $x > logs/gate_${arm}_s${seed}.log 2>&1
done; done
```

Then collect worker 0's TRAJ lines as `<tag> <arm> s<seed> TRAJ {...}` and evaluate:

```bash
for f in logs/gate_*_s*.log; do b=$(basename $f .log); a=${b#gate_}; a=${a%_s*}
  grep -m1 '^TRAJ' $f | sed "s/^/w0 $a ${b##*_} /"; done > gate.txt
python3 tpu_smoke/gate_eval.py gate.txt --base up --endpoint eval   # fixed held-out set (version 5)
python3 tpu_smoke/gate_eval.py gate.txt --base up                   # training-loss windows
```

Without `--eval-every 22 --eval-start 10` the command is gate version 4, with training-loss endpoints. `--sched-epoch 3` starts the schedule near peak learning rate. Without it, phase 1 sits in the warmup from zero, as in version 1.

Placeholders: `<VENV>`, `<ASSETS>`, `<HF_CACHE>`, `<DATA>`, `<TPU_NAME>`, `<ZONE>`.
