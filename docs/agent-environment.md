# Environment preparation before GPU adaptation

Read this before the adaptation agent runs experiments. Work from the repository root in an isolated environment. This workflow requires Linux and a CUDA-capable NVIDIA GPU with BF16 support. The recorded reference stack is Python 3.10, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, Triton 3.4.0, and transformers 4.56.2. The inherited `environment.yml` describes an older upstream stack; use this recipe for the optimized code.

## Install the runtime

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-adapt.txt
python -m pip check
```

The torch/torchvision pairing and CUDA wheel index follow [PyTorch's official previous-version instructions](https://pytorch.org/get-started/previous-versions/). Check the installed driver against that stack. `nvidia-smi`'s CUDA label is the driver's supported API, not proof of the Python wheel's version. Verify both in preflight.

On a managed machine with this exact working stack, an isolated `python -m venv --system-site-packages <new-directory>` is also usable. It reuses packages and is not a clean dependency installation test. Never pip-install into a shared research environment. Record which preparation route was used.

`gcc`, `g++`, and `nvidia-smi` must be on PATH. PyTorch's CUDA wheel supplies the runtime and Triton; this workflow has no custom CUDA extension and does not require installing flash-attn or a separate CUDA toolkit merely to start. `ninja` can help other compilation workloads but is not a prerequisite here. The preflight actually compiles and backpropagates a small Inductor workload, so a broken compiler/toolchain fails before model training.

Keep at least 5 GiB free in both scratch and output storage; allow more for compiler caches and traces. Set `TORCHINDUCTOR_COMPILE_THREADS=4` to bound compilation CPU concurrency on a shared host. Use short `TMPDIR` paths if DataLoader workers report `AF_UNIX path too long`. CPU/pinned-memory and worker prefetch use the effective per-rank batch, not the smaller CUDA microbatch. Check `/dev/shm` when increasing `--workers` or reducing target GPU count; `--workers 0` is a diagnostic option for worker/shared-memory failures.

## Prepare assets once

Use data and weights you are authorized to access. Prepare them before probing, then keep them read-only. The probe is offline and does not download missing assets.

| Input | Expected location or setting |
|---|---|
| DINOv2 with registers encoder | Local HF cache for `facebook/dinov2-with-registers-base`, or local directory in both `stage_1.params.encoder_config_path` and `encoder_params.dinov2_path` |
| Encoder processor/config | In the same local HF snapshot; model weights alone are insufficient |
| Decoder configuration | Tracked `configs/decoder/ViTXL/config.json` |
| DINO discriminator weights | `models/discs/dino_vit_small_patch8_224.pth`, or `--disc-checkpoint /absolute/path` |
| VGG16 backbone | `$TORCH_HOME/hub/checkpoints/vgg16-397923af.pth` |
| LPIPS linear weights | `RAE_LPIPS_CHECKPOINT=/absolute/path/vgg.pth`, MD5 `d507d7349b931f0638a25a48a722f98a` |
| Images | `ImageFolder`: class subdirectories of readable images, at least one effective per-rank batch (`global_batch_size / target_world_size`; default 64) |

The upstream README describes public asset sources. VGG16's roughly 528 MiB backbone and LPIPS's small `vgg.pth` are different files. Merely preparing one does not prepare the other. A small real ImageFolder subset suffices for the capacity probe; full datasets and model files should never be committed.

```bash
export HF_HOME=/path/to/huggingface-cache
export TORCH_HOME=/path/to/torch-cache
export RAE_LPIPS_CHECKPOINT=/path/to/lpips/vgg.pth
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_COMPILE_THREADS=4
```

Use filesystem paths local to the compute machine. Do not stage company OSS data through a laptop or another cluster. Any stricter local data-governance rules apply to reports, inventories, and metadata too.

## Preflight

The adaptation launcher runs this automatically. For standalone diagnosis, select an idle GPU first:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/check_environment.py \
  --data-path /path/to/imagefolder \
  --disc-checkpoint /path/to/dino_vit_small_patch8_224.pth \
  --min-images 64 --output runs/preflight/environment.json
```

The check imports the actual training entrypoint, including torchvision, torchmetrics, torch-fidelity and its evaluation dependencies; verifies local asset availability and LPIPS checksum; records package/tool versions; and executes CUDA compilation/backward and small cuDNN/Flash SDPA probes. The attention probes are preliminary support checks, not validation of every model shape. Missing assets/imports or failed compilation stop preparation. `python -m pip check` separately catches dependency conflicts.

## Tools to use when adaptation fails

| Symptom | First evidence/tool | Next action |
|---|---|---|
| Import/ABI/package error | `environment.json`, `python -m pip check` | Repair the isolated environment; do not reduce batch |
| OOM | Phase log, allocated/reserved peaks, sampled `nvidia-smi` memory | Reduce microbatch and increase accumulation at fixed global batch; retain failed attempt |
| SDPA kernel unavailable | Actual phase error plus preflight backend results | Test the alternate explicit backend and rerun all phases |
| Compile failure | Phase log; installed compiler; torch/Triton versions | Repair toolchain or create an explicitly different preset; do not silently drop compilation |
| Capacity fits but time is surprising | A separate `--profile` run | Inspect PyTorch operator table and Chrome trace; profiler timings are diagnostic |
| Need CUDA timeline detail | Optional Nsight Systems (`nsys`) | Install the official standalone tool for the host if permitted; bracket a short warmed workload |
| Need a specific kernel's counters | Optional Nsight Compute (`ncu`) | Check counter permissions; use scheduler/admin-supported access, not privilege changes |

PyTorch profiler and allocator statistics are included in the torch install. Nsight is optional: no nsys/ncu installation or privileged counters are needed to establish whether the 80 GiB workload runs. A full kernel optimization investigation belongs in the separate harness.
