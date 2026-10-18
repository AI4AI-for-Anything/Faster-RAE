# Faster-RAE

**3.22× faster Stage-1 training on 8 NVIDIA H20 GPUs**, preserving the upstream global batch of **512**.

| Implementation | PSNR (dB) ↑ | rFID ↓ | Iter-time (ms) ↓ | Speedup ↑ |
|---|---:|---:|---:|---:|
| RAE reference | 18.86 | 0.569 | 3,375.80 | 1.00× |
| **Faster-RAE** | **19.00** | **0.541** | **1,049.22** | **3.22×** |

**Timing:** fresh measurements on **8 H20s**, 256×256 images, **8 × 64 × 1 = 512** (64 images per GPU, no accumulation). Both timing rows use the current trainer at `6f0df05`, with optimization switches off/on. Iter-time weights reconstruction / discriminator / GAN phases by 6:2:8 and excludes compilation and warmup. Optimized sampled peak memory is **52.96 GiB/GPU**; this timing run uses the cards' full capacity, without an 80 GiB cap. [Timing protocol and results](docs/validation/h20-readme-benchmark-2026-10-15.md).

**Quality:** the reference is the upstream public FP32 decoder; Faster-RAE is the original `fd7800d0633a` snapshot's **16-epoch final EMA checkpoint**. Both were evaluated on **ImageNet validation 50k** with the same upstream pipeline; rFID uses ADM. All 73 source-manifest files match the completed H20 run. These quality numbers **predate the later gradient-accumulation / discriminator-DDP changes**; completed-training quality of the current trainer remains unvalidated. [Checkpoint provenance and evaluation](docs/validation/h20-full-training-quality-2026-10-10.md).

RAE Stage-1 training optimizations, based on [bytetriper/RAE](https://github.com/bytetriper/RAE) at `a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041`. The optimized source is candidate `fd7800d0633a`, the best validated snapshot retained by the H200 optimization run.

## What changed, and where did the speedup come from?

The change atlas separates the **original performance snapshot** from **later memory / gradient-accumulation adaptation**. It identifies the actual files against the upstream revision, with links and change counts; it does not assign a speedup to an individual file without an ablation.

![RAE file-change map](docs/change-views/module-map.svg)

| View | Best for |
|---|---|
| [File-by-file review table](docs/change-views/review-table.md) | Exact changed files, purpose, and diff size |
| [Module map](docs/change-views/module-map.svg) | Seeing how changes fit into the training pipeline |
| [Interactive file cards](docs/change-views/index.html) | Filtering the changes by phase and module; open the HTML locally |

![Historical RAE optimization progress](docs/performance-views/01-paper.svg)

The [performance gallery](docs/performance-views/README.md) presents the same recorded optimization history in three visual styles. Historical two-GPU measurements are kept separate from the current **8×H20, global-batch-512, 3.22×** result. Adjacent historical points are not isolated one-switch ablations; the gallery distinguishes reported same-run parent gains, remeasurement, and a rejected candidate.

## Adapt to your GPU first

For an 80 GiB per-GPU budget, start with [environment preparation](docs/agent-environment.md) and [the GPU adaptation workflow](docs/gpu-adaptation.md). An agent should read [AGENTS.md](AGENTS.md) and the repository-local [adaptation skill](skills/adapt-rae-gpu/SKILL.md). The independent research harness remains a separate release; see [the boundary](docs/release-boundary.md).

The adaptation tool checks the real environment, runs all three Stage-1 phases under an allocator budget plus device-memory monitoring, and chooses a microbatch plus gradient accumulation while preserving upstream global batch 512. The target defaults to 8 GPUs; `--world-size` selects another count. A single-GPU probe simulates one target rank. It produces capacity evidence, not a new numerical-equivalence or speedup verdict.

The fixed-batch H200 probes passed with **8 × 64 × 1 = 512 at 80 GiB** (53.73 GiB sampled per-device peak) and **8 × 32 × 2 = 512 at 48 GiB** (31.58 GiB peak after a microbatch-64 GAN OOM). These equations describe the target eight-GPU deployment; each probe executed **one rank's local batch 64**, including accumulation. See [the accumulation validation report](docs/validation/h200-accumulation-2026-10-15.md) for evidence and limitations. These are memory-limited H200 tests, not physical 80/48 GiB GPU or eight-GPU capacity measurements.

## Optimized training

Use the [tested environment guide](docs/agent-environment.md), prepare the upstream data and models described below, then enable the exact validated switches before starting training:

```bash
source scripts/fd7800d0633a.env.sh
export EXPERIMENT_NAME=faster-rae-training
torchrun --standalone --nproc_per_node=8 src/train_stage1.py \
  --config configs/stage1/training/DINOv2-B_decXL.yaml \
  --data-path /path/to/imagenet/train \
  --results-dir results/faster-rae \
  --precision fp32
```

The original configuration uses global batch 512 (64 per GPU on eight GPUs). The H20 timing above uses this eight-GPU batch plan; the original H200 optimization benchmark used two GPUs with 64 images per GPU. Local BF16 attention/linear operations and TF32 are enabled by the preset while the training precision argument remains `fp32`. The preset also enables regional compilation, cuDNN attention, and LPIPS/discriminator optimizations. The exact environment is preserved in [`.harness/config.json`](.harness/config.json).

## Historical validation

A separate [fresh H200-class benchmark](docs/validation/h200-readme-benchmark-2026-10-15.md) measured **5.41×** at global batch 512 on two GPUs with accumulation 4. It uses a different hardware and accumulation setup from the eight-H20 result above.

The recorded benchmark measured **345.13 ms/step**, **5.3167x** the same-run reference, and **48.52 GiB** peak memory for DINOv2-B + ViT-XL Stage-1 on two H200-class GPUs. These are historical measurements, not a fresh benchmark of this repository upload.

The fixed evaluator recorded **PASS / N2 / statistical equivalent** across three seeds with 300 steps per seed. The result is not bitwise equivalence: 3,434 numerical diagnostic violations and 11 unresolved diagnostics remain in the original evaluation record. Short trajectories used a compressed phase schedule and do not establish full 16-epoch quality or eight-GPU equivalence. Stage-2 performance is outside this snapshot's validation scope.

Source snapshot SHA-256: `b380522d42af751aa015b5ffc27cbb614c8b61e69bf6f3ce1dcacd814a47ba36`. [The original source manifest](docs/source-snapshot.json) records the 73 source files and their hashes; the initial import added this README introduction, the shell preset, and the manifest. Subsequent capacity-adaptation changes are tracked by Git and described in the adaptation documentation; the historical manifest identifies the original optimized snapshot, not every file in the current checkout. Model weights, training data, and machine-local experiment outputs are not distributed here.

## Upstream documentation

The original README follows, with upstream authorship and MIT license retained.

---

## Diffusion Transformers with Representation Autoencoders (RAE)<br><sub>Official PyTorch Implementation</sub>

### [Paper](https://arxiv.org/abs/2510.11690) | [Project Page](https://rae-dit.github.io/) 


This repository contains **PyTorch/GPU** and **TorchXLA/TPU** implementations of our paper: 
Diffusion Transformers with Representation Autoencoders. For JAX/TPU implementation, please refer to [diffuse_nnx](https://github.com/willisma/diffuse_nnx)

> [**Diffusion Transformers with Representation Autoencoders**](https://arxiv.org/abs/2510.11690)<br>
> [Boyang Zheng](https://bytetriper.github.io/), [Nanye Ma](https://willisma.github.io), [Shengbang Tong](https://tsb0601.github.io/),  [Saining Xie](https://www.sainingxie.com)
> <br>New York University<br>

We present Representation Autoencoders (RAE), a class of autoencoders that utilize  pretrained, frozen representation encoders such as [DINOv2](https://arxiv.org/abs/2304.07193) and [SigLIP2](https://arxiv.org/abs/2502.14786) as encoders with trained ViT decoders. RAE can be used in a two-stage training pipeline for high-fidelity image synthesis, where a Stage 2 diffusion model is trained on the latent space of a pretrained RAE to generate images.

This repository contains:

PyTorch/GPU:
* A PyTorch implementation of RAE and pretrained weights.
* A PyTorch implementation of LightningDiT, DiT<sup>DH</sup> and pretrained weights.
* Training and sampling scripts for the two-stage RAE+DiT pipeline.

TorchXLA/TPU:
* A TPU implementation of RAE and pretrained weights.
* Sampling of RAE and DiT<sup>DH</sup> on TPU.

### Update (Dec. 2025)

We **refactored the codebase** and added support for statistic calculation, training
resumption, Weights & Biases logging, online evaluation, and sanity checks (see the end of the page).
This is a **major update** over the Oct. 2025 release, and some APIs have changed.
Please refer to the [deprecated branch](https://github.com/bytetriper/RAE/tree/deprecated-gpu) for the old codebase.


>Note: You might need to update environment and script accordingly.

## Environment

### Dependency Setup
1. Create environment and install via `uv`:
   ```bash
   conda create -n rae python=3.10 -y
   conda activate rae
   pip install uv
   
   # Install PyTorch 2.8.0 with CUDA 12.9 # or your own cuda version
   uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129 
   
   # Install other dependencies
   uv pip install -r requirements.txt
   ```

## Data & Model Preparation

### Download Pre-trained Models

We release three kind of models: RAE decoders, DiT<sup>DH</sup> diffusion transformers and stats for latent normalization. To download all models at once:


```bash

cd RAE
pip install huggingface_hub
hf download nyu-visionx/RAE-collections \
  --local-dir models 
```


To download specific models, run:
```bash
hf download nyu-visionx/RAE-collections \
  <remote_model_path> \
  --local-dir models 
```

### Prepare Dataset

1. Download ImageNet-1k.
2. Point Stage 1 and Stage 2 scripts to the training split via `--data-path`.


## Config-based Initialization

All training and sampling entrypoints are driven by OmegaConf YAML files. A
single config describes the Stage 1 autoencoder, the Stage 2 diffusion model,
and the solver used during training or inference. A minimal example looks like:

```yaml
stage_1:
   target: stage1.RAE
   params: { ... }
   ckpt: <path_to_ckpt>  

stage_2:
   target: stage2.models.DDT.DiTwDDTHead
   params: { ... }
   ckpt: <path_to_ckpt>  

transport:
   params:
      path_type: Linear
      prediction: velocity
      ...
sampler:
   mode: ODE
   params:
      num_steps: 50
      ...
guidance:
   method: cfg/autoguidance
   scale: 1.0
   ...
misc:
   latent_size: [768, 16, 16]
   num_classes: 1000
training:
   ...
eval:
   ...
```

- `stage_1` instantiates the frozen encoder and trainable decoder. For Stage 1
  training you can point to an existing checkpoint via `stage_1.ckpt` or start
  from `pretrained_decoder_path`.
- `stage_2` defines the diffusion transformer. During sampling you must provide
  `ckpt`; during training you typically omit it so weights initialise randomly.
- `transport`, `sampler`, and `guidance` select the forward/backward SDE/ODE
  integrator and optional classifier-free or autoguidance schedule.
- `misc` collects shapes, class counts, and scaling constants used by both
  stages.
- `training` contains defaults that the training scripts consume (epochs,
  learning rate, EMA decay, gradient accumulation, etc.).
- `eval` contains settings for online evaluation during training.

Stage 1 training configs additionally include a top-level `gan` block that
configures the discriminator architecture and the LPIPS/GAN loss schedule.


### Provided Configs:

#### Stage1

We release decoders for DINOv2-B, SigLIP-B, MAE-B, at `configs/stage1/pretrained/`.

There is also a training script for training a ViT-XL decoder on DINOv2-B: `configs/stage1/training/DINOv2-B_decXL.yaml`

> **Note**: In the released training configs, optimization hyperparameters are tuned
to adapt to GPU hardware. The updated configs yield slightly better gFID than the
reported numbers.

#### Stage2

We release our best model, DiT<sup>DH</sup>-XL and it's guidance model on both $256\times 256$ and $512\times 512$, at `configs/stage2/sampling/`.

We also provide training configs for DiT<sup>DH</sup> at `configs/stage2/training/`.

## Stage 1: Representation Autoencoder

### Train the decoder

`src/train_stage1.py` trains the ViT decoder while keeping the
representation encoder frozen. Launch it with PyTorch DDP (single or multi-GPU):

```bash
torchrun --standalone --nproc_per_node=N \
  src/train_stage1.py \
  --config <config> \
  --data-path <imagenet_train_split> \
  --results-dir ckpts/stage1 \
  --image-size 256 --precision bf16/fp32 
```

where `N` refers to the number of GPU cards available.

You need to specify the checkpointing folder by `export EXPERIMENT_NAME="your_experiment_name"` before launching the script. The checkpoints and logs will be saved under `results-dir/$EXPERIMENT_NAME/`. 

**Logging.** To enable `wandb`, firstly set `WANDB_KEY`, `ENTITY`, and `PROJECT` as environment variables:

```bash
export WANDB_KEY="key"
export ENTITY="entity name"
export PROJECT="project name"
```

Then in training command add the `--wandb` flag


**Resuming.** If the checkpoint folder already exists (`results-dir/$EXPERIMENT_NAME/`), the script will automatically resume from the latest checkpoint.

**Online Eval.** The script supports online evaluation during training to monitor reconstruction quality. Paste the following block into the training config:

```yaml
eval:
  eval_interval: 2500 # Eval interval by optimization step
  eval_model: true # By default only evaluates EMA model. Set to true to eval non-EMA model as well.
  data_path: 'data/imagenet/val/' # path to ImageNet val images
  reference_npz_path: 'data/imagenet/val_256.npz' # packed npz of ImageNet val images for FID calculation
  metrics: ['psnr', 'ssim', 'rfid'] # metrics to calculate
```

**torch.compile.** Use `--compile` flag to enable `torch.compile` for potentially faster training. 
### Sampling/Reconstruction

Use `src/stage1_sample.py` to encode/decode a single image:

```bash
python src/stage1_sample.py \
  --config <config> \
  --image assets/pixabay_cat.png \
```

For batched reconstructions and `.npz` export, run the DDP variant:

```bash
torchrun --standalone --nproc_per_node=N \
  src/stage1_sample_ddp.py \
  --config <config> \
  --data-path <imagenet_val_split> \
  --sample-dir recon_samples \
  --image-size 256
```

The script writes per-image PNGs as well as a packed `.npz` suitable for FID.


We also provide an individual script `pack_images.py` to convert a folder of images into a FID-ready `.npz`:

```bash
python src/pack_images.py <image_folder> 
```

### Calculating Encoder Statistics

We use a batchnorm-like normalization layer in the latent space for calculating the mean and variance statistics of the encoder outputs. Specifically, we calculate the mean and variance of the same shape as the latent `[C, H, W]`. To run:

```bash
torchrun --standalone --nproc_per_node=N \
  src/calculate_stat.py \
  --config <config> \ # w/o pre-computed stats
  --sample-dir stats \
  --precision fp32/bf16 \
  --data-path <imagenet_train_split> \
  --image-size 256 \
  --tf32
```

We note that the results may differ slightly from the released statistics because we use a momentum-based update rule for the mean and variance. As a result, the outcome is not deterministic and depends on batch size and data shuffling.

As a reproduction, we ran the script to calculate the statistic of DINOv2-B on 4 GPUs with a batch size of 256 under fp32. The config is exactly the same as `configs/stage1/pretrained/DINOv2-B.yaml` except commenting out the pre-computed stats. The computed variance give around `0.008` L1 difference compared to the released statistics.


## Stage 2: Latent Diffusion Transformer

### Training

`src/train.py` trains the Stage 2 diffusion transformer using PyTorch DDP. Edit
one of the configs under `configs/training/` and launch:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=N \
  src/train.py \
  --config <training_config> \
  --data-path <imagenet_train_split> \
  --results-dir ckpts/stage2 \
  --precision fp32
```

Although `bf16` is supported, we recommend using **fp32** for more stable training.

The logging and checkpointing behaviour is the same as Stage 1 training.

**Online Eval**: Paste the following block into training config to support online eval:

```yaml
eval:
  eval_interval: 25000 
  eval_model: true
  data_path: 'data/imagenet/val/'
  reference_npz_path: 'data/imagenet/VIRTUAL_imagenet256_labeled.npz'
```

### Sampling

`src/sample.py` uses the same config schema to draw a small batch of images on a
single device and saves them to `sample.png`:

```bash
python src/sample.py \
  --config <sample_config> \
  --seed 42
```

### Distributed sampling for evaluation

`src/sample_ddp.py` parallelises sampling across GPUs, producing PNGs and an
FID-ready `.npz`:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=N \
  src/sample_ddp.py \
  --config <sample_config> \
  --sample-dir samples \
  --precision fp32/bf16 \
  --label-sampling equal
```
`--label-sampling {equal,random}`: `equal` uses exactly 50 images per class for FID-50k; `random` uniformly samples labels. Using `equal` brings consistently lower FID than `random` by around 0.1. We use `equal` by default. We recommend using fp32 when model FID is low.

Autoguidance and classifier-free guidance are controlled via the config’s `guidance` block.


## Evaluation

### ADM Suite FID setup

Use the ADM evaluation suite to score generated samples:

1. Clone the repo:

   ```bash
   git clone https://github.com/openai/guided-diffusion.git
   cd guided-diffusion/evaluation
   ```

2. Create an environment and install dependencies:

   ```bash
   conda create -n adm-fid python=3.10
   conda activate adm-fid
   pip install 'tensorflow[and-cuda]'==2.19 scipy requests tqdm
   ```

3. Download ImageNet statistics (256×256 shown here):

   ```bash
   wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
   ```

4. Evaluate:

   ```bash
   python evaluator.py VIRTUAL_imagenet256_labeled.npz /path/to/samples.npz
   ```

## TorchXLA / TPU support

See `XLA` branch for TPU support.


## GPU training reproduction results:

---
| Model                | rFID-50k (reported) | rFID-50k (reproduced) |
|----------------------|--------------------|--------------------|
| DINOv2-B  ($\tau$ = 0.8)| 0.57 | 0.54 |


| Model  (80 epochs)             | gFID-50k (reported) | gFID-50k (reproduced) |
|---------------------|--------------------|--------------------|
| DiT<sup>DH</sup>-XL (DINOv2-B) |  2.16 | 2.16|
---


## Acknowledgement

This code is built upon the following repositories:

* [SiT](https://github.com/willisma/sit) - for diffusion implementation and training codebase.
* [DDT](https://github.com/MCG-NJU/DDT) - for some of the DiT<sup>DH</sup> implementation.
* [LightningDiT](https://github.com/hustvl/LightningDiT/) - for the PyTorch Lightning based DiT implementation.
* [MAE](https://github.com/facebookresearch/mae) - for the ViT decoder architecture.