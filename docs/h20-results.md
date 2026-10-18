# H20 training and performance results

Faster-RAE completed **16 epochs of ImageNet training on 8 NVIDIA H20 GPUs**. A separate timing experiment measured **3.22×** faster Stage-1 iterations with the optimization preset enabled.

## Completed-training quality

The final EMA decoder was evaluated on all **50,000 ImageNet-1k validation images** using the upstream reconstruction and evaluation pipeline. The reference is the publicly released FP32 decoder.

| Model | PSNR (dB) ↑ | rFID (ADM) ↓ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| Upstream public FP32 decoder | 18.861973 | 0.568864 | 0.483470 | 0.255582 |
| Faster-RAE, final EMA | 19.000647 | 0.541238 | 0.480697 | 0.247410 |

Training used DINOv2-with-registers base + ViT-XL, 256×256 images, global batch **512** (8 GPUs × 64 images, no gradient accumulation), and completed **40,032 generator updates**. The upstream learning-rate schedule and loss settings were retained. The precision argument was `fp32`, with TF32 and selected local BF16 operations enabled by the optimization preset.

Evaluation used ADM center cropping at 256×256, the same image order for references and reconstructions, and the upstream PSNR, SSIM and LPIPS implementations. rFID above uses ADM; torch-fidelity measured 0.568879 for the reference and 0.541289 for Faster-RAE. Evaluation completed on October 10, 2026.

These quality results cover one completed run of the original optimized implementation. Later gradient-accumulation and discriminator-DDP changes have not had a separate full-training quality evaluation.

## Training speed

The October 15, 2026 timing experiment compared the same trainer, data and batch plan with optimization switches off and on. It used **8 H20 GPUs × microbatch 64 × accumulation 1 = global batch 512**, with DINOv2-B + ViT-XL and 256×256 images.

| Configuration | Weighted iteration time (ms) ↓ | Speedup ↑ |
|---|---:|---:|
| Baseline (preset off) | 3,375.80 | 1.00× |
| Faster-RAE (preset on) | 1,049.22 | 3.22× |

The GPUs exposed approximately 95 GiB each, with no 80 GiB cap. The environment used Python 3.10, PyTorch 2.8.0 with CUDA 12.8, torchvision 0.23.0, Triton 3.4.0 and transformers 4.56.2. The preset is recorded in [`.harness/config.json`](../.harness/config.json).

Each run covered reconstruction + LPIPS, discriminator updates and adversarial generator updates, with five warmup updates and two five-update timing windows per phase. Window timing used the slowest rank. Phase times were weighted **6:2:8**; the baseline averages runs before and after the optimized run.

| Run | Reconstruction + LPIPS (ms) | D-update phase (ms) | GAN phase (ms) | Sampled device peak (GiB) |
|---|---:|---:|---:|---:|
| Baseline A | 2,260.80 | 3,467.17 | 4,182.74 | 93.78 |
| Faster-RAE | 689.96 | 1,064.36 | 1,314.88 | 52.96 |
| Baseline B | 2,277.76 | 3,469.87 | 4,182.28 | 93.78 |

These are steady-state iteration times, excluding compilation, warmup, checkpoint saving and evaluation. The timing experiment is separate from the completed training run above; it does not measure total 16-epoch training time or establish new quality results.
