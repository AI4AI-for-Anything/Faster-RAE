# Faster-RAE

[English](README.md) | [简体中文](README.zh-CN.md)

**在 H200 GPU 上，RAE 第一阶段训练速度达到基线的 5.41 倍，全部优化由 AI 智能体完成。**

![Faster-RAE：全部优化由 AI 智能体完成，在 H200 GPU 上实现 5.41 倍加速](docs/performance-views/04-h200-benchmark.svg)

| 实现 | 加权迭代耗时（ms）↓ | 加速比 ↑ |
|---|---:|---:|
| 基线（关闭优化开关）| 8,381.01 | 1.00× |
| **Faster-RAE** | **1,547.90** | **5.41×** |

以上结果在 DINOv2-B + ViT-XL、256×256 图像分辨率的设置下测得。

## 快速开始

在 Linux/CUDA 环境中，先准备好[运行环境](docs/agent-environment.md#install-the-runtime)、[数据和模型权重](docs/agent-environment.md#prepare-assets-once)，再启动训练。下面以 **8 张 GPU** 为例，每卡 64 张图像，全局批量大小为 512：

```bash
source scripts/fd7800d0633a.env.sh
export EXPERIMENT_NAME=faster-rae-training
torchrun --standalone --nproc_per_node=8 src/train_stage1.py \
  --config configs/stage1/training/DINOv2-B_decXL.yaml \
  --data-path /path/to/imagenet/train \
  --results-dir results/faster-rae \
  --precision fp32
```

使用其他数量的 GPU，或显存不足时，请按 [GPU 适配指南](docs/gpu-adaptation.md)调整每卡单次处理的图像数和梯度累积次数，保持全局批量大小 512。

## Agent 优化过程

此前的优化过程使用 **2 张 H200 GPU，每卡 64 张图像，全局批量大小 128**，同样按 6:2:8 对各阶段耗时加权。最终保留的配置测得 **345.13 ms/步**，速度为同次测试中基线的 **5.3167 倍**，显存峰值为 **48.52 GiB**。

![此前的 H200 优化过程，全局批量大小 128](docs/performance-views/01-paper.svg)

## 主要改动

- **解码器：** 分块编译，启用 TF32，部分注意力和线性运算使用 BF16。
- **LPIPS 和判别器：** 调整 LPIPS 的内存布局，编译 LPIPS 距离计算和判别器中的模型块、归一化运算；判别器的部分注意力和线性运算使用 BF16。
- **显存适配：** 通过调整每卡单次处理的图像数和梯度累积次数，保持全局批量大小 512。适配工具会检查第一阶段的全部三个训练阶段，再生成训练配置。

训练精度参数仍为 `fp32`，预设脚本会启用指定运算的 BF16/TF32 路径。完整开关设置见 [`.harness/config.json`](.harness/config.json)。

![解码器、LPIPS、判别器和显存适配工具的文件改动](docs/change-views/module-map.svg)

[逐文件说明](docs/change-views/review-table.md)列出了改动位置和用途；也可以在本地打开[交互式文件列表](docs/change-views/index.html)，按模块筛选。图中分别列出了最初的性能优化和后续 GPU 适配改动。各项优化单独带来多少加速，需要通过消融实验确定。

## 完整训练结果

我们已在 **8 张 H20 上完成一次 ImageNet 训练，共 16 轮**，并使用训练结束后的 EMA 模型，按上游评估流程测试了 **ImageNet 验证集的全部 50,000 张图像**。

| 模型 | PSNR（dB）↑ | rFID ↓ |
|---|---:|---:|
| 上游公开的 FP32 解码器 | 18.86 | 0.569 |
| **Faster-RAE，最终 EMA 模型** | **19.00** | **0.541** |

这次训练共完成 40,032 次生成器更新。另一次八卡 H20 耗时测试在全局批量大小 512 下测得 **3.22×** 加速。训练质量、速度和测试条件见 [H20 结果](docs/h20-results.md)。

## 上游 RAE 文档

本项目基于 [bytetriper/RAE](https://github.com/bytetriper/RAE)，原作者为 Boyang Zheng、Nanye Ma、Shengbang Tong 和 Saining Xie。[论文](https://arxiv.org/abs/2510.11690) · [项目主页](https://rae-dit.github.io/) · [MIT 许可证](LICENSE)。下方保留上游 README 的中文译文和原作者署名。

<details>
<summary>展开上游 RAE 文档：环境、数据、训练与评估</summary>

## 采用表征自编码器的扩散 Transformer（RAE）<br><sub>官方 PyTorch 实现</sub>

### [论文](https://arxiv.org/abs/2510.11690) | [项目主页](https://rae-dit.github.io/)

本仓库提供论文《Diffusion Transformers with Representation Autoencoders》的 **PyTorch/GPU** 和 **TorchXLA/TPU** 实现。JAX/TPU 实现请参见 [diffuse_nnx](https://github.com/willisma/diffuse_nnx)。

> [**Diffusion Transformers with Representation Autoencoders**](https://arxiv.org/abs/2510.11690)<br>
> [Boyang Zheng](https://bytetriper.github.io/), [Nanye Ma](https://willisma.github.io), [Shengbang Tong](https://tsb0601.github.io/), [Saining Xie](https://www.sainingxie.com)
> <br>纽约大学<br>

我们提出表征自编码器（Representation Autoencoders，RAE）。这类自编码器采用预训练且参数冻结的表征编码器，如 [DINOv2](https://arxiv.org/abs/2304.07193) 和 [SigLIP2](https://arxiv.org/abs/2502.14786)，并训练 ViT 解码器。RAE 可用于高保真图像合成的两阶段训练流程：第二阶段在预训练 RAE 的潜在空间中训练扩散模型，用于生成图像。

本仓库包含：

PyTorch/GPU：

* RAE 的 PyTorch 实现及预训练权重。
* LightningDiT、DiT<sup>DH</sup> 的 PyTorch 实现及预训练权重。
* 两阶段 RAE+DiT 流程的训练与采样脚本。

TorchXLA/TPU：

* RAE 的 TPU 实现及预训练权重。
* RAE 和 DiT<sup>DH</sup> 的 TPU 采样支持。

### 更新（2025 年 12 月）

我们**重构了代码库**，并增加了统计量计算、训练恢复、Weights & Biases 日志、训练期间在线评估和基本正确性检查的支持（见本页后文）。相较于 2025 年 10 月的版本，这是一次**重大更新**，部分 API 已发生变化。旧版代码请参见[已弃用分支](https://github.com/bytetriper/RAE/tree/deprecated-gpu)。

> 注意：你可能需要相应更新运行环境和脚本。

## 环境

### 安装依赖

1. 创建环境，并通过 `uv` 安装依赖：

   ```bash
   conda create -n rae python=3.10 -y
   conda activate rae
   pip install uv

   # Install PyTorch 2.8.0 with CUDA 12.9 # or your own cuda version
   uv pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

   # Install other dependencies
   uv pip install -r requirements.txt
   ```

## 数据与模型准备

### 下载预训练模型

我们发布了三类模型及相关文件：RAE 解码器、DiT<sup>DH</sup> 扩散 Transformer，以及用于潜在空间归一化的统计量。一次性下载全部模型：

```bash

cd RAE
pip install huggingface_hub
hf download nyu-visionx/RAE-collections \
  --local-dir models
```

下载指定模型：

```bash
hf download nyu-visionx/RAE-collections \
  <remote_model_path> \
  --local-dir models
```

### 准备数据集

1. 下载 ImageNet-1k。
2. 通过 `--data-path` 为第一阶段和第二阶段脚本指定训练集路径。

## 基于配置的初始化

所有训练和采样入口均由 OmegaConf YAML 文件配置。一个配置文件描述第一阶段自编码器、第二阶段扩散模型，以及训练或推理时使用的求解器。最小示例如下：

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

- `stage_1` 实例化参数冻结的编码器和可训练的解码器。第一阶段训练可通过 `stage_1.ckpt` 指定已有检查点，也可从 `pretrained_decoder_path` 指定的权重开始。
- `stage_2` 定义扩散 Transformer。采样时必须提供 `ckpt`；训练时通常省略此项，以随机初始化权重。
- `transport`、`sampler` 和 `guidance` 用于选择正向或反向 SDE/ODE 积分器，以及可选的无分类器引导（classifier-free guidance）或自引导（autoguidance）调度。
- `misc` 汇总两个阶段使用的形状、类别数和缩放常数。
- `training` 包含训练脚本使用的默认参数，如训练轮次、学习率、EMA 衰减系数和梯度累积设置。
- `eval` 包含训练期间在线评估的设置。

第一阶段训练配置还包含顶层 `gan` 配置块，用于设置判别器架构及 LPIPS/GAN 损失调度。

### 已提供的配置

#### 第一阶段

我们在 `configs/stage1/pretrained/` 中提供了 DINOv2-B、SigLIP-B 和 MAE-B 对应的解码器。

另提供了用于在 DINOv2-B 上训练 ViT-XL 解码器的训练脚本：`configs/stage1/training/DINOv2-B_decXL.yaml`。

> **注意：** 已发布训练配置中的优化超参数经过调整，以适配 GPU 硬件。更新后的配置得到的 gFID 略优于报告值。

#### 第二阶段

我们在 `configs/stage2/sampling/` 中提供了最佳模型 DiT<sup>DH</sup>-XL 及其引导模型，覆盖 $256\times 256$ 和 $512\times 512$ 两种分辨率。

此外，`configs/stage2/training/` 中提供了 DiT<sup>DH</sup> 的训练配置。

## 第一阶段：表征自编码器

### 训练解码器

`src/train_stage1.py` 在保持表征编码器参数冻结的同时，训练 ViT 解码器。使用 PyTorch DDP 启动，支持单 GPU 或多 GPU：

```bash
torchrun --standalone --nproc_per_node=N \
  src/train_stage1.py \
  --config <config> \
  --data-path <imagenet_train_split> \
  --results-dir ckpts/stage1 \
  --image-size 256 --precision bf16/fp32
```

其中，`N` 为可用的 GPU 数量。

启动脚本前，需通过 `export EXPERIMENT_NAME="your_experiment_name"` 指定检查点目录名称。检查点和日志将保存在 `results-dir/$EXPERIMENT_NAME/` 下。

**日志记录。** 如需启用 `wandb`，先设置环境变量 `WANDB_KEY`、`ENTITY` 和 `PROJECT`：

```bash
export WANDB_KEY="key"
export ENTITY="entity name"
export PROJECT="project name"
```

然后在训练命令中添加 `--wandb` 参数。

**恢复训练。** 如果检查点目录（`results-dir/$EXPERIMENT_NAME/`）已存在，脚本会自动从最新的检查点恢复训练。

**在线评估。** 脚本支持在训练期间进行在线评估，以监测重建质量。将以下配置块加入训练配置：

```yaml
eval:
  eval_interval: 2500 # Eval interval by optimization step
  eval_model: true # By default only evaluates EMA model. Set to true to eval non-EMA model as well.
  data_path: 'data/imagenet/val/' # path to ImageNet val images
  reference_npz_path: 'data/imagenet/val_256.npz' # packed npz of ImageNet val images for FID calculation
  metrics: ['psnr', 'ssim', 'rfid'] # metrics to calculate
```

**torch.compile。** 添加 `--compile` 参数可启用 `torch.compile`，可能加快训练。

### 采样与重建

使用 `src/stage1_sample.py` 对单张图像进行编码和解码：

```bash
python src/stage1_sample.py \
  --config <config> \
  --image assets/pixabay_cat.png \
```

如需批量重建并导出 `.npz` 文件，使用 DDP 版本：

```bash
torchrun --standalone --nproc_per_node=N \
  src/stage1_sample_ddp.py \
  --config <config> \
  --data-path <imagenet_val_split> \
  --sample-dir recon_samples \
  --image-size 256
```

脚本会将每张图像保存为 PNG，同时生成可用于 FID 计算的打包 `.npz` 文件。

我们还提供独立脚本 `pack_images.py`，可将一个文件夹中的图像转换为适用于 FID 计算的 `.npz` 文件：

```bash
python src/pack_images.py <image_folder>
```

### 计算编码器统计量

我们在潜在空间中使用类似批归一化（BatchNorm）的归一化层，计算编码器输出的均值和方差统计量。具体而言，计算得到的均值和方差均与潜在表示具有相同的形状 `[C, H, W]`。运行方式如下：

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

由于均值和方差采用基于动量的更新规则，计算结果可能与已发布的统计量略有不同。因此，结果并非确定性的，会受到批量大小和数据打乱顺序的影响。

在一次复现实验中，我们使用 4 张 GPU、批量大小 256 和 fp32 精度，运行该脚本计算 DINOv2-B 的统计量。配置与 `configs/stage1/pretrained/DINOv2-B.yaml` 完全相同，唯一区别是注释掉了预计算统计量。计算得到的方差与已发布统计量之间的 L1 差异约为 `0.008`。

## 第二阶段：潜在空间扩散 Transformer

### 训练

`src/train.py` 使用 PyTorch DDP 训练第二阶段扩散 Transformer。修改 `configs/training/` 下的一个配置文件，然后运行：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=N \
  src/train.py \
  --config <training_config> \
  --data-path <imagenet_train_split> \
  --results-dir ckpts/stage2 \
  --precision fp32
```

虽然支持 `bf16`，但我们推荐使用 **fp32**，以获得更稳定的训练。

日志记录和检查点保存行为与第一阶段训练相同。

**在线评估：** 将以下配置块加入训练配置，以启用在线评估：

```yaml
eval:
  eval_interval: 25000
  eval_model: true
  data_path: 'data/imagenet/val/'
  reference_npz_path: 'data/imagenet/VIRTUAL_imagenet256_labeled.npz'
```

### 采样

`src/sample.py` 使用相同的配置结构，在单个设备上采样生成一小批图像，并保存为 `sample.png`：

```bash
python src/sample.py \
  --config <sample_config> \
  --seed 42
```

### 用于评估的分布式采样

`src/sample_ddp.py` 在多张 GPU 上并行采样，生成 PNG 图像及可用于 FID 计算的 `.npz` 文件：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=N \
  src/sample_ddp.py \
  --config <sample_config> \
  --sample-dir samples \
  --precision fp32/bf16 \
  --label-sampling equal
```

`--label-sampling {equal,random}`：`equal` 为每个类别生成恰好 50 张图像，用于 FID-50k 评估；`random` 均匀随机采样类别标签。使用 `equal` 得到的 FID 稳定地比 `random` 低约 0.1。我们默认使用 `equal`。当模型 FID 较低时，建议使用 fp32。

自引导（autoguidance）和无分类器引导（classifier-free guidance）通过配置中的 `guidance` 块控制。

## 评估

### 配置 ADM 评估套件以计算 FID

使用 ADM 评估套件对生成样本进行评估：

1. 克隆仓库：

   ```bash
   git clone https://github.com/openai/guided-diffusion.git
   cd guided-diffusion/evaluation
   ```

2. 创建环境并安装依赖：

   ```bash
   conda create -n adm-fid python=3.10
   conda activate adm-fid
   pip install 'tensorflow[and-cuda]'==2.19 scipy requests tqdm
   ```

3. 下载 ImageNet 统计量（此处以 256×256 为例）：

   ```bash
   wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
   ```

4. 运行评估：

   ```bash
   python evaluator.py VIRTUAL_imagenet256_labeled.npz /path/to/samples.npz
   ```

## TorchXLA / TPU 支持

TPU 支持请参见 `XLA` 分支。

## GPU 训练复现结果

---

| 模型 | rFID-50k（报告值） | rFID-50k（复现值） |
|----------------------|--------------------|--------------------|
| DINOv2-B（$\tau$ = 0.8）| 0.57 | 0.54 |

| 模型（80 个训练轮次）| gFID-50k（报告值） | gFID-50k（复现值） |
|---------------------|--------------------|--------------------|
| DiT<sup>DH</sup>-XL (DINOv2-B) | 2.16 | 2.16 |

---

## 致谢

本代码基于以下仓库：

* [SiT](https://github.com/willisma/sit)：扩散模型实现和训练代码。
* [DDT](https://github.com/MCG-NJU/DDT)：部分 DiT<sup>DH</sup> 实现。
* [LightningDiT](https://github.com/hustvl/LightningDiT/)：基于 PyTorch Lightning 的 DiT 实现。
* [MAE](https://github.com/facebookresearch/mae)：ViT 解码器架构。

</details>
