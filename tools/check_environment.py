#!/usr/bin/env python3
"""Check the actual training imports, local assets, and CUDA/compiler path."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def check(args):
    result = {'ready': False, 'python': sys.version, 'versions': {}, 'tools': {}, 'errors': []}
    for package in ('torch', 'torchvision', 'triton', 'transformers', 'timm', 'numpy', 'omegaconf',
                    'torchmetrics', 'torch-fidelity', 'wandb', 'scipy', 'Pillow'):
        try:
            result['versions'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result['errors'].append('Missing package: ' + package)
    for name in ('gcc', 'g++', 'ninja', 'nvidia-smi', 'nsys', 'ncu'):
        result['tools'][name] = shutil.which(name)
    for name in ('gcc', 'g++', 'nvidia-smi'):
        if result['tools'][name] is None:
            result['errors'].append('Missing required tool: ' + name)
    result['scratch_free_gib'] = shutil.disk_usage(os.environ.get('TMPDIR', '/tmp')).free / 2**30
    result['output_free_gib'] = shutil.disk_usage(args.output.parent).free / 2**30
    if min(result['scratch_free_gib'], result['output_free_gib']) < 5:
        result['errors'].append('Need at least 5 GiB free in scratch and output storage')
    try:
        import torch
        import train_stage1  # All real eager imports, including evaluation dependencies.
        from omegaconf import OmegaConf
        from torchvision.datasets import ImageFolder
        from transformers import AutoConfig, AutoImageProcessor
        from huggingface_hub import snapshot_download
        from disc.lpips_utils import md5_hash, MD5_MAP
        cfg = OmegaConf.load(args.config)
        model_id = cfg.stage_1.params.encoder_params.dinov2_path
        local = Path(model_id)
        if not local.is_dir():
            local = Path(snapshot_download(model_id, local_files_only=True))
        AutoConfig.from_pretrained(local, local_files_only=True)
        AutoImageProcessor.from_pretrained(local, local_files_only=True)
        if not any(local.glob('*.safetensors')) and not any(local.glob('pytorch_model*.bin')):
            raise FileNotFoundError('Encoder weights missing from local Hugging Face snapshot')
        AutoConfig.from_pretrained(cfg.stage_1.params.decoder_config_path, local_files_only=True)
        disc = Path(args.disc_checkpoint or cfg.gan.disc.arch.dino_ckpt_path)
        if not disc.is_file():
            raise FileNotFoundError('DINO discriminator checkpoint is missing: ' + str(disc))
        lpips = Path(os.environ.get('RAE_LPIPS_CHECKPOINT', ROOT / 'src/disc/.caches/vgg.pth'))
        if not lpips.is_file() or md5_hash(lpips) != MD5_MAP['vgg_lpips']:
            raise ValueError('LPIPS linear weights missing/invalid; set RAE_LPIPS_CHECKPOINT')
        vgg = Path(torch.hub.get_dir()) / 'checkpoints/vgg16-397923af.pth'
        if not vgg.is_file():
            raise FileNotFoundError('VGG16 backbone is missing from TORCH_HOME hub/checkpoints')
        result['assets'] = {'encoder_local': True, 'disc_bytes': disc.stat().st_size,
                            'lpips_md5_valid': True, 'vgg_bytes': vgg.stat().st_size}
        if args.data_path:
            dataset = ImageFolder(args.data_path)
            result['dataset_images'] = len(dataset)
            if len(dataset) < args.min_images:
                raise ValueError(f'Need at least {args.min_images} images for this batch/step plan')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES')
        prop = torch.cuda.get_device_properties(0)
        result['gpu'] = {'name': prop.name, 'capability': list(torch.cuda.get_device_capability()),
                         'total_gib': prop.total_memory / 2**30, 'cuda': torch.version.cuda,
                         'cudnn': torch.backends.cudnn.version(), 'bf16': torch.cuda.is_bf16_supported()}
        if not result['gpu']['bf16']:
            raise RuntimeError('The validated preset requires BF16 support')
        # This actually exercises Inductor's compiler/toolchain, forward and backward.
        x = torch.randn(128, 128, device='cuda', requires_grad=True)
        torch.compile(lambda t: (t.sin() * t).sum(), fullgraph=True)(x).backward()
        torch.cuda.synchronize()
        result['compile_backward_ok'] = bool(torch.isfinite(x.grad).all())
        if not result['compile_backward_ok']:
            raise RuntimeError('Compiler smoke produced non-finite gradients')
        result['attention_backends'] = {}
        from torch.nn.attention import SDPBackend, sdpa_kernel
        for name, backend in (('cudnn', SDPBackend.CUDNN_ATTENTION), ('flash', SDPBackend.FLASH_ATTENTION)):
            try:
                qkv = [torch.randn(1, 16, 256, 72, device='cuda', dtype=torch.bfloat16,
                                   requires_grad=True) for _ in range(3)]
                with sdpa_kernel(backend):
                    torch.nn.functional.scaled_dot_product_attention(*qkv).float().sum().backward()
                torch.cuda.synchronize()
                result['attention_backends'][name] = all(bool(torch.isfinite(t.grad).all()) for t in qkv)
            except Exception as e:
                result['attention_backends'][name] = str(e)
    except Exception as e:
        result['errors'].append(type(e).__name__ + ': ' + str(e))
    result['ready'] = not result['errors']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/stage1/training/DINOv2-B_decXL.yaml')
    parser.add_argument('--data-path')
    parser.add_argument('--disc-checkpoint')
    parser.add_argument('--min-images', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    result = check(args)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['ready'] else 2)
