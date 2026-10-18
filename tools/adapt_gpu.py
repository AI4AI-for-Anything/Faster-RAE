#!/usr/bin/env python3
"""Bounded single-GPU capacity search using the real Stage-1 trainer."""
import argparse
import csv
import io
import hashlib
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from utils.batch_plan import batch_plan
PHASES = ('recon', 'disc', 'gan')


def source_fingerprint():
    digest = hashlib.sha256()
    paths = sorted(list((ROOT/'src').rglob('*.py')) + list((ROOT/'tools').glob('*.py')))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def gpu_state(gpu):
    output = subprocess.check_output([
        'nvidia-smi', '-i', str(gpu), '--query-gpu=uuid,memory.total,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    rows = list(csv.reader(io.StringIO(output)))
    if len(rows) != 1:
        raise RuntimeError('Select exactly one physical GPU index or UUID')
    uuid, total, used = [v.strip() for v in rows[0]]
    return uuid, float(total) / 1024, float(used) / 1024


def stop_process(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def run_monitored(command, env, log, gpu, budget, timeout):
    peak, samples, reason = 0., 0, None
    start = time.monotonic()
    with log.open('w') as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while proc.poll() is None:
                _, _, used = gpu_state(gpu)
                peak, samples = max(peak, used), samples + 1
                if used > budget:
                    reason = 'DEVICE_BUDGET_EXCEEDED'
                    break
                if time.monotonic() - start > timeout:
                    reason = 'TIMEOUT'
                    break
                time.sleep(.5)
        finally:
            stop_process(proc)
    return {'returncode': proc.returncode, 'monitor_reason': reason,
            'device_peak_sampled_gib': peak, 'device_memory_samples': samples,
            'wall_seconds': time.monotonic() - start}


def configure_batch(cfg, batch, world_size):
    if 'global_batch_size' not in cfg.training:
        raise ValueError('Adaptation requires an explicit upstream training.global_batch_size')
    plan = batch_plan(int(cfg.training.global_batch_size), world_size, batch)
    cfg.training.micro_batch_size = plan.micro_batch_size
    cfg.training.gradient_accumulation_steps = plan.gradient_accumulation_steps
    cfg.training.expected_world_size = plan.world_size
    return plan


def make_config(original, batch, phase, steps, workers, disc_checkpoint, world_size=8):
    # Import only configuration support in the supervisor: no CUDA context is retained.
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(OmegaConf.to_container(original, resolve=True))
    configure_batch(cfg, batch, world_size)
    cfg.training.epochs = steps  # Also works with tiny fixtures (one batch per epoch).
    cfg.training.num_workers = workers
    cfg.training.checkpoint_interval = 0
    cfg.training.sample_every = 0
    cfg.training.log_interval = 1
    cfg.pop('eval', None)
    cfg.gan.loss.lpips_start = 0
    cfg.gan.loss.disc_upd_start = 0 if phase in ('disc', 'gan') else steps + 1
    cfg.gan.loss.disc_start = 0 if phase == 'gan' else steps + 1
    if disc_checkpoint:
        cfg.gan.disc.arch.dino_ckpt_path = str(Path(disc_checkpoint).resolve())
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', default='0', help='Physical index or GPU UUID, not a masked index')
    p.add_argument('--budget-gib', type=float, default=80)
    p.add_argument('--reserve-gib', type=float, default=4)
    p.add_argument('--world-size', type=int, default=8, help='Target total GPU/rank count (default: 8); probe still uses one GPU')
    p.add_argument('--batches', type=int, nargs='+', help='Candidate per-GPU microbatches; each must divide global batch / world size')
    p.add_argument('--steps', type=int, default=8)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--timeout', type=float, default=1200, help='Seconds per phase, including compilation')
    p.add_argument('--data-path', required=True)
    p.add_argument('--disc-checkpoint')
    p.add_argument('--config', default='configs/stage1/training/DINOv2-B_decXL.yaml')
    p.add_argument('--preset', default='.harness/config.json')
    p.add_argument('--attention-backend', choices=('cudnn', 'flash'), default='cudnn')
    p.add_argument('--profile', action='store_true', help='Diagnostic trace; timings include profiler overhead')
    p.add_argument('--out', type=Path, required=True, help='New output directory, never an existing run')
    args = p.parse_args()
    from omegaconf import OmegaConf
    os.chdir(ROOT)
    original = OmegaConf.load(args.config)
    if 'global_batch_size' not in original.training:
        p.error('Source config must explicitly set training.global_batch_size')
    try:
        local_batch = batch_plan(int(original.training.global_batch_size), args.world_size).local_batch_size
        if args.batches is None:
            args.batches = [b for b in (64, 32, 16, 8, 4, 2, 1) if local_batch % b == 0]
        for candidate in args.batches:
            batch_plan(int(original.training.global_batch_size), args.world_size, candidate)
    except ValueError as error:
        p.error(str(error))
    if args.steps <= args.warmup or args.warmup < 1 or min(args.batches) < 1:
        p.error('Require steps > warmup >= 1 and positive batches')
    if args.timeout <= 0 or args.workers < 0:
        p.error('Require positive timeout and nonnegative workers')
    os.chdir(ROOT)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    gpu, total, used = gpu_state(args.gpu)
    budget = min(args.budget_gib, total)
    if not 0 < args.reserve_gib < budget:
        p.error('Require 0 < reserve-gib < effective budget')
    if used > 1:
        raise RuntimeError('GPU is busy (>1 GiB); choose an idle GPU, do not kill other jobs')
    env = os.environ.copy()
    # No inherited experimental RAE flags can silently change the named preset.
    env = {k: v for k, v in env.items() if not k.startswith('RAE_') or k == 'RAE_LPIPS_CHECKPOINT'}
    env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               WANDB_MODE='disabled', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4')
    doctor = [sys.executable, str(ROOT/'tools/check_environment.py'), '--config', args.config,
              '--data-path', str(Path(args.data_path).resolve()), '--min-images', str(local_batch),
              '--output', str(args.out/'environment.json')]
    if args.disc_checkpoint:
        doctor += ['--disc-checkpoint', args.disc_checkpoint]
    subprocess.run(doctor, cwd=ROOT, env=env, check=True, timeout=args.timeout)
    environment = json.loads((args.out/'environment.json').read_text())
    if environment['attention_backends'].get(args.attention_backend) is not True:
        raise RuntimeError('Selected attention backend failed preflight; inspect environment.json')
    preset = json.loads(Path(args.preset).read_text())['environment']
    env.update(preset)
    env['RAE_LOCAL_ATTN_BACKEND'] = args.attention_backend
    (args.out/'preset.env.sh').write_text(''.join(
        'export ' + k + '=' + shlex.quote(v) + '\n' for k, v in
        (preset | {'RAE_LOCAL_ATTN_BACKEND': args.attention_backend}).items()))
    (args.out/'environment-preset.json').write_text(json.dumps({'environment': preset | {
        'RAE_LOCAL_ATTN_BACKEND': args.attention_backend}}, indent=2)+'\n')
    summary = {'kind': 'capacity_smoke_only', 'budget_gib': budget, 'reserve_gib': args.reserve_gib,
               'hardware_total_gib': total, 'hardware_name': environment['gpu']['name'],
               'gpu_uuid': gpu, 'source_sha256': source_fingerprint(), 'selected_batch': None, 'attempts': [], 'complete': False,
               'profile_enabled': args.profile, 'target_world_size': args.world_size,
               'global_batch_size': int(original.training.global_batch_size), 'batch_plan': None,
               'executed_world_size': 1,
               'notes': ['One target rank is simulated; generated training config preserves upstream global batch. Distributed capacity is not validated.',
                         'Forced phases and short runs establish capacity, not convergence or numerical equivalence.',
                         'Allocator cap plus sampled device memory; not hardware isolation or an A100 benchmark.']}
    def save():
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    save()
    for batch in args.batches:
        passed = True
        for phase in PHASES:
            if gpu_state(gpu)[2] > 1:
                raise RuntimeError('GPU became busy before phase start')
            folder = args.out/f'b{batch}-{phase}'
            folder.mkdir()
            cfg = make_config(original, batch, phase, args.steps, args.workers, args.disc_checkpoint, args.world_size)
            OmegaConf.save(cfg, folder/'probe.yaml')
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
                       'src/train_stage1.py', '--config', str(folder/'probe.yaml'),
                       '--data-path', str(Path(args.data_path).resolve()), '--results-dir', str(folder/'training'),
                       '--precision', 'fp32', '--probe-steps', str(args.steps), '--probe-warmup', str(args.warmup),
                       '--probe-output', str(folder/'metrics.json'), '--memory-budget-gib', str(budget),
                       '--memory-reserve-gib', str(args.reserve_gib),
                       '--probe-target-world-size', str(args.world_size)]
            if args.profile:
                command.append('--probe-profile')
            env['EXPERIMENT_NAME'] = 'capacity-probe'
            print(f'PROBE batch={batch} phase={phase} budget={budget:.2f} GiB', flush=True)
            attempt = {'batch': batch, 'phase': phase,
                       'batch_plan': batch_plan(int(original.training.global_batch_size), args.world_size, batch).to_dict()}
            attempt.update(run_monitored(command, env, folder/'run.log', gpu, budget, args.timeout))
            metrics = json.loads((folder/'metrics.json').read_text()) if (folder/'metrics.json').exists() else {}
            attempt['metrics'] = metrics
            text = (folder/'run.log').read_text(errors='replace')
            if attempt['returncode'] == 0 and metrics.get('complete') and not attempt['monitor_reason']:
                attempt['status'] = 'PASS'
            elif attempt['monitor_reason'] == 'DEVICE_BUDGET_EXCEEDED' or 'CUDA out of memory' in text or 'OutOfMemoryError' in text:
                attempt['status'] = 'OOM'
            else:
                attempt['status'] = 'ERROR'
            summary['attempts'].append(attempt)
            save()
            print(f"RESULT batch={batch} phase={phase} status={attempt['status']} peak={attempt['device_peak_sampled_gib']:.2f} GiB", flush=True)
            # Context teardown is asynchronous; wait only for our own completed child to release memory.
            for _ in range(20):
                if gpu_state(gpu)[2] <= 1:
                    break
                time.sleep(.5)
            if attempt['status'] == 'ERROR':
                raise RuntimeError(f'Non-OOM failure: inspect {folder / "run.log"}; no silent batch/backend fallback')
            if attempt['status'] != 'PASS':
                passed = False
                break
        if passed:
            summary['selected_batch'] = batch
            training = OmegaConf.create(OmegaConf.to_container(original, resolve=True))
            summary['batch_plan'] = configure_batch(training, batch, args.world_size).to_dict()
            training.training.num_workers = args.workers
            if args.disc_checkpoint:
                training.gan.disc.arch.dino_ckpt_path = str(Path(args.disc_checkpoint).resolve())
            OmegaConf.save(training, args.out/'training.yaml')
            (args.out/'launch.sh').write_text(
                '#!/usr/bin/env bash\nset -euo pipefail\n'
                '# Single-node example; for multi-node configure torchrun rendezvous.\n'
                + 'source ' + shlex.quote(str(args.out/'preset.env.sh')) + '\n'
                + 'torchrun --standalone --nproc_per_node=' + str(args.world_size)
                + ' src/train_stage1.py --config ' + shlex.quote(str(args.out/'training.yaml'))
                + ' --data-path ' + shlex.quote(str(Path(args.data_path).resolve()))
                + ' --precision fp32 "$@"\n')
            break
    if source_fingerprint() != summary['source_sha256']:
        raise RuntimeError('Source changed during adaptation; result is invalid')
    summary['complete'] = True
    save()
    print(json.dumps({'selected_batch': summary['selected_batch'], 'summary': str(args.out/'summary.json')}))
    return 0 if summary['selected_batch'] is not None else 3


if __name__ == '__main__':
    raise SystemExit(main())
