"""RAE Stage-1 full recipe step on TPU (torch_xla SPMD), one phase per process.

AGENTS.md requires every candidate to run all three phases of
src/train_stage1.py at global batch 512 (configs/stage1/training/DINOv2-B_decXL.yaml):
  phase 1  epochs 0-5   reconstruction (L1) + LPIPS                       (6 epochs)
  phase 2  epochs 6-7   phase 1 + discriminator update                    (2 epochs)
  phase 3  epochs 8-15  phase 2 + adversarial generator, adaptive weight  (8 epochs)
README's "6:2:8 phase weighting" is the epoch split above; report
(6*p1 + 2*p2 + 8*p3) / 16 as the recipe step time.

The step mirrors train_stage1.py:380-470 and imports the upstream pieces
unchanged: RAE, build_discriminator, LPIPS, hinge/vanilla losses, DiffAug,
generator_backward (adaptive weight and microbatch replay), update_ema,
build_optimizer/build_scheduler. Order per step as upstream: G losses with D
frozen (requires_grad off, eval) -> G step -> scheduler -> EMA -> [D update on
fresh no-grad reconstructions, quantised to 8 bit, D scheduler] -> back to train().

KNOWN DEVIATION (not bitwise vs the 8-GPU run): upstream's adaptive GAN weight is
rank-local (accumulation.py docstring), each of 8 ranks computing it from its own
64 images before DDP averages gradients. One SPMD program has no ranks, so the
weight here is computed from last-layer gradients of the whole effective batch.
Per-chip work is the same two autograd.grad calls, so timing is representative;
numerics are not. DiffAug crops and spectral-norm power iterations also draw from
one RNG instead of eight, which AGENTS.md already lists as differing across plans.

Not covered: data loading (synthetic images, fixed per process), checkpointing,
eval, sampling. Assets (see tpu_smoke/README.md): RAE_LPIPS_CHECKPOINT (LPIPS vgg.pth),
TORCH_HOME (torchvision vgg16), HF_HOME (DINOv2-with-registers base), --disc-ckpt.

Run from the repository root (whole slice, same command on every worker):
  python -m tpu_smoke.rae_recipe_step --phase 3 --batch 512 --recipe-bf16 --disc-ckpt PATH [--no-fallbacks]
"""
import argparse
import json
import os
import statistics
import sys
import time
from copy import deepcopy

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

import torch_xla
import torch_xla.debug.metrics as tmetrics
from torch_xla.core import xla_model as xm

STEPS_PER_EPOCH = 1281167 // 512   # ImageNet-1k train, drop_last loader at global 512
PHASE_START_EPOCH = {1: 0, 2: 6, 3: 8}


def uncached():
    names = tmetrics.counter_names()
    return tmetrics.counter_value("UncachedCompile") if "UncachedCompile" in names else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=(1, 2, 3), required=True)
    ap.add_argument("--batch", type=int, default=512, help="GLOBAL batch (recipe: 512)")
    ap.add_argument("--micro", type=int, default=None,
                    help="GLOBAL microbatch; accumulation = batch / micro (default: no accumulation)")
    ap.add_argument("--image", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--traj", type=int, default=0,
                    help="statistical-gate mode: N steps from init, a FRESH synthetic batch per step "
                         "drawn from (seed, step), print every step's losses as TRAJ json, no timing")
    ap.add_argument("--data-pool", default=None,
                    help="traj mode: uint8 [N, 384, 384, 3] .npy from build_image_pool.py (real images); "
                         "each step draws batch images and RandomCrop(image) offsets from (seed, step)")
    ap.add_argument("--seed", type=int, default=0, help="model init + XLA RNG + data seed (traj mode)")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="traj mode with --data-pool: after step eval-start+k*eval-every, the generator's "
                         "reconstruction loss (L1 + perceptual) on a FIXED held-out set, in eval mode "
                         "(no noise_tau). Training loss on fresh batches could not tell lr x1.05 apart "
                         "(seed-to-seed data-sampling variance, 11-07); this endpoint removes that term")
    ap.add_argument("--eval-start", type=int, default=10)
    ap.add_argument("--eval-n", type=int, default=512, help="fixed eval set: the pool's last N images, center crop")
    ap.add_argument("--sched-epoch", type=int, default=None,
                    help="start the LR schedules at this epoch instead of the phase's (gate only). "
                         "Phase 1 starts in the warmup-from-zero, so 60 gate steps ran at ~2%% of "
                         "the peak LR and lr x1.02 could not be told apart (11-07)")
    ap.add_argument("--lr-scale", type=float, default=1.0,
                    help="multiply G and D learning rates; sensitivity control for the statistical gate")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--disc-ckpt", required=True,
                    help="DINO ViT-S/8 discriminator weights (dino_vit_small_patch8_224.pth)")
    ap.add_argument("--no-fallbacks", action="store_true",
                    help="tpu_smoke/rae_fallbacks.py (bitwise, test_rae_fallbacks.py)")
    ap.add_argument("--diffaug-rows", action="store_true",
                    help="tpu_smoke/rae_diffaug.py: DiffAug translation as two batch-aligned "
                         "gathers instead of one advanced-index gather (bitwise, "
                         "test_rae_diffaug.py); removes ~956 whole-batch all-gathers per step")
    ap.add_argument("--dynamo", action="store_true",
                    help="torch.compile(rae.decoder, backend='openxla'): the step is host-bound "
                         "after the fallback/DiffAug fixes (64 chips: trace 243-429 ms vs device "
                         "56-101 ms, 11-07). Needs TORCHDYNAMO_DISABLE unset")
    ap.add_argument("--dynamo-all", action="store_true",
                    help="--dynamo plus the frozen DINOv2 encoder and LPIPS forwards; keeps "
                         "RAE.encoder_mean/std on the device (plain CPU attributes moved by "
                         ".to(device) on every encode/decode in upstream, same values)")
    ap.add_argument("--adamw-single", action="store_true",
                    help="AdamW single-tensor path (foreach=False) for G and D: on XLA the foreach "
                         "ops decompose through torch._refs with per-tensor meta checks "
                         "(cProfile 11-07: _foreach_addcdiv_/addcmul_/lerp_ ~90 ms/step each)")
    ap.add_argument("--compile-opt", action="store_true",
                    help="torch.compile(opt.step / dopt.step, backend='openxla'): the AdamW foreach "
                         "decomposition is the largest Python cost left (cProfile 11-07)")
    ap.add_argument("--gc-freeze", action="store_true",
                    help="after warmup: gc.collect(); gc.freeze(); gc.set_threshold(50000, 20, 20). "
                         "dynamo graphs allocate many Python objects; one phase-2 window paid "
                         "41.9 ms/step of GC (11-07). Values unchanged")
    ap.add_argument("--host-profile", action="store_true",
                    help="cProfile 3 steady steps of the host trace (one_step + sync) and print the top entries")
    ap.add_argument("--recipe-bf16", action="store_true",
                    help="scripts/fd7800d0633a.env.sh precision switches: decoder AND "
                         "discriminator attn/mlp/proj bf16; RAE_LOCAL_ATTN_BACKEND=math "
                         "(cudnn has no XLA meaning). Layout/compile switches are CUDA-only.")
    args = ap.parse_args()
    if args.recipe_bf16:
        for k in ("RAE_DECODER_ATTN_BF16", "RAE_DECODER_MLP_BF16", "RAE_DECODER_PROJ_BF16",
                  "RAE_DISC_ATTN_BF16", "RAE_DISC_MLP_BF16", "RAE_DISC_PROJ_BF16"):
            os.environ[k] = "1"
        os.environ["RAE_LOCAL_ATTN_BACKEND"] = "math"
    # Upstream default RAE_ADAMW_IMPL=fused is CUDA's fused kernel and refuses xla
    # tensors (11-06); foreach is the other upstream choice, same AdamW update.
    os.environ.setdefault("RAE_ADAMW_IMPL", "foreach")
    if not os.path.isfile(args.disc_ckpt):
        raise FileNotFoundError(f"--disc-ckpt {args.disc_ckpt} does not exist")

    import numpy as np
    import torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs
    xr.use_spmd()
    n = xr.global_runtime_device_count()
    mesh = xs.Mesh(np.arange(n), (n,), ("data",))

    from omegaconf import OmegaConf
    from stage1 import RAE
    from disc import LPIPS, build_discriminator, hinge_d_loss, vanilla_g_loss
    from utils.accumulation import generator_backward
    from utils.optim_utils import build_optimizer, build_scheduler
    from utils.train_utils import update_ema

    cfg = OmegaConf.to_container(OmegaConf.load(os.path.join(
        _ROOT, "configs", "stage1", "training", "DINOv2-B_decXL.yaml")), resolve=True)
    tcfg, gcfg = cfg["training"], cfg["gan"]
    lcfg = gcfg["loss"]
    dcfg = deepcopy(gcfg["disc"]); dcfg["arch"]["dino_ckpt_path"] = args.disc_ckpt
    if args.batch != tcfg["global_batch_size"]:
        print(f"WARNING batch {args.batch} != recipe global_batch_size {tcfg['global_batch_size']}", flush=True)
    micro = args.micro or args.batch
    if args.batch % micro or micro % n:
        raise ValueError(f"micro {micro} must divide batch {args.batch} and be a multiple of {n} chips")
    accum = args.batch // micro
    epoch = PHASE_START_EPOCH[args.phase]
    use_lpips = epoch >= lcfg["lpips_start"] and lcfg["perceptual_weight"] > 0
    train_disc = epoch >= lcfg["disc_upd_start"] and lcfg["disc_weight"] > 0
    use_gan = epoch >= lcfg["disc_start"] and lcfg["disc_weight"] > 0
    if use_gan and accum > 1:
        # generator_backward replays each microbatch under the saved CPU/CUDA RNG
        # state; the encoder noise (noise_tau) and DiffAug draw from the XLA RNG,
        # which it does not save, so the replay would see different noise.
        raise NotImplementedError("phase 3 with accumulation: XLA RNG is not replayed")

    dev = torch_xla.device()
    torch.manual_seed(args.seed)
    xm.set_rng_state(args.seed)
    p = cfg["stage_1"]["params"]
    rae = RAE(encoder_cls=p["encoder_cls"], encoder_config_path=p["encoder_config_path"],
              encoder_input_size=p["encoder_input_size"], encoder_params=p["encoder_params"],
              decoder_config_path=os.path.join(_ROOT, p["decoder_config_path"]),
              noise_tau=p["noise_tau"], reshape_to_2d=p["reshape_to_2d"]).to(dev)
    rae.encoder.eval(); rae.decoder.train()
    ema = deepcopy(rae).to(dev).eval(); ema.requires_grad_(False)
    rae.encoder.requires_grad_(False); rae.decoder.requires_grad_(True)
    if args.diffaug_rows:
        from tpu_smoke import rae_diffaug
        rae_diffaug.install()
    disc, disc_aug = build_discriminator(dcfg, dev)
    disc.eval()
    lpips = LPIPS().to(dev).eval()
    disc_params = [q for q in disc.parameters() if q.requires_grad]
    opt, _ = build_optimizer(rae.decoder.parameters(), tcfg)
    dopt, _ = build_optimizer(disc_params, dcfg)
    if args.adamw_single:
        for o in (opt, dopt):
            for g in o.param_groups:
                g["foreach"], g["fused"] = False, False
    if args.compile_opt:
        opt.step = torch.compile(opt.step, backend="openxla")
        dopt.step = torch.compile(dopt.step, backend="openxla")
    if args.lr_scale != 1.0:
        for o in (opt, dopt):
            for g in o.param_groups:
                g["lr"] *= args.lr_scale
                g["initial_lr"] = g.get("initial_lr", g["lr"] / args.lr_scale) * args.lr_scale
        tcfg = dict(tcfg, base_lr=float(tcfg["scheduler"]["base_lr"]) * args.lr_scale,
                    final_lr=float(tcfg["scheduler"]["final_lr"]) * args.lr_scale)
    sched, _ = build_scheduler(opt, STEPS_PER_EPOCH, tcfg)
    dsched, _ = build_scheduler(dopt, STEPS_PER_EPOCH, dcfg)
    # Start the schedules where this phase starts (host-side LambdaLR only).
    for _ in range((epoch if args.sched_epoch is None else args.sched_epoch) * STEPS_PER_EPOCH):
        sched.step(); dsched.step()
    last_layer = rae.decoder.decoder_pred.weight
    if args.dynamo_all:
        args.dynamo = True
        rae.encoder_mean, rae.encoder_std = rae.encoder_mean.to(dev), rae.encoder_std.to(dev)
        ema.encoder_mean, ema.encoder_std = ema.encoder_mean.to(dev), ema.encoder_std.to(dev)
        rae.encoder.forward = torch.compile(rae.encoder.forward, backend="openxla")
        lpips.forward = torch.compile(lpips.forward, backend="openxla")
    if args.dynamo:
        # Compile the trainable decoder only, as the recipe's
        # RAE_COMPILE_REGION=decoder_blocks; the frozen encoder stays eager.
        # Compile forward, not the module: torch.compile(module) renames params to
        # `_orig_mod.*` and update_ema's name lookup then fails (11-07 KeyError).
        # Grad-enabled (training) forward only. The D update re-runs the decoder
        # under no_grad in eval; dynamo compiled that as a separate graph holding
        # the param buffers, which AdamW's in-place (donated) update then deleted:
        # phase 2/3 aborted with "buffer with shape f32[1,257,1152] ... is deleted"
        # (decoder_pos_embed, 11-07). That call stays eager.
        _dec_eager = rae.decoder.forward
        _dec_comp = torch.compile(_dec_eager, backend="openxla")
        rae.decoder.forward = lambda *a, **k: (_dec_comp if torch.is_grad_enabled() else _dec_eager)(*a, **k)
    clip = tcfg.get("clip_grad")
    clip = float(clip) if clip else None
    dw, max_dw = float(lcfg["disc_weight"]), float(lcfg["max_d_weight"])
    pw = float(lcfg["perceptual_weight"])

    gen = torch.Generator().manual_seed(0)
    imgs_cpu = torch.rand(args.batch, 3, args.image, args.image, generator=gen)
    if args.no_fallbacks:
        from tpu_smoke.rae_fallbacks import cache_pos_encoding, host_resize
        cache_pos_encoding(rae.encoder.encoder.embeddings, 224, 224)
        cache_pos_encoding(ema.encoder.encoder.embeddings, 224, 224)

    def to_dev(t):
        t = t.to(dev)
        xs.mark_sharding(t, mesh, ("data", None, None, None))
        return t

    # Microbatches live on the device for the whole run (fixed synthetic batch):
    # the recipe's per-step H2D copy is a data-loader cost, out of scope here.
    micro_batches = [(to_dev(m), to_dev(host_resize(m, 224)) if args.no_fallbacks else None)
                     for m in imgs_cpu.split(micro)]

    def aug(t):
        # DiffAug's translation gathers with unsharded arange grids
        # (disc/diffaug.py:73-77); the gather output came out REPLICATED, so every
        # chip ran the DINO discriminator on all 512 images (f32[3072,785,785]
        # attention, 148 GiB/chip at compile, 11-06). Re-annotating the batch
        # axis is a sharding hint only; values are unchanged.
        t = disc_aug.aug(t)
        xs.mark_sharding(t, mesh, ("data", None, None, None))
        return t

    def recon_of(model, batch):
        x, x224 = batch
        return model.decode(model.encode(x224)) if x224 is not None else model(x)

    def generator_losses(model, batch):           # train_stage1.py:391-405, fp32 autocast off
        x = batch[0]
        recon = recon_of(rae, batch)
        recon_normed = recon * 2.0 - 1.0
        rec = (recon - x).abs().mean()
        perceptual = lpips(x * 2.0 - 1.0, recon_normed) if use_lpips else rec.new_zeros(())
        reconstruction = rec + pw * perceptual
        if use_gan:
            logits_fake, _ = disc(aug(recon_normed), None)
            adversarial = vanilla_g_loss(logits_fake)
        else:
            adversarial = reconstruction.new_zeros(())
        return {"recon": rec, "lpips": perceptual, "gan": adversarial, "recon_total": reconstruction}

    def one_step():
        opt.zero_grad(set_to_none=True)
        dopt.zero_grad(set_to_none=True)
        disc.eval()
        for q in disc_params:
            q.requires_grad_(False)
        metrics, _w = generator_backward(rae, micro_batches, generator_losses, last_layer,
                                         use_gan=use_gan, disc_weight=dw, max_d_weight=max_dw)
        for q in disc_params:
            q.requires_grad_(True)
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(rae.parameters(), clip)
        opt.step(); sched.step()
        update_ema(ema, rae, float(tcfg["ema_decay"]))
        d_loss_out = None
        if train_disc:                            # train_stage1.py:426-468, disc_updates=1
            rae.eval(); disc.train()
            for _ in range(int(lcfg.get("disc_updates", 1))):
                dopt.zero_grad(set_to_none=True)
                acc = 0
                for batch in micro_batches:
                    x = batch[0]
                    with torch.no_grad():
                        fake = recon_of(rae, batch) * 2.0 - 1.0
                    fake = fake.clamp(-1.0, 1.0)
                    fake = torch.round((fake + 1.0) * 127.5) / 127.5 - 1.0
                    lf, lr_ = disc(aug(fake), aug(x * 2.0 - 1.0))
                    d_loss = hinge_d_loss(lr_, lf)
                    (d_loss / accum).backward()
                    acc = acc + d_loss.detach() / accum
                dopt.step(); dsched.step()
                d_loss_out = acc
            disc.eval(); rae.train()
        return metrics["total"].detach(), d_loss_out

    if args.traj:
        traj = []
        pool = np.load(args.data_pool, mmap_mode="r") if args.data_pool else None
        evals, eval_batches = [], []
        if args.eval_every:
            if pool is None:
                raise SystemExit("--eval-every needs --data-pool")
            o = (pool.shape[1] - args.image) // 2
            ev = torch.from_numpy(np.ascontiguousarray(pool[-args.eval_n:, o:o + args.image, o:o + args.image]))
            ev = ev.permute(0, 3, 1, 2).float().div_(255)
            # Split like training (micro); eval mode is a separate graph anyway (no noise_tau).
            eval_batches = [(to_dev(m), to_dev(host_resize(m, 224)) if args.no_fallbacks else None)
                            for m in ev.split(micro)]

        def eval_loss():
            rae.eval()
            with torch.no_grad():
                acc = 0.0
                for batch in eval_batches:
                    x = batch[0]
                    recon = recon_of(rae, batch)
                    r = (recon - x).abs().mean()
                    if use_lpips:
                        r = r + pw * lpips(x * 2.0 - 1.0, recon * 2.0 - 1.0)
                    acc = acc + r / len(eval_batches)
                torch_xla.sync()
                out = float(acc)
            rae.train()
            return out
        for step in range(args.traj):
            g = torch.Generator().manual_seed(args.seed * 1000003 + step)
            if pool is not None:
                r = np.random.default_rng(args.seed * 1000003 + step)
                idx = r.integers(0, pool.shape[0], args.batch)
                off = r.integers(0, pool.shape[1] - args.image + 1, (args.batch, 2))
                crops = np.stack([pool[i, y:y + args.image, x:x + args.image] for i, (y, x) in zip(idx, off)])
                imgs = torch.from_numpy(crops).permute(0, 3, 1, 2).float().div_(255)   # ToTensor
            else:
                imgs = torch.rand(args.batch, 3, args.image, args.image, generator=g)
            micro_batches[:] = [(to_dev(m), to_dev(host_resize(m, 224)) if args.no_fallbacks else None)
                                for m in imgs.split(micro)]
            tot, dl = one_step()
            torch_xla.sync()
            traj.append([float(tot), None if dl is None else float(dl)])
            if eval_batches and (step + 1 - args.eval_start) > 0 and (step + 1 - args.eval_start) % args.eval_every == 0:
                evals.append(eval_loss())
        print("TRAJ " + json.dumps(dict(phase=args.phase, seed=args.seed, lr_scale=args.lr_scale,
                                        sched_epoch=args.sched_epoch, lr=opt.param_groups[0]["lr"], data_pool=args.data_pool,
                                        no_fallbacks=args.no_fallbacks, diffaug_rows=args.diffaug_rows,
                                        dynamo=args.dynamo, dynamo_all=args.dynamo_all,
                                        ema_foreach=os.environ.get("RAE_EMA_FOREACH"), eval=evals, traj=traj)), flush=True)
        return

    losses = []
    t0 = time.time()
    for _ in range(args.warmup):
        tot, dl = one_step()
        torch_xla.sync(); xm.wait_device_ops()
        losses.append(float(tot))
    warm_s = time.time() - t0
    if args.gc_freeze:
        import gc
        gc.collect(); gc.freeze(); gc.set_threshold(50000, 20, 20)

    from tpu_smoke.step_counters import Probe
    wins = []
    for wi in range(args.windows):
        u0 = uncached()
        probe = Probe(); probe.start()
        t_a = time.perf_counter()
        for _ in range(args.window):
            tot, dl = one_step()
            torch_xla.sync()
        xm.wait_device_ops()
        ms = round(1000 * (time.perf_counter() - t_a) / args.window, 2)
        pr = probe.report(args.window)
        u1 = uncached()
        wins.append(dict(ms=ms, tainted=bool(u1 - u0) if None not in (u0, u1) else None, **pr))
    if args.host_profile:
        import cProfile, io, pstats
        pr = cProfile.Profile(); pr.enable()
        for _ in range(3):
            one_step(); torch_xla.sync()
        pr.disable(); xm.wait_device_ops()
        buf = io.StringIO(); pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(25)
        print("HOSTPROF_TOTTIME\n" + buf.getvalue(), flush=True)
        buf = io.StringIO(); pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(30)
        print("HOSTPROF_CUM\n" + buf.getvalue(), flush=True)
    # One blocking window: each step from an idle device, split into host trace
    # (one_step), submit (sync) and device (wait_device_ops), to tell whether the
    # pipelined time above is host- or device-bound.
    tr, sb, dv = [], [], []
    for _ in range(args.window):
        a = time.perf_counter(); tot, dl = one_step(); b = time.perf_counter()
        torch_xla.sync(); c = time.perf_counter(); xm.wait_device_ops(); d = time.perf_counter()
        tr.append(1000 * (b - a)); sb.append(1000 * (c - b)); dv.append(1000 * (d - c))
    blocking = dict(trace_ms=round(statistics.median(tr), 1), submit_ms=round(statistics.median(sb), 1),
                    device_ms=round(statistics.median(dv), 1))
    last = [float(tot), None if dl is None else float(dl)]
    ms = [w["ms"] for w in wins]
    print("SUMMARY " + json.dumps(dict(
        project="RAE-recipe", phase=args.phase, use_lpips=use_lpips, train_disc=train_disc,
        use_gan=use_gan, batch=args.batch, micro=micro, accum=accum, chips=n,
        no_fallbacks=args.no_fallbacks, diffaug_rows=args.diffaug_rows, dynamo=args.dynamo, dynamo_all=args.dynamo_all, adamw_single=args.adamw_single, compile_opt=args.compile_opt, gc_freeze=args.gc_freeze, ema_foreach=os.environ.get('RAE_EMA_FOREACH'), recipe_bf16=args.recipe_bf16,
        lr=opt.param_groups[0]["lr"], warmup_s=round(warm_s, 1),
        warmup_losses=[round(v, 4) for v in losses],
        pipelined_ms=statistics.median(ms), windows_ms=ms, blocking=blocking,
        spread=round((max(ms) - min(ms)) / statistics.median(ms), 3),
        tainted=[w["tainted"] for w in wins],
        counters_last_window={k: v for k, v in wins[-1].items() if k not in ("ms", "tainted")},
        images_per_s=round(args.batch / (statistics.median(ms) / 1000), 1),
        last_total_dloss=last)), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAILED " + type(exc).__name__ + ": " + str(exc)[:3000], flush=True)
        raise
