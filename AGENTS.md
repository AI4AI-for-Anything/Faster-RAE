# Working on Faster-RAE

This repository ships the model/training code and a small deployment-adaptation workflow. The research harness, numerical evaluator, optimization-loop skills, experiment ledger, datasets, and checkpoints are separate artifacts. Do not copy a machine's research workspace into this repository.

For a new GPU or memory budget, read [skills/adapt-rae-gpu/SKILL.md](skills/adapt-rae-gpu/SKILL.md) and [docs/agent-environment.md](docs/agent-environment.md) before GPU work. Ask how many GPUs the user intends to train on when useful; if unspecified, use **8 total GPUs**. Default to an 80 GiB per-device budget and 4 GiB allocator headroom, capped to actual capacity. The single-GPU probe simulates one rank of that deployment.

Preserve the upstream `training.global_batch_size` (512 in DINOv2-B_decXL). Always satisfy `global_batch_size = world_size × micro_batch_size × gradient_accumulation_steps`. Reduce the microbatch on OOM and increase accumulation; never reduce global batch to fit VRAM. Use `--world-size` for the target GPU count. Generated training configs must reject a different actual world size rather than silently changing the plan. Do not override the upstream LR or training schedule during adaptation.

Use an isolated environment and a fresh output directory. Check that the selected GPU is idle, honor scheduler/locking rules, and never kill another user's job. Reuse existing read-only weights and datasets. Resolve missing assets before probing; do not download assets inside an experiment. Keep machine-specific paths and traces in ignored output directories.

Run the real trainer through `tools/adapt_gpu.py`. Each candidate must complete reconstruction+LPIPS, discriminator-update, and adversarial-generator phases with its planned accumulation. Only OOM triggers automatic microbatch reduction. Retain failed attempts. Do not change model size, resolution, loss, or whole-step precision to pass.

G updates once per effective batch; D then updates using fresh reconstructions for that batch. EMA and G scheduler advance once per G update; D scheduler advances once per D update. Accumulated GAN adaptive weighting aggregates rank-local last-layer gradients and replays microbatches. DDP synchronizes G and D gradients. The default BatchNormLocal uses per-sample groups without running statistics; spectral-normalization buffers advance per microbatch and random crops can differ; equal effective batch does not establish numerical equivalence or convergence. A one-rank capacity PASS does not validate full distributed memory/communication, checkpointing, or evaluation.

The original snapshot hashes remain in `docs/source-snapshot.json`. Validate changes with CPU tests and an appropriate bounded GPU probe; do not start full training unless requested. Publish only code, instructions, and selected aggregate results. Never commit credentials, weights, datasets, raw inventories, or caches. Local/user data-governance instructions always apply.
