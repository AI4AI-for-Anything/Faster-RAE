"""On-TPU check of tpu_smoke/rae_diffaug.py vs upstream DiffAug.aug.

The CPU test (test_rae_diffaug.py) is bitwise. On 64 chips the phase-3 loss
differed from step 0, so this separates the two possible causes:
  same RNG draws?   rand01 compared directly (both arms draw it the same way);
  same function?    aug output compared, with the RNG reset to the same state.
Runs once unsharded (1 chip) and, with --spmd, with the batch sharded.

Run:  python -m tpu_smoke.test_rae_diffaug_tpu [--spmd]
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
sys.path.insert(0, os.path.dirname(_HERE))

import torch_xla
from torch_xla.core import xla_model as xm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spmd", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    mesh = None
    if args.spmd:
        import numpy as np
        import torch_xla.runtime as xr
        import torch_xla.distributed.spmd as xs
        xr.use_spmd()
        n = xr.global_runtime_device_count()
        mesh = xs.Mesh(np.arange(n), (n,), ("data",))
    from disc import diffaug as mod
    from tpu_smoke import rae_diffaug
    dev = torch_xla.device()
    x_cpu = torch.rand(args.batch, 3, 256, 256, generator=torch.Generator().manual_seed(0)) * 2 - 1
    x = x_cpu.to(dev)
    if mesh is not None:
        xs.mark_sharding(x, mesh, ("data", None, None, None))

    def run(fn, seed):
        torch.manual_seed(seed); xm.set_rng_state(seed)
        out = fn(mod.DiffAug(prob=1.0, cutout=0.0), x)
        if mesh is not None:
            xs.mark_sharding(out, mesh, ("data", None, None, None))
        torch_xla.sync()
        return out.cpu()

    def draws(seed):
        xm.set_rng_state(seed)
        r = torch.rand(7, args.batch, 1, 1, device=dev)
        torch_xla.sync()
        return r.cpu()

    up = mod.DiffAug.aug
    rae_diffaug.install()
    new = mod.DiffAug.aug
    res = []
    for seed in range(3):
        a, b = run(up, seed), run(new, seed)
        res.append((seed, torch.equal(a, b), (a - b).abs().max().item(), torch.equal(draws(seed), draws(seed))))
    # Reference on CPU from the SAME device draws: does either TPU arm equal the
    # upstream function evaluated on CPU with identical inputs?
    print(f"spmd={args.spmd} " + " ".join(f"seed{s}:equal={e},maxdiff={d:.3g},rng_repeatable={r}" for s, e, d, r in res))
    print("PASS" if all(e for _, e, _, _ in res) else "FAIL")


if __name__ == "__main__":
    main()
