"""Bitwise check: tpu_smoke/rae_diffaug.py vs upstream DiffAug.aug (CPU).

Locks: with the same torch RNG state, the patched aug returns tensors
torch.equal to upstream for the recipe policy (prob 1.0, cutout 0.0) and for a
cutout policy, at B>1 so per-sample offsets differ. Mutation: shifting the row
index by one must be caught.

Run from the repository root (CPU, no TPU needed):  python -m tpu_smoke.test_rae_diffaug
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "src"))
sys.path.insert(0, os.path.dirname(_HERE))

from disc import diffaug as mod                     # noqa: E402
from tpu_smoke import rae_diffaug                   # noqa: E402


def run(aug_fn, policy, x, seed):
    torch.manual_seed(seed)
    return aug_fn(mod.DiffAug(**policy), x)


def main():
    up = mod.DiffAug.aug
    rae_diffaug.install()
    new = mod.DiffAug.aug
    gen = torch.Generator().manual_seed(1)
    x = torch.rand(6, 3, 64, 48, generator=gen) * 2 - 1
    ok = True
    for policy in (dict(prob=1.0, cutout=0.0), dict(prob=1.0, cutout=0.2)):
        for seed in range(8):
            a, b = run(up, policy, x, seed), run(new, policy, x, seed)
            same = torch.equal(a, b)
            ok &= same
            if not same:
                print("MISMATCH", policy, seed, (a - b).abs().max().item())
    # The translation must actually move pixels, or the test proves nothing.
    torch.manual_seed(0)
    moved = not torch.equal(run(new, dict(prob=1.0, cutout=0.0), x, 0), x)
    ok &= moved
    # Mutation: off-by-one rows.
    orig = rae_diffaug._translate_rows_cols
    def bad(p, gh, gw):
        return orig(p, (gh + 1).clamp(max=p.shape[2] - 1), gw)
    mod_ns = mod.DiffAug.aug.__globals__
    mod_ns["_translate_rows_cols"] = bad
    caught = not torch.equal(run(up, dict(prob=1.0, cutout=0.0), x, 3),
                             run(mod.DiffAug.aug, dict(prob=1.0, cutout=0.0), x, 3))
    mod_ns["_translate_rows_cols"] = orig
    print(f"bitwise={ok} translation_moves_pixels={moved} mutation_caught={caught}")
    print("PASS" if ok and caught else "FAIL")
    sys.exit(0 if ok and caught else 1)


if __name__ == "__main__":
    main()
