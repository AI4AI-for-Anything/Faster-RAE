"""Build a fixed uint8 image pool from Imagenette (public ImageNet subset) for RAE traj gates.

Takes the first N training images in sorted path order, applies the first half of
train_stage1.py's transform (Resize(384, BICUBIC) on the short side), then a
384x384 center crop so the pool is one array; RandomCrop(256) happens per step in
rae_recipe_step.py --data-pool. The center crop drops a strip of the long side,
which upstream's RandomCrop could reach; that is the only difference.

Usage: python -m tpu_smoke.build_image_pool SRC_DIR OUT.npy [--n 4096]
"""
import argparse
import os

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()
    paths = sorted(os.path.join(r, f) for r, _, fs in os.walk(args.src) for f in fs
                   if f.lower().endswith((".jpeg", ".jpg", ".png")))
    if len(paths) < args.n:
        raise SystemExit(f"only {len(paths)} images under {args.src}, need {args.n}")
    s = args.size
    pool = np.empty((args.n, s, s, 3), dtype=np.uint8)
    for i, p in enumerate(paths[:args.n]):
        im = Image.open(p).convert("RGB")
        w, h = im.size
        if w < h:
            im = im.resize((s, round(h * s / w)), Image.BICUBIC)
        else:
            im = im.resize((round(w * s / h), s), Image.BICUBIC)
        w, h = im.size
        l, t = (w - s) // 2, (h - s) // 2
        pool[i] = np.asarray(im.crop((l, t, l + s, t + s)))
    np.save(args.out, pool)
    print(f"pool {pool.shape} from {len(paths)} images, mean {pool.mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
