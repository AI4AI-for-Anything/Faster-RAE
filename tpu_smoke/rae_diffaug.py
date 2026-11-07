"""DiffAug translation without the advanced-indexing gather, same values.

src/disc/diffaug.py:77 translates every image by a per-sample
offset with

    BCHW = bchw_pad.permute(0, 2, 3, 1).contiguous()[grid_B, grid_h, grid_w].permute(0, 3, 1, 2).contiguous()

On torch_xla SPMD that index lowers to a gather the partitioner cannot keep
batch-local: a phase-2 HLO dump (4 chips, batch 32) held 956 all-gathers of
the whole padded batch f32[32,258,258,3], and the step cost ~4.7 s whatever
the batch (4 chips/32 images 4736 ms, 64 chips/512 images 4869 ms).

The indices have structure the generic gather ignores: grid_h[b, i, j] depends
only on (b, i) and grid_w[b, i, j] only on (b, j) (meshgrid 'ij' plus a [B,1,1]
offset, then clamp). So the same result is two batch-aligned torch.gather calls,
rows then columns. Pure data movement: bitwise equal, checked in
tpu_smoke/test_rae_diffaug.py against upstream DiffAug.aug under the same RNG.

install() rewrites ONLY that line of DiffAug.aug from outside; the upstream
file is not touched, and install()
refuses to run unless the line occurs exactly once.
"""
import inspect
import textwrap

import torch

OLD = ("BCHW = bchw_pad.permute(0, 2, 3, 1).contiguous()[grid_B, grid_h, grid_w]"
       ".permute(0, 3, 1, 2).contiguous()")
NEW = "BCHW = _translate_rows_cols(bchw_pad, grid_h, grid_w)"


def _translate_rows_cols(bchw_pad, grid_h, grid_w):
    """== bchw_pad.permute(0,2,3,1)[arange(B)[:,None,None], grid_h, grid_w].permute(0,3,1,2)."""
    B, C = bchw_pad.shape[:2]
    H, W = grid_h.shape[1:]
    ih = grid_h[:, :, 0]                                    # [B, H], row index per (b, i)
    iw = grid_w[:, 0, :]                                    # [B, W], col index per (b, j)
    rows = torch.gather(bchw_pad, 2, ih[:, None, :, None].expand(B, C, H, bchw_pad.shape[3]))
    return torch.gather(rows, 3, iw[:, None, None, :].expand(B, C, H, W)).contiguous()


def install():
    from disc import diffaug as mod
    src = textwrap.dedent(inspect.getsource(mod.DiffAug.aug))
    if src.count(OLD) != 1:
        raise RuntimeError(f"expected the translation gather exactly once in DiffAug.aug, found {src.count(OLD)}")
    ns = dict(vars(mod))
    ns["_translate_rows_cols"] = _translate_rows_cols
    exec(compile(src.replace(OLD, NEW), inspect.getsourcefile(mod.DiffAug.aug), "exec"), ns)
    mod.DiffAug.aug = ns["aug"]
