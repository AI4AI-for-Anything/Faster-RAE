"""Remove RAE's two per-step CPU fallbacks on XLA, with bitwise-identical results.

A 4-chip diagnostic run (global 256) counted, per step, one `aten::_upsample_bicubic2d_aa`
and one `aten::upsample_bicubic2d` falling back to the CPU and TWO executions
instead of one: the host had to fetch device data mid-step, which serialised
tracing and device work (pipelined 631 ms vs max(host 439, device 305)).

  1. DINOv2 position encoding. The checkpoint stores 37x37 positions (518 input);
     RAE feeds 224, so transformers' interpolate_pos_encoding bicubic-resizes the
     FROZEN position table to 16x16 on every forward. Input size is fixed and the
     table is frozen, so the result is a constant: computed once, on the CPU, by
     the SAME HF function, and returned from cache afterwards. Identical bytes to
     what the CPU fallback produced every step.
  2. Input resize 256 -> 224 (src/stage1/rae.py:84) inside encode().
     It acts on the images, not on parameters, so it moves to the host data side:
     the batch is resized once on the CPU with the same F.interpolate call, and
     encode() then sees 224 input and skips its own resize. The loss is still
     taken against the 256 originals. Same CPU op, same inputs -> same bytes.

Both are installed from outside (monkeypatching the instance), upstream files are
unchanged. Equivalence: tpu_smoke/test_rae_fallbacks.py (CPU, torch.equal).
"""
import torch
import torch.nn.functional as F


def cache_pos_encoding(encoder_embeddings, height, width):
    """Freeze HF Dinov2WithRegistersEmbeddings.interpolate_pos_encoding at one size."""
    emb = encoder_embeddings
    orig = type(emb).interpolate_pos_encoding
    with torch.no_grad():
        n = (height // emb.config.patch_size) * (width // emb.config.patch_size)
        dummy = torch.zeros(1, n + 1, emb.position_embeddings.shape[-1],
                            dtype=emb.position_embeddings.dtype)
        cpu_pos = emb.position_embeddings.detach().cpu()
        holder = type("H", (), {})()
        holder.position_embeddings = cpu_pos
        holder.config = emb.config
        holder.patch_size = getattr(emb, "patch_size", emb.config.patch_size)
        table = orig(holder, dummy, height, width)               # [1, n+1, dim], CPU
    state = {"table": table, "dev": None, "hw": (height, width)}

    def cached(self, embeddings, h, w):
        if (h, w) != state["hw"]:
            raise ValueError(f"pos-encoding cache built for {state['hw']}, got {(h, w)}")
        if state["dev"] is None or state["dev"].device != embeddings.device:
            state["dev"] = state["table"].to(embeddings.device)
        return state["dev"]

    emb.interpolate_pos_encoding = cached.__get__(emb, type(emb))
    return table


def host_resize(images_cpu, size):
    """rae.py:84's resize, applied on the host to a CPU batch."""
    return F.interpolate(images_cpu, size=(size, size), mode="bicubic", align_corners=False)
