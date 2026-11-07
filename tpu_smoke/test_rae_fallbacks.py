"""CPU bitwise check for tpu_smoke/rae_fallbacks.py (locks: same bytes as upstream).

  1. cached pos-encoding table == HF interpolate_pos_encoding at 224 (torch.equal);
  2. rae.encode(host_resize(x)) == rae.encode(x) for x at 256 (eval mode, no noise),
     i.e. moving rae.py:84's resize to the host changes nothing;
  3. the cache refuses another size instead of returning a wrong table.
Run from the repository root (CPU; needs the DINOv2-with-registers base weights in the
HF cache):  python -m tpu_smoke.test_rae_fallbacks
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from tpu_smoke.rae_fallbacks import cache_pos_encoding, host_resize


def main():
    from stage1 import RAE
    torch.manual_seed(0)
    rae = RAE(encoder_cls="Dinov2withNorm",
              encoder_config_path="facebook/dinov2-with-registers-base",
              encoder_input_size=224,
              encoder_params=dict(dinov2_path="facebook/dinov2-with-registers-base", normalize=True),
              decoder_config_path=os.path.join(_ROOT, "configs", "decoder", "ViTXL"),
              noise_tau=0.8, reshape_to_2d=True).eval()
    emb = rae.encoder.encoder.embeddings
    x = torch.rand(2, 3, 256, 256, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref_pos = emb.interpolate_pos_encoding(torch.zeros(1, 257, 768), 224, 224)
        ref_z = rae.encode(x)
        table = cache_pos_encoding(emb, 224, 224)
        assert torch.equal(table, ref_pos), "cached table differs"
        got_z = rae.encode(host_resize(x, 224))
    assert torch.equal(got_z, ref_z), f"encode differs: {(got_z - ref_z).abs().max()}"
    print("PASS pos-encoding cache and host resize are bitwise equal to upstream")
    try:
        emb.interpolate_pos_encoding(torch.zeros(1, 1370, 768), 518, 518)
    except ValueError:
        print("PASS cache refuses another input size")
    else:
        raise AssertionError("cache accepted a different size")


if __name__ == "__main__":
    main()
