"""Opt-in standard batching and block compilation; no module replacement."""
import os
from collections import Counter
from functools import wraps

import torch


_DIAGNOSTICS = os.environ.get("RAE_FUSION_DIAGNOSTICS", "0") == "1"
_COUNTS = Counter()


def record_call(name):
    if _DIAGNOSTICS:
        _COUNTS[name] += 1


def fusion_counts():
    return dict(_COUNTS)


def configure_compile_compatibility():
    """Keep AOT saved buffers for repeated backward, for all compiled regions.

    This process-wide opt-in also affects decoder/discriminator blocks.
    It must persist through lazy forward compilation and backward execution.
    """
    if os.environ.get("RAE_DISABLE_DONATED_BUFFER", "0") == "1":
        import torch._functorch.config as config
        config.donated_buffer = False


def compile_decoder_block(forward):
    region = os.environ.get("RAE_COMPILE_REGION", "")
    if not region:
        return forward
    if region != "decoder_blocks":
        raise ValueError(f"Unsupported RAE_COMPILE_REGION: {region}")
    mode = os.environ.get("RAE_COMPILE_MODE", "default")
    modes = torch._inductor.list_mode_options()
    if mode not in modes:
        raise ValueError(f"Unsupported RAE_COMPILE_MODE: {mode}")
    options = dict(modes[mode])
    # Keep GEMM selection on the parent's ATen/cuBLAS route in every mode.
    options["max_autotune_gemm_backends"] = "ATEN"
    configure_compile_compatibility()
    compiled = torch.compile(forward, options=options, fullgraph=True)
    if not _DIAGNOSTICS:
        return compiled

    @wraps(forward)
    def counted(self, *args, **kwargs):
        key = f"decoder_blocks/{mode}/train={self.training}/grad={torch.is_grad_enabled()}"
        _COUNTS[key] += 1
        return compiled(self, *args, **kwargs)

    return counted


def compile_auxiliary(region):
    """Compile only a tensor region, preserving module ownership/state keys."""
    def decorate(forward):
        if os.environ.get("RAE_COMPILE_" + region.upper(), "0") != "1":
            return forward
        override = {"lpips_compare": "RAE_LPIPS_COMPILE_MODE",
                    "disc_norm": "RAE_DISC_NORM_COMPILE_MODE"}.get(region)
        mode = os.environ.get(override, os.environ.get("RAE_AUX_COMPILE_MODE", "default")) if override else os.environ.get("RAE_AUX_COMPILE_MODE", "default")
        if mode not in ("default", "max-autotune-no-cudagraphs"):
            raise ValueError(f"Unsupported RAE_AUX_COMPILE_MODE: {mode}")
        options = dict(torch._inductor.list_mode_options()[mode])
        options["triton.cudagraphs"] = False
        options["max_autotune_gemm_backends"] = "ATEN"
        configure_compile_compatibility()
        compiled = torch.compile(forward, options=options, fullgraph=True)
        @wraps(forward)
        def call(*args, **kwargs):
            record_call(f"{region}/{mode}/grad={torch.is_grad_enabled()}")
            return compiled(*args, **kwargs)
        return call
    return decorate
