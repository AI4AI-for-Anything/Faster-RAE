"""Explicit, opt-in local QKV precision for standard SDPA backends."""
import os

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def local_bf16_attention(query, key, value, *, scale=None, dropout_p=0.0):
    backend = os.environ.get("RAE_LOCAL_ATTN_BACKEND", "math")
    choices = {"math": SDPBackend.MATH, "efficient": SDPBackend.EFFICIENT_ATTENTION,
               "flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}
    if backend not in choices:
        raise ValueError(f"Unsupported RAE_LOCAL_ATTN_BACKEND: {backend}")
    dtype = query.dtype
    with sdpa_kernel(choices[backend]):
        result = torch.nn.functional.scaled_dot_product_attention(
            query.to(torch.bfloat16), key.to(torch.bfloat16), value.to(torch.bfloat16),
            dropout_p=dropout_p, scale=scale)
    return result.to(dtype)
