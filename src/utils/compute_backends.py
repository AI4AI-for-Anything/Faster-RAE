"""Opt-in native compute backends; defaults retain upstream behavior."""
import os
from contextlib import nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def enabled(name):
    return os.environ.get(name, "0") == "1"


def configure_cudnn_benchmark():
    """Opt in to process-wide convolution autotuning in nondeterministic runs."""
    if (enabled("RAE_CUDNN_BENCHMARK")
            and not torch.are_deterministic_algorithms_enabled()
            and not torch.backends.cudnn.deterministic):
        torch.backends.cudnn.benchmark = True


def configure_matmul():
    if "RAE_MATMUL_TF32" in os.environ:
        torch.backends.cuda.matmul.allow_tf32 = enabled("RAE_MATMUL_TF32")
    backend = os.environ.get("RAE_BLAS_LIBRARY")
    if backend is not None:
        if backend not in ("cublas", "cublaslt"):
            raise ValueError(f"Unsupported RAE_BLAS_LIBRARY: {backend}")
        torch.backends.cuda.preferred_blas_library(backend)


def attention_backend():
    backend = os.environ.get("RAE_SDPA_BACKEND", "auto")
    if backend == "auto":
        return nullcontext()
    choices = {"math": SDPBackend.MATH, "efficient": SDPBackend.EFFICIENT_ATTENTION,
               "flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}
    if backend not in choices:
        raise ValueError(f"Unsupported RAE_SDPA_BACKEND: {backend}")
    return sdpa_kernel(choices[backend])
