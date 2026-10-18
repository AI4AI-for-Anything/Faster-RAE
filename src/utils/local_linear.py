"""Local decoder Linear precision, with FP32 parameter ownership unchanged."""
import torch
import torch.nn.functional as F
from utils.compute_backends import enabled


def decoder_linear(layer, x, group):
    """Cast only the Linear operands; restore dtype before activation/residual.

    No low precision parameter cache: every call reflects optimizer updates.
    Cast autograd accumulates gradients into the original master parameters.
    BLAS preference is configured process-wide by configure_matmul, as in the
    parent implementation, and is not changed within this function.
    """
    if not enabled("RAE_DECODER_" + group.upper() + "_BF16"):
        return layer(x)
    bias = None if layer.bias is None else layer.bias.to(torch.bfloat16)
    return F.linear(x.to(torch.bfloat16), layer.weight.to(torch.bfloat16), bias).to(x.dtype)
