"""Scoped auxiliary BF16 operands, preserving FP32 parameter ownership."""
import torch
from torch import nn
from torch.nn import functional as F
from utils.compute_backends import enabled


def auxiliary_linear(layer, x, group):
    if not enabled("RAE_" + group + "_BF16"):
        return F.linear(x, layer.weight, layer.bias)
    bias = None if layer.bias is None else layer.bias.to(torch.bfloat16)
    return F.linear(x.to(torch.bfloat16), layer.weight.to(torch.bfloat16), bias).to(x.dtype)


class EncoderLinear(nn.Linear):
    def forward(self, x):
        return auxiliary_linear(self, x, "ENCODER_LINEAR")


class VGGConv2d(nn.Conv2d):
    def forward(self, x):
        if not enabled("RAE_LPIPS_VGG_CONV_BF16"):
            return super().forward(x)
        bias = None if self.bias is None else self.bias.to(torch.bfloat16)
        # _conv_forward preserves the module padding mode as well as groups,
        # dilation and stride. No parameters or persistent buffers are replaced.
        return self._conv_forward(x.to(torch.bfloat16), self.weight.to(torch.bfloat16), bias).to(x.dtype)
