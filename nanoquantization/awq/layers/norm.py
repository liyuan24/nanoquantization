"""
This is the RMSNorm layer that enables kernel fusion
"""

from typing import Optional, Tuple
from torch import nn
import torch


class RMSNorm(nn.Module):
    def __init__(self, norm_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm_size = norm_size
        self.eps = eps
        # notice that the weight data type is the same as model data type, e.g. bfloat16
        self.weight = nn.Parameter(torch.ones(norm_size))

    @torch.compile
    def rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Use in-place operation to avoid unnecessary memory allocation

        kernel fusion for the point-wise multiplication and addition
        """
        orig_dtype = x.dtype
        # convert to float32 for better precision, create a new tensor
        x = x.float()
        inverse_vars = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x.mul_(inverse_vars)
        # convert back to original data type and then multiply by weight since weight is of original data type
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rms_norm(x)
