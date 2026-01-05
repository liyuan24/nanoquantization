import torch
from typing import Tuple


def pseudo_quantize(
    x: torch.Tensor, group_size: int, w_bits: int, zero_point: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate the quantization and de-quantization of a tensor to w_bits bits.
    """
    orig_shape = x.shape
    if group_size > 0:
        assert (
            x.shape[-1] % group_size == 0
        ), "x.shape[-1] must be divisible by group_size"
        x = x.view(-1, group_size)
    assert torch.isnan(x).sum() == 0, "weight should not contain nan"
    device = x.device

    if zero_point:
        # asymmetric quantization formula: q = round(x / scale) + zero_point
        # get the max for each group(row)
        max_val = x.amax(dim=-1, keepdim=True).to(device)
        # get the min for each group(row)
        min_val = x.amin(dim=-1, keepdim=True).to(device)
        max_quant_int = 2**w_bits - 1
        min_quant_int = 0
        # map the values between min_val and max_val to min_quant_int and max_quant_int
        scales = (max_val - min_val).clamp(min=1e-5) / max_quant_int
        # make sure the min_val is mapped to 0 which is the minimum quantized value
        zeros = (
            (-torch.round(min_val / scales))
            .clamp(min=min_quant_int, max=max_quant_int)
            .to(device)
        )
        # simulation quantization and de-quantization
        x = (torch.round(x / scales) + zeros).clamp(
            min=min_quant_int, max=max_quant_int
        )  # quantization
        x = (x - zeros) * scales  # de-quantization
        zeros = zeros.view(orig_shape[0], -1)
    else:
        # symmetric quantization formula: q = round(x / scale)
        max_quant_int = 2 ** (w_bits - 1) - 1
        min_quant_int = -(2 ** (w_bits - 1))
        max_val = x.abs().amax(dim=-1, keepdim=True).to(device)
        scales = max_val / max_quant_int
        # simulate quantization and de-quantization
        x = torch.round(x / scales).clamp(
            min=min_quant_int, max=max_quant_int
        )  # quantization
        x = x * scales  # de-quantization
        zeros = None

    x = x.reshape(orig_shape)
    scales = scales.view(orig_shape[0], -1)
    # NOTE: this scales is the block quantization scale, not the weight scale factors from activation magnitude
    return x, scales, zeros
