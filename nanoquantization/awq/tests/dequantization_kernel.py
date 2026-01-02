from typing import Tuple
import torch
import pytest
import torch.nn as nn
from nanoquantization.awq.modules.gemm import awq_dequantize, WQLinear_GEMM, awq_gemm

"""
python -m nanoquantization.awq.tests.dequantization_kernel
"""


def pseudo_quantize_tensor(
    x: torch.Tensor, group_size: int, w_bits: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate the quantization and de-quantization of a tensor to w_bits bits.
    """
    orig_shape = x.shape
    x = x.view(-1, group_size)
    assert torch.isnan(x).sum() == 0, "weight should not contain nan"
    device = x.device

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

    x = x.reshape(orig_shape)
    scales = scales.view(orig_shape[0], -1)
    # NOTE: this scales is the block quantization scale, not the weight scale factors from activation magnitude
    return x, scales, zeros


class TestAWQDequantizeKernel:
    @pytest.fixture
    def sample_quantized_data(self):
        """Create sample quantized data for testing."""
        in_features = 1024
        out_features = 2048
        group_size = 32
        w_bits = 4

        weight = torch.randn(
            out_features,
            in_features,
            dtype=torch.float16,
            device="cuda",
        )

        weight, scales, zeros = pseudo_quantize_tensor(weight, group_size, w_bits)
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        linear_layer = (
            nn.Linear(in_features, out_features, bias=False)
            .to("cuda")
            .to(torch.float16)
        )
        linear_layer.weight.data = weight
        awq_linear = WQLinear_GEMM.from_linear(
            linear_layer, w_bits, group_size, scales, zeros
        )

        return {
            "original_weight": weight,
            "qweight": awq_linear.qweight,
            "qscales": awq_linear.qscales,
            "qzeros": awq_linear.qzeros,
            "in_features": in_features,
            "out_features": out_features,
        }

    def test_dequantize_output(self, sample_quantized_data):
        original_weight = sample_quantized_data["original_weight"]
        qweight = sample_quantized_data["qweight"]
        qscales = sample_quantized_data["qscales"]
        qzeros = sample_quantized_data["qzeros"]
        in_features = sample_quantized_data["in_features"]
        out_features = sample_quantized_data["out_features"]

        result = awq_dequantize(qweight, qscales, qzeros)

        assert result.shape == (
            in_features,
            out_features,
        ), f"Expected shape ({in_features}, {out_features}), got {result.shape}"
        assert (
            result.dtype == qscales.dtype
        ), f"Expected dtype {qscales.dtype}, got {result.dtype}"
        assert torch.allclose(result, original_weight.t().contiguous(), atol=1e-2)
    
    def test_gemm_output(self, sample_quantized_data):
        original_weight = sample_quantized_data["original_weight"]
        qweight = sample_quantized_data["qweight"]
        qscales = sample_quantized_data["qscales"]
        qzeros = sample_quantized_data["qzeros"]
        in_features = sample_quantized_data["in_features"]
        out_features = sample_quantized_data["out_features"]
        num_tokens = 8

        # the input tensor
        x = torch.randn(num_tokens, in_features, dtype=torch.float16, device=original_weight.device)

        print(f"x type: {x.dtype}, original_weight type: {original_weight.dtype}, qweight type: {qweight.dtype}, qscales type: {qscales.dtype}, qzeros type: {qzeros.dtype}")

        result = awq_gemm(x, qweight, qscales, qzeros, split_k_iters=8)

        assert result.shape == (
            num_tokens,
            out_features,
        ), f"Expected shape ({in_features}, {out_features}), got {result.shape}"
        assert (
            result.dtype == qscales.dtype
        ), f"Expected dtype {qscales.dtype}, got {result.dtype}"
        expected_result = torch.matmul(x, original_weight.t().contiguous())
        assert torch.allclose(result, torch.matmul(x, original_weight.t().contiguous()), atol=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
