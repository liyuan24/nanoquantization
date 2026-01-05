import torch.nn as nn
import torch
import torch
import triton
import triton.language as tl

AWQ_TRITON_SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]


@triton.jit
def awq_dequantize_kernel(
    qweight_ptr,
    qscales_ptr,
    qzeros_ptr,
    result_ptr,
    num_cols,
    num_rows,
    group_size,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    offsets_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_qweights = offsets_y[:, None] * num_cols + offsets_x[None, :]

    masks_x = offsets_x < num_cols
    masks_y = offsets_y < num_rows
    masks = masks_x[:, None] & masks_y[None, :]

    # load qweights
    iweight = tl.load(qweight_ptr + offsets_qweights, mask=masks, other=0.0)
    # duplicate along columns for 8 times, shape: [BLOCK_SIZE_X, BLOCK_SIZE_Y * 8]
    # for each 32-bit integer in the row, we will extract the 4-bit integer from it as the unpacked quantized weight
    # the extraction is done with AND operation with a shift mask and each mask is 0xF (1111 in binary) to get the corresponding 4-bit integer
    iweight = tl.interleave(iweight, iweight)
    iweight = tl.interleave(iweight, iweight)
    iweight = tl.interleave(iweight, iweight)

    # The unpack logic is as follows:
    # When we pack 8 4-bit integers into 1 32-bit integer,
    # 0th 4-bit integer is the least significant 4 bits
    # followed by 2nd 4-bit integer
    # followed by 4th 4-bit integer
    # followed by 6th 4-bit integer
    # followed by 1st 4-bit integer
    # followed by 3rd 4-bit integer
    # followed by 5th 4-bit integer
    # followed by 7th 4-bit integer
    # so the right shift for 0th 4-bit is 0(= 0 * 4)
    # so the right shift for 1st 4-bit is 16(= 4 * 4)
    # so the right shift for 2nd 4-bit is 4(= 1 * 4)
    # so the right shift for 3rd 4-bit is 20(= 5 * 4)
    # so the right shift for 4th 4-bit is 8(= 2 * 4)
    # so the right shift for 5th 4-bit is 24(= 6 * 4)
    # so the right shift for 6th 4-bit is 12(= 3 * 4)
    # so the right shift for 7th 4-bit is 28(= 7 * 4)
    # [[0],
    #  [1],
    #  [2],   + [[0, 4]] = [[0, 4], [1, 5], [2, 6], [3, 7]]
    #  [3]]
    shift_order = tl.arange(0, 4)[:, None] + tl.arange(0, 2)[None, :] * 4
    shift_order = shift_order * 4
    shift_order = shift_order.reshape(1, 8)
    shift_order = shift_order.broadcast_to(BLOCK_SIZE_X * BLOCK_SIZE_Y, 8)
    shift_order = shift_order.reshape(BLOCK_SIZE_Y, BLOCK_SIZE_X * 8)
    iweight = (iweight >> shift_order) & 0xF

    # process qzeros

    # the assumption is that each column in iweight share the same qzero which is guaranteed by group_size % BLOCK_SIZE_Y = 0
    offsets_qzeros_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    offsets_qzeros_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offsets_qzeros = offsets_qzeros_y[:, None] * num_cols + offsets_qzeros_x[None, :]
    masks_qzeros_x = offsets_qzeros_x < num_cols
    masks_qzeros_y = offsets_qzeros_y < num_rows // group_size
    masks_qzeros = masks_qzeros_y[:, None] & masks_qzeros_x[None, :]

    qzeros = tl.load(
        qzeros_ptr + offsets_qzeros, mask=masks_qzeros, other=0.0
    )  # shape: [1, BLOCK_SIZE_X]
    # repeat qzeros along columns for 8 times to be shape [1, BLOCK_SIZE_X * 8]
    qzeros = tl.interleave(qzeros, qzeros)
    qzeros = tl.interleave(qzeros, qzeros)
    qzeros = tl.interleave(qzeros, qzeros)
    # broadcast to be shape [BLOCK_SIZE_Y, BLOCK_SIZE_X * 8]
    qzeros = qzeros.broadcast_to(BLOCK_SIZE_Y, BLOCK_SIZE_X * 8)
    # use the same shift_order to unpack the 4-bit integer from qzeros
    qzeros = (qzeros >> shift_order) & 0xF

    # process qscales
    offsets_qscales_y = pid_y * BLOCK_SIZE_Y // group_size + tl.arange(0, 1)
    offsets_qscales_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    offsets_qscales = (
        offsets_qscales_y[:, None] * num_cols * 8 + offsets_qscales_x[None, :]
    )
    masks_qscales_x = offsets_qscales_x < num_cols * 8
    masks_qscales_y = offsets_qscales_y < num_rows // group_size
    masks_qscales = masks_qscales_y[:, None] & masks_qscales_x[None, :]
    qscales = tl.load(
        qscales_ptr + offsets_qscales, mask=masks_qscales, other=0.0
    )  # shape: [1, BLOCK_SIZE_X * 8]
    qscales = qscales.broadcast_to(
        BLOCK_SIZE_Y, BLOCK_SIZE_X * 8
    )  # shape: [BLOCK_SIZE_Y, BLOCK_SIZE_X * 8]
    # no need to unpack as qscales is already in float16

    # change the type to float16 by type promotion
    # https://triton-lang.org/main/python-api/triton-semantics.html#type-promotion
    iweight = (iweight - qzeros) * qscales

    # store the result
    offsets_results_x = pid_x * BLOCK_SIZE_X * 8 + tl.arange(0, BLOCK_SIZE_X * 8)
    offsets_results_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offsets_results = (
        offsets_results_y[:, None] * num_cols * 8 + offsets_results_x[None, :]
    )
    masks_results_x = offsets_results_x < num_cols * 8
    masks_results_y = offsets_results_y < num_rows
    masks_results = masks_results_y[:, None] & masks_results_x[None, :]
    tl.store(result_ptr + offsets_results, iweight, mask=masks_results)


def awq_dequantize(
    qweight: torch.Tensor,
    qscales: torch.Tensor,
    qzeros: torch.Tensor,
    block_size_x: int = 32,
    block_size_y: int = 32,
) -> torch.Tensor:
    """
    Dequantize the quantized weight into float16 weight.
    Args:
        qweight: the quantized weight, shape: [in_features, out_features // (32 // w_bits)]
        qscales: the scales to quantize the weight, shape: [in_features // group_size, out_features]
        qzeros: the zeros to quantize the weight, shape: [in_features // group_size, out_features // (32 // w_bits)]
        block_size_x: the block size of the input tensor, default is 32
        block_size_y: the block size of the output tensor, default is 32
    Returns:
        the dequantized weight, shape: [in_features, out_features]
    """
    K = qweight.shape[0]
    M = qscales.shape[1]
    group_size = qweight.shape[0] // qscales.shape[0]

    assert K > 0 and M > 0
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == M // 8
    assert qweight.shape[1] == M // 8
    assert group_size <= K
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    result = torch.empty(
        qweight.shape[0],
        qweight.shape[1] * 8,
        dtype=qscales.dtype,
        device=qweight.device,
    )

    Y = qweight.shape[0]  # number of rows
    X = qweight.shape[1]  # number of columns

    grid = lambda META: (
        triton.cdiv(X, META["BLOCK_SIZE_X"]),
        triton.cdiv(Y, META["BLOCK_SIZE_Y"]),
    )

    awq_dequantize_kernel[grid](
        qweight,
        qscales,
        qzeros,
        result,
        X,
        Y,
        group_size,
        BLOCK_SIZE_X=block_size_x,
        BLOCK_SIZE_Y=block_size_y,
    )
    return result


@triton.jit
def awq_gemm_kernel(
    x_ptr,
    qweight_ptr,
    qscales_ptr,
    qzeros_ptr,
    out_ptr,
    group_size,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offsets_x_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    masks_x_m = offsets_x_m < M

    offsets_qweight_n = pid_n * BLOCK_SIZE_N // 8 + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_qweight_n = offsets_qweight_n < N // 8

    offsets_qscales_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    masks_qscales_n = offsets_qscales_n < N

    offsets_qzeros_n = pid_n * BLOCK_SIZE_N // 8 + tl.arange(0, BLOCK_SIZE_N // 8)
    masks_qzeros_n = offsets_qzeros_n < N // 8

    offsets_k = pid_k * BLOCK_SIZE_K + tl.arange(
        0, BLOCK_SIZE_K
    )  # for the first block in K dimension
    offsets_x = offsets_x_m[:, None] * K + offsets_k[None, :]
    offsets_qweight = offsets_k[:, None] * N // 8 + offsets_qweight_n[None, :]

    # The unpack logic is as follows:
    # When we pack 8 4-bit integers into 1 32-bit integer,
    # 0th 4-bit integer is the least significant 4 bits
    # followed by 2nd 4-bit integer
    # followed by 4th 4-bit integer
    # followed by 6th 4-bit integer
    # followed by 1st 4-bit integer
    # followed by 3rd 4-bit integer
    # followed by 5th 4-bit integer
    # followed by 7th 4-bit integer
    # so the right shift for 0th 4-bit is 0(= 0 * 4)
    # so the right shift for 1st 4-bit is 16(= 4 * 4)
    # so the right shift for 2nd 4-bit is 4(= 1 * 4)
    # so the right shift for 3rd 4-bit is 20(= 5 * 4)
    # so the right shift for 4th 4-bit is 8(= 2 * 4)
    # so the right shift for 5th 4-bit is 24(= 6 * 4)
    # so the right shift for 6th 4-bit is 12(= 3 * 4)
    # so the right shift for 7th 4-bit is 28(= 7 * 4)
    # [[0],
    #  [1],
    #  [2],   + [[0, 4]] = [[0, 4], [1, 5], [2, 6], [3, 7]]
    #  [3]]
    shift_order = tl.arange(0, 4)[:, None] + tl.arange(0, 2)[None, :] * 4
    shift_order = shift_order * 4
    shift_order = shift_order.reshape(1, 8)
    shift_order = shift_order.broadcast_to(BLOCK_SIZE_K * (BLOCK_SIZE_N // 8), 8)
    shift_order = shift_order.reshape(BLOCK_SIZE_K, BLOCK_SIZE_N)

    accumulator_dtype = tl.float32
    output_dtype = out_ptr.type.element_ty
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        masks_k = offsets_k < K
        masks_x = masks_x_m[:, None] & masks_k[None, :]
        x = tl.load(x_ptr + offsets_x, mask=masks_x, other=0.0)
        x = x.to(output_dtype)

        masks_qweight = masks_k[:, None] & masks_qweight_n[None, :]
        qweight = tl.load(qweight_ptr + offsets_qweight, mask=masks_qweight, other=0.0)
        # replicate the qweight 8 times along the column dimension since each column in qweight is 8 4-bit integers packed into 1 32-bit integer
        qweight = tl.interleave(qweight, qweight)
        qweight = tl.interleave(qweight, qweight)
        qweight = tl.interleave(qweight, qweight)  # shape: [BLOCK_SIZE_K, BLOCK_SIZE_N]

        # the assumption is that each column in qweight share the same qscales and qzeros which is guaranteed by group_size % BLOCK_SIZE_K = 0
        offsets_qscales_zeros_k = (
            BLOCK_SIZE_K * SPLIT_K * k + pid_k * BLOCK_SIZE_K
        ) // group_size + tl.arange(0, 1)
        offsets_qscales = (
            offsets_qscales_zeros_k[:, None] * N + offsets_qscales_n[None, :]
        )
        masks_qscales_k = offsets_qscales_zeros_k < K // group_size
        masks_qscales = masks_qscales_k[:, None] & masks_qscales_n[None, :]
        qscales = tl.load(
            qscales_ptr + offsets_qscales, mask=masks_qscales, other=0.0
        )  # shape: [1, BLOCK_SIZE_N]
        qscales = qscales.broadcast_to(
            BLOCK_SIZE_K, BLOCK_SIZE_N
        )  # shape: [BLOCK_SIZE_K, BLOCK_SIZE_N]

        offsets_qzeros = (
            offsets_qscales_zeros_k[:, None] * N // 8 + offsets_qzeros_n[None, :]
        )
        masks_qzeros_k = offsets_qscales_zeros_k < K // group_size
        masks_qzeros = masks_qzeros_k[:, None] & masks_qzeros_n[None, :]
        qzeros = tl.load(
            qzeros_ptr + offsets_qzeros, mask=masks_qzeros, other=0.0
        )  # shape: [1, BLOCK_SIZE_N // 8]
        # replicate the qzeros 8 times along the column dimension since each column in qzeros is 8 4-bit integers packed into 1 32-bit integer
        qzeros = tl.interleave(qzeros, qzeros)
        qzeros = tl.interleave(qzeros, qzeros)
        qzeros = tl.interleave(qzeros, qzeros)  # shape: [1, BLOCK_SIZE_N]
        qzeros = qzeros.broadcast_to(
            BLOCK_SIZE_K, BLOCK_SIZE_N
        )  # shape: [BLOCK_SIZE_K, BLOCK_SIZE_N]

        # do the unpacking for both qweight and qzeros
        qweight = (qweight >> shift_order) & 0xF
        qzeros = (qzeros >> shift_order) & 0xF
        qweight = (qweight - qzeros) * qscales
        qweight = qweight.to(output_dtype)

        # need to assign the updated accumulator back to the accumulator variable otherwise it will not be updated
        # NOTICE: for bfloat16 type of activation and weight, here we are doing matrix multiplication in bf16 and accumulate in fp32
        # since tensor core can only support fp32 accumulation for bf16 matrix multiplication
        accumulator = tl.dot(
            x, qweight, acc=accumulator, out_dtype=accumulator_dtype
        )  # accumulator += x * qweight

        offsets_k += BLOCK_SIZE_K * SPLIT_K
        offsets_x += BLOCK_SIZE_K * SPLIT_K
        offsets_qweight += BLOCK_SIZE_K * SPLIT_K * (N // 8)

    offsets_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_out = offsets_out_m[:, None] * N + offsets_out_n[None, :]
    masks_out = (offsets_out_m[:, None] < M) & (offsets_out_n[None, :] < N)
    # NOTICE: for bf16 output type, we cast the fp32 accumulator back to bf16
    tl.store(
        out_ptr + offsets_out + pid_k * M * N,
        accumulator.to(output_dtype),
        mask=masks_out,
    )


def awq_gemm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qscales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
) -> torch.Tensor:
    """
    x: [M, K]
    qweight: [K, N // 8]
    qscales: [K // group_size, N]
    qzeros: [K // group_size, N // 8]
    split_k_iters: int, power of 2
    """
    M, K = x.shape
    N = qweight.shape[1] * 8
    group_size = qweight.shape[0] // qscales.shape[0]

    assert qweight.shape[0] == K and qweight.shape[1] == N // 8
    assert qscales.shape[0] == K // group_size and qscales.shape[1] == N
    assert qzeros.shape[0] == K // group_size and qzeros.shape[1] == N // 8
    assert split_k_iters & (split_k_iters - 1) == 0
    assert group_size in AWQ_TRITON_SUPPORTED_GROUP_SIZES or group_size == K

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        split_k_iters,
    )

    out = torch.zeros(split_k_iters, M, N, dtype=qscales.dtype, device=x.device)

    awq_gemm_kernel[grid](
        x,
        qweight,
        qscales,
        qzeros,
        out,
        group_size,
        M,
        N,
        K,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        SPLIT_K=split_k_iters,
    )
    out = out.sum(dim=0)
    return out


class WQLinear_GEMM(nn.Module):
    def __init__(
        self,
        w_bits: int,
        group_size: int,
        in_features: int,
        out_features: int,
        device,
        bias: bool = False,
        output_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if w_bits not in [4]:
            raise ValueError(f"Only 4-bit quantization is supported")

        self.w_bits = w_bits
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        assert (
            in_features % group_size == 0
        ), "in_features must be divisible by group_size"
        assert (
            out_features % (32 // self.w_bits) == 0
        ), "out_features must be divisible by (32 // self.w_bits)"

        self.register_buffer(
            "qweight",
            torch.zeros(
                in_features,
                out_features // (32 // self.w_bits),
                dtype=torch.int32,
                device=device,
            ),
        )
        self.register_buffer(
            "qscales",
            torch.zeros(
                (in_features // group_size, out_features),
                dtype=output_dtype,
                device=device,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // group_size, out_features // (32 // self.w_bits)),
                dtype=torch.int32,
                device=device,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.bfloat16,
                    device=device,
                ),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear_layer: nn.Linear,
        w_bits: int,
        group_size: int,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> "WQLinear_GEMM":
        """
        Args:
            linear_layer: the linear layer to be converted, weight shape: [n_output, n_input]
            w_bits: the number of bits to quantize the weight
            group_size: the group size to quantize the weight
            scales: the scales to quantize the weight, shape: [n_groups, n_output] where n_groups = n_input // group_size
            zeros: the zeros to quantize the weight, shape: [n_groups, n_output]
        """
        awq_linear = cls(
            w_bits,
            group_size,
            linear_layer.in_features,
            linear_layer.out_features,
            linear_layer.weight.device,
            linear_layer.bias is not None,
            output_dtype,
        )
        awq_linear.qscales = scales.clone().to(output_dtype)
        if linear_layer.bias is not None:
            awq_linear.bias = linear_layer.bias.clone().to(output_dtype)

        # convert float weight to int weight(quantization)
        intweight = []
        for i in range(awq_linear.in_features):
            intweight.append(
                torch.round(
                    linear_layer.weight[:, i] / scales[i // group_size]
                    + zeros[i // group_size]
                ).to(torch.int32)[:, None]
            )
        # shape: [out_features, in_features]
        intweight = torch.cat(intweight, dim=1)
        # shape: [in_features, out_features]
        intweight = intweight.t().contiguous()
        intweight = intweight.to(torch.int32)

        # pack 8 4-bit ints along out_features to 1 32-bit int for linear weight
        qweight = torch.zeros(
            awq_linear.in_features,
            awq_linear.out_features // (32 // awq_linear.w_bits),
            dtype=torch.int32,
            device=intweight.device,
        )

        num_pack = 32 // awq_linear.w_bits
        for i in range(qweight.shape[1]):
            if awq_linear.w_bits == 4:
                col_maps = [
                    0,
                    2,
                    4,
                    6,
                    1,
                    3,
                    5,
                    7,
                ]  # first even indices then odd indices
            else:
                raise ValueError(f"Only 4-bit quantization is supported")
            for j in range(num_pack):
                qweight[:, i] |= intweight[:, i * num_pack + col_maps[j]] << (
                    j * awq_linear.w_bits
                )
        awq_linear.qweight = qweight

        # pack 8 4-bit ints along out_features to 1 32-bit int for zeros
        zeros = zeros.to(torch.int32)
        qzeros = torch.zeros(
            awq_linear.in_features // awq_linear.group_size,
            awq_linear.out_features // (32 // awq_linear.w_bits),
            dtype=torch.int32,
            device=zeros.device,
        )
        for i in range(qzeros.shape[1]):
            if awq_linear.w_bits == 4:
                col_maps = [
                    0,
                    2,
                    4,
                    6,
                    1,
                    3,
                    5,
                    7,
                ]  # first even indices then odd indices
            else:
                raise ValueError(f"Only 4-bit quantization is supported")
            for j in range(num_pack):
                qzeros[:, i] |= zeros[:, i * num_pack + col_maps[j]] << (
                    j * awq_linear.w_bits
                )
        awq_linear.qzeros = qzeros

        return awq_linear

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = x.shape[:-1] + (self.out_features,)
        # when number of tokens is equal or greater than 256, we will NOT fuse dequantization and matrix multiplication into one kernel
        # since it is likely that it is compute bound instead of memory bound
        x = x.reshape(-1, x.shape[-1])
        FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256
        if FP16_MATMUL_HEURISTIC_CONDITION:
            out = awq_dequantize(self.qweight, self.qscales, self.qzeros)
            out = torch.matmul(x, out)
        else:
            # fuse dequantization and matrix multiplication into one kernel
            out = awq_gemm(x, self.qweight, self.qscales, self.qzeros, split_k_iters=8)
        if self.bias is not None:
            out.add_(self.bias)
        return out.reshape(out_shape)
