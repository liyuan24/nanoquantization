from torch import nn
import torch
import triton
import triton.language as tl
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from nanoquantization.context import get_context


@triton.jit
def store_kv_cache_kernel(
    k_ptr: torch.Tensor,
    k_stride: int,
    v_ptr: torch.Tensor,
    v_stride: int,
    k_cache_ptr: torch.Tensor,
    v_cache_ptr: torch.Tensor,
    slot_mapping_ptr: torch.Tensor,
    D: tl.constexpr,
) -> None:
    """
    Store the new k and v tensors into the kv cache.
    Arguments:
        k_ptr: pointer to the k tensor
        k_stride: stride of the k tensor
        v_ptr: pointer to the v tensor
        v_stride: stride of the v tensor
        k_cache_ptr: pointer to the k cache tensor
        v_cache_ptr: pointer to the v cache tensor
        slot_mapping_ptr: pointer to the slot mapping tensor of shape [N]
        D: the size of KV cache
    """
    # which token to store KV cache
    id = tl.program_id(0)
    # the slot in the paged attention kv cache
    slot_idx = tl.load(slot_mapping_ptr + id)
    if slot_idx == -1:
        return
    # the range of k tensor for this token
    k_offset = id * k_stride + tl.arange(0, D)
    # the range of v tensor for this token
    v_offset = id * v_stride + tl.arange(0, D)
    # load the k and v tensors for this token
    k_tensor = tl.load(k_ptr + k_offset)
    v_tensor = tl.load(v_ptr + v_offset)
    cache_offset = slot_idx * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offset, k_tensor)
    tl.store(v_cache_ptr + cache_offset, v_tensor)


def store_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """
    Store the new k and v tensors into the kv cache. Since the input shape of k and v are [total_tokens, num_kv_heads, head_dim],
    we don't need to consider whether this is prefill or decoding. Just need to store the new k and v tensors into the kv cache for each token.

    Arguments:
        k: [total_tokens, num_kv_heads, head_dim]
        v: [total_tokens, num_kv_heads, head_dim]
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        slot_mapping: [total_tokens], the absolute position in kv cache block for each token
    """
    N, num_kv_heads, head_dim = k.shape
    D = num_kv_heads * head_dim  # the stride of k and v tensor in first dimension
    # launch a triton kernel to store the new k and v tensors in parallel for each token
    # NOTE: make sure to use k.stride(0) and v.stride(0) to get the correct stride for the first dimension
    # for qwen3, the k and v are derived by
    #     q, k, v = torch.split(
    #     x,
    #     [
    #         self.num_heads * self.head_dim,
    #         self.num_kv_heads * self.head_dim,
    #         self.num_kv_heads * self.head_dim,
    #     ],
    #     dim=-1,
    # )
    # and split create views for k and v. So the stride of k and v are not D, but much larger.
    store_kv_cache_kernel[(N,)](
        k, k.stride(0), v, v.stride(0), k_cache, v_cache, slot_mapping, D
    )


class Attention(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Arguments:
            q: [total_tokens, num_heads, head_dim]
            k: [total_tokens, num_kv_heads, head_dim]
            v: [total_tokens, num_kv_heads, head_dim]
        Returns:
            output: [total_tokens, num_heads, head_dim]
        """
        context = get_context()
        # existing kv cache of shape [num_blocks, block_size, num_kv_heads, head_dim] each
        k_cache, v_cache = self.k_cache, self.v_cache
        # we need to update the kv cache with new input k and v tensors
        if k_cache.numel() > 0 and v_cache.numel() > 0:
            store_kv_cache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:  # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=context.max_seqlen_q,
                max_seqlen_k=context.max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        else:
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                causal=True,
                softmax_scale=self.scale,
                block_table=context.block_tables,
                cache_seqlens=context.context_lens,
            )
        return o
