from typing import Optional, Tuple
from nanoquantization.context import get_context
from nanoquantization.layers.attention import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoquantization.layers.norm import RMSNorm
from nanoquantization.layers.rope import RoPE


class Qwen3Attention(nn.Module):
    """
    Per device attention layer for Qwen3.
    """

    def __init__(
        self,
        hidden_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        max_position_embeddings: int,
        head_dim: Optional[int] = None,
        rope_theta: float = 10000,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.head_dim = head_dim or hidden_size // total_num_heads
        self.hidden_size = hidden_size
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope = RoPE(
            head_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        self.attn = Attention(scale=self.head_dim**-0.5)
        self.q_proj = nn.Linear(hidden_size, total_num_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_size, total_num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, total_num_kv_heads * self.head_dim)
        self.o_proj = nn.Linear(total_num_heads * self.head_dim, hidden_size)
        self.q_norm = RMSNorm(norm_size=self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(norm_size=self.head_dim, eps=rms_norm_eps)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: [total_tokens, hidden_size]
            position_ids: [total_tokens]
        Returns:
            output: [total_tokens, hidden_size]
        """
        # q shape: [total_tokens, num_heads * head_dim]
        # k shape: [total_tokens, num_kv_heads * head_dim]
        # v shape: [total_tokens, num_kv_heads * head_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # shape: [total_tokens, num_heads, head_dim], [total_tokens, num_kv_heads, head_dim], [total_tokens, num_kv_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # apply QK norm
        q, k = self.q_norm(q), self.k_norm(k)
        # apply rotary positional embeding
        q, k = self.rope(q, k, position_ids)
        # shape: [total_tokens, num_heads, head_dim]
        o = self.attn(q, k, v)
        # shape: [total_tokens, num_heads * head_dim]
        o = o.flatten(1, -1)
        # apply output projection, shape: [total_tokens, hidden_size]
        return self.o_proj(o)


class Qwen3MLP(nn.Module):
    """
    Per device MLP layer for Qwen3.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: [total_tokens, hidden_size]
        Returns:
            output: [total_tokens, hidden_size]
        """
        # shape: [total_tokens, intermediate_size]
        y1 = self.gate_proj(x)
        y2 = self.up_proj(x)
        # shape: [total_tokens, intermediate_size]
        x = F.silu(y1) * y2
        # shape: [total_tokens, hidden_size]
        x = self.down_proj(x)
        return x


class Qwen3Block(nn.Module):
    """
    Per device block for Qwen3, which consists of an attention layer and an MLP layer
    """

    def __init__(
        self,
        hidden_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        max_position_embeddings: int,
        intermediate_size: int,
        head_dim: Optional[int] = None,
        rope_theta: float = 10000,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
        )
        self.mlp = Qwen3MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(norm_size=hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(norm_size=hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        we can simply do
        x = x + self.self_attn(positions, norm(hiddent_states))
        x = x + self.mlp(norm(x))

        But we can see that add is one kernel and norm is another kernel. Here we fuse the add and norm into one kernel.

        Arguments:
            x: [total_tokens, hidden_size]
            position_ids: [total_tokens]
            residual: [total_tokens, hidden_size]
        Returns:
            output: [total_tokens, hidden_size]
        """
        if residual is None:
            x, residual = self.input_layernorm(x), x
        else:
            x, residual = self.input_layernorm(x, residual)
        # shape: [total_tokens, hidden_size]
        x = self.self_attn(x, position_ids)
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual


class Qwen3Model(nn.Module):
    """
    Qwen3 model with multiple blocks and a final layer norm before the vocab projection.
    """

    def __init__(
        self,
        num_hidden_layers: int,
        vocab_size: int,
        hidden_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        max_position_embeddings: int,
        intermediate_size: int,
        head_dim: Optional[int] = None,
        rope_theta: float = 10000,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Qwen3Block(
                    hidden_size,
                    total_num_heads,
                    total_num_kv_heads,
                    max_position_embeddings,
                    intermediate_size,
                    head_dim,
                    rope_theta,
                    rms_norm_eps,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(norm_size=hidden_size, eps=rms_norm_eps)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: [total_tokens]
            position_ids: [total_tokens]
        Returns:
            output: [total_tokens, hidden_size]
        """
        # shape: [total_tokens, hidden_size]
        x = self.embed_tokens(x)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, position_ids, residual)
        # post-norm
        x, _ = self.norm(x, residual)
        return x


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 model for causal language modeling with LM head.
    """

    def __init__(
        self,
        num_hidden_layers: int,
        vocab_size: int,
        hidden_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        max_position_embeddings: int,
        intermediate_size: int,
        head_dim: Optional[int] = None,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.model = Qwen3Model(
            num_hidden_layers,
            vocab_size,
            hidden_size,
            total_num_heads,
            total_num_kv_heads,
            max_position_embeddings,
            intermediate_size,
            head_dim,
            rope_theta,
            rms_norm_eps,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        if tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: [total_tokens]
            position_ids: [total_tokens]
        Returns:
            output: [total_tokens, hidden_size]
        """
        return self.model(x, position_ids)

    def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: [total_tokens, hidden_size]
        Returns:
            output: [total_tokens, vocab_size]
        """
        context = get_context()
        if context.is_prefill:
            last_token_ids = context.cu_seqlens_q[1:] - 1
            x = x[last_token_ids].contiguous()
        return self.lm_head(x)

    def compute_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: [sequence_length]
            labels: [sequence_length]
        Returns:
            loss: [1]
        """
        position_ids = torch.arange(x.shape[0], dtype=torch.int32).cuda(
            non_blocking=True
        )
        hidden_states = self.model(x, position_ids)
        logits = self.lm_head(hidden_states)
        loss = F.cross_entropy(logits, labels)
        return loss
