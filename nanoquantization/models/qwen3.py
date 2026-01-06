from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoquantization.layers.norm import RMSNorm
from nanoquantization.layers.rope import RoPE


class Qwen3Attention(nn.Module):
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
        self.q_proj = nn.Linear(
            hidden_size, total_num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, total_num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, total_num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            total_num_heads * self.head_dim, hidden_size, bias=False
        )
        self.q_norm = RMSNorm(norm_size=self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(norm_size=self.head_dim, eps=rms_norm_eps)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: [batch_size, sequence_length, hidden_size]
            position_ids: [batch_size, sequence_length]
        Returns:
            output: [batch_size, sequence_length, hidden_size]
        """
        # q shape: [batch_size, sequence_length, num_heads * head_dim]
        # k shape: [batch_size, sequence_length, num_kv_heads * head_dim]
        # v shape: [batch_size, sequence_length, num_kv_heads * head_dim]
        bs = x.shape[0]
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # shape: [batch_size, sequence_length, num_heads, head_dim],
        # [batch_size, sequence_length, num_kv_heads, head_dim],
        # [batch_size, sequence_length, num_kv_heads, head_dim]
        q = q.view(bs, -1, self.num_heads, self.head_dim)
        k = k.view(bs, -1, self.num_kv_heads, self.head_dim)
        v = v.view(bs, -1, self.num_kv_heads, self.head_dim)
        # apply QK norm
        q, k = self.q_norm(q), self.k_norm(k)
        # apply rotary positional embeding
        q, k = self.rope(q, k, position_ids)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # shape: [batch_size, num_heads, seqeuence_length, head_dim]
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=self.num_heads != self.num_kv_heads
        )
        # shape: [batch_size, sequence_length, num_heads * head_dim]
        o = o.transpose(1, 2).view(bs, -1, self.num_heads * self.head_dim)
        # apply output projection, shape: [batch_size, sequence_length, hidden_size]
        return self.o_proj(o)


class Qwen3MLP(nn.Module):
    """
    Per device MLP layer for Qwen3.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: [batch_size, sequence_length, hidden_size]
        Returns:
            output: [batch_size, sequence_length, hidden_size]
        """
        # shape: [batch_size, sequence_length, intermediate_size]
        y1 = self.gate_proj(x)
        y2 = self.up_proj(x)
        # shape: [batch_size, sequence_length, intermediate_size]
        x = F.silu(y1) * y2
        # shape: [batch_size, sequence_length, hidden_size]
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        we can simply do
        x = x + self.self_attn(positions, norm(hiddent_states))
        x = x + self.mlp(norm(x))

        But we can see that add is one kernel and norm is another kernel. Here we fuse the add and norm into one kernel.

        Arguments:
            x: [batch_size, sequence_length, hidden_size]
            position_ids: [batch_size, sequence_length]
            residual: [batch_size, sequence_length, hidden_size]
        Returns:
            output: [batch_size, sequence_length, hidden_size]
        """
        x = x + self.self_attn(self.input_layernorm(x), position_ids)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


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
            x: [batch_size, sequence_length]
            position_ids: [batch_size, sequence_length]
        Returns:
            output: [batch_size, sequence_length, hidden_size]
        """
        # shape: [batch_size, sequence_length, hidden_size]
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x, position_ids)
        # post-norm
        x = self.norm(x)
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

    def get_embed_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: [batch_size, sequence_length]
        Returns:
            output: [batch_size, sequence_length, hidden_size]
        """
        return self.model.embed_tokens(x)

    def get_model_layers(self) -> List[nn.Module]:
        return self.model.layers

    @staticmethod
    def get_layers_for_scaling(
        module: Qwen3Block, inputs: Dict[str, Any], module_kwargs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        layers = []

        # attention input
        layers.append(
            {
                "prev_op": module.input_layernorm,
                "layers": [
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                "inp": inputs["self_attn.q_proj"],
                "module2inspect": module.self_attn,
                "module_kwargs": module_kwargs,
            }
        )
        # skip scaling o_proj when GQA is enabled
        # attention output: https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.num_heads == module.self_attn.num_kv_heads:
            layers.append(
                {
                    "prev_op": module.self_attn.v_proj,
                    "layers": [
                        module.self_attn.o_proj,
                    ],
                    "inp": inputs["self_attn.o_proj"],
                }
            )

        # MLP up projection
        layers.append(
            {
                "prev_op": module.post_attention_layernorm,
                "layers": [
                    module.mlp.up_proj,
                    module.mlp.gate_proj,
                ],
                "inp": inputs["mlp.up_proj"],
                "module2inspect": module.mlp,
            }
        )

        # MLP down projection
        layers.append(
            {
                "prev_op": module.mlp.up_proj,
                "layers": [
                    module.mlp.down_proj,
                ],
                "inp": inputs["mlp.down_proj"],
            }
        )

        return layers

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: [batch_size, sequence_length]
            position_ids: [batch_size, sequence_length]
        Returns:
            output: [batch_size, sequence_length, hidden_size]
        """
        return self.model(x, position_ids)

    def compute_loss(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: [batch_size, sequence_length]
            labels: [batch_size, sequence_length]
        """
        seq_len = x.shape[1]
        position_ids = (
            torch.arange(seq_len, dtype=torch.int32, device=x.device)
            .unsqueeze(0)
            .cuda(non_blocking=True)
        )
        hidden_states = self.model(x, position_ids)
        logits = self.lm_head(hidden_states)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return loss
