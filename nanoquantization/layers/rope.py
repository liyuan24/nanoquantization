import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(
        self, head_dim: int, max_position_embeddings: int, base: float = 10000
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        t = torch.arange(max_position_embeddings, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        # unsqueeze to [max_position_embeddings, 1, head_dim]
        cos_sin_cache = torch.concat([cos, sin], dim=-1).unsqueeze_(1)
        # no need to store it in state_dict
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

    def apply_rope(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x1, x2 = torch.chunk(x.float(), 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.concat([y1, y2], dim=-1).to(x.dtype)

    @torch.compile
    def forward(
        self, query: torch.Tensor, key: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [total_tokens, num_heads, head_dim]
        position_ids: [total_tokens]
        output: [total_tokens, num_heads, head_dim]
        """
        cos_sin = self.cos_sin_cache[position_ids]
        cos, sin = torch.chunk(cos_sin, 2, dim=-1)
        query = self.apply_rope(query, cos, sin)
        key = self.apply_rope(key, cos, sin)
        return query, key
