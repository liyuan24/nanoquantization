import torch
from torch import nn


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        # shape: [n_seqs, vocab_size]
        logits = logits.float().div_(temperatures.unsqueeze(1))
        probs = logits.softmax(dim=-1)
        # Gumbel-max sampling
        # exponential distribution
        # median = ln(2) / lambda ~= 0.693 / lambda
        # 95th Percentile = ln(20) / lambda ~= 3.00 / lambda
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        # shape: [n_seqs,]
        return sample_tokens
