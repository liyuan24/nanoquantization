from typing import Optional
from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: Optional[int] = None
    max_seqlen_k: Optional[int] = None
    block_tables: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    slot_mapping: Optional[torch.Tensor] = None


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(
    is_prefill: bool,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    block_tables: Optional[torch.Tensor] = None,
    context_lens: Optional[torch.Tensor] = None,
    slot_mapping: Optional[torch.Tensor] = None,
) -> None:
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        block_tables,
        context_lens,
        slot_mapping,
    )


def reset_context() -> None:
    global _CONTEXT
    _CONTEXT = Context()
