from dataclasses import dataclass
import os
from typing import Optional
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_seqs: int = 512
    max_num_batched_tokens: int = 16384
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    hf_config: Optional[AutoConfig] = None
    kv_cache_block_size: int = 256
    num_kv_cache_blocks: int = -1
    eos_token_id: int = -1
    max_model_len: int = 4096

    def __post_init__(self):
        os.path.isdir(self.model)
        assert self.kv_cache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings
        )
        assert self.max_num_batched_tokens >= self.max_model_len
