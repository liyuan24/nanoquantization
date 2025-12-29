from typing import List
from copy import copy
from enum import Enum, auto
from itertools import count

from nanoquantization.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    # start from 0
    counter = count()
    block_size = 256

    def __init__(
        self, token_ids: List[int], sampling_params: SamplingParams = SamplingParams()
    ):
        self.id = next(Sequence.counter)
        self.token_ids = copy(token_ids)
        self.status = SequenceStatus.WAITING
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.last_token_id = token_ids[-1]
        self.num_cached_tokens = 0
        self.block_table = []  # paged attention
        self.cached_tokens = 0
        self.temperature = sampling_params.temperature
        self.ignore_eos = sampling_params.ignore_eos
        self.max_tokens = sampling_params.max_tokens

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, index: int):
        return self.token_ids[index]

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def num_last_block_tokens(self) -> int:
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    @property
    def num_cached_blocks(self) -> int:
        return self.num_cached_tokens // self.block_size

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def completion_token_ids(self) -> List[int]:
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def prompt_token_ids(self) -> List[int]:
        return self.token_ids[: self.num_prompt_tokens]

    def block(self, i: int) -> List[int]:
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)
        self.num_tokens += 1
        self.last_token_id = token_id
