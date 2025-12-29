from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 64

    def __post_init__(self):
        assert self.temperature >= 1e-10, "greedy sampling is not allowed"
