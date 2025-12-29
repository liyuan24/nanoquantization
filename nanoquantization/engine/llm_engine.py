import atexit
from tqdm import tqdm
from dataclasses import fields
from typing import List
from nanoquantization.engine.scheduler import Scheduler
from nanoquantization.engine.sequence import Sequence
from nanoquantization.sampling_params import SamplingParams
from transformers import AutoTokenizer
from nanoquantization.config import Config
import multiprocessing as mp

from nanoquantization.engine.model_runner import ModelRunner


class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = [field.name for field in fields(Config)]
        config_values = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_values)
        self.ps = []
        self.events = []
        # create processes for each worker
        ctx = mp.get_context("spawn")  # each worker get a fresh process
        for rank in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            p = ctx.Process(target=ModelRunner, args=(config, rank, event))
            p.start()
            self.ps.append(p)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        config.eos_token_id = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(
            self.exit
        )  # register exit handler so that it will be called when the program exits

    def exit(self) -> None:
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(
        self, prompt: str | List[int], sampling_params: SamplingParams
    ) -> None:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        self.scheduler.add_sequence(Sequence(prompt, sampling_params))

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: SamplingParams | List[SamplingParams],
    ) -> List[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sampling_param in zip(prompts, sampling_params):
            self.add_request(prompt, sampling_param)
        outputs = {}
        while not self.scheduler.is_done():
            sequences, is_prefill = self.scheduler.schedule()
            token_ids = self.model_runner.call("run", sequences, is_prefill)
            self.scheduler.post_processing(sequences, token_ids)
            for seq in sequences:
                if seq.is_finished:
                    outputs[seq.id] = seq.completion_token_ids
                    pbar.update(1)
        pbar.close()
        return [self.tokenizer.decode(outputs[i]) for i in sorted(outputs.keys())]
