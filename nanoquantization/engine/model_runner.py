from multiprocessing import Event
import pickle
from typing import Any, List, Tuple
from nanoquantization.config import Config
from nanoquantization.engine.sequence import Sequence
from nanoquantization.models.sampler import Sampler
from nanoquantization.context import reset_context, set_context
import torch
import torch.distributed as dist
from nanoquantization.models.qwen3 import Qwen3ForCausalLM
from nanoquantization.utils.loader import load_model
from multiprocessing.shared_memory import SharedMemory


class ModelRunner:
    """
    Each device has its own model runner
    """

    def __init__(self, config: Config, rank: int, event: Event | List[Event]):
        self.config = config
        hf_config = config.hf_config
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://localhost:2333",
            rank=rank,
            world_size=self.world_size,
        )
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(
            num_hidden_layers=hf_config.num_hidden_layers,
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            total_num_heads=hf_config.num_attention_heads,
            total_num_kv_heads=hf_config.num_key_value_heads,
            max_position_embeddings=hf_config.max_position_embeddings,
            intermediate_size=hf_config.intermediate_size,
            head_dim=hf_config.head_dim,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            rope_theta=hf_config.rope_theta,
            rms_norm_eps=hf_config.rms_norm_eps,
        )
        self.sampler = Sampler()
        load_model(self.model, config.model)
        self.warmup_model()
        self.allocate_kv_cache()
        self.event = event
        if self.world_size > 1:
            if self.rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)  # 1MB
                dist.barrier()
            else:
                dist.barrier()  # use barrier to make sure the shared memory is already created in rank 0
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def loop(self) -> None:
        while True:
            method, args = self.read_shm()
            self.call(method, *args)
            if method == "exit":
                break

    def read_shm(self):
        self.event.wait()
        n = int.from_bytes(self.shm.buf[:4], byteorder="little")
        method, *args = pickle.loads(self.shm.buf[4 : 4 + n])
        self.event.clear()  # reset the signal so that wait() can work again
        return method, args

    def write_shm(self, method: str, *args: Any) -> None:
        assert (
            self.world_size > 1 and self.rank == 0
        ), "only rank 0 can write to shared memory"
        data = pickle.dumps([method, *args])
        n = len(data)
        self.shm.buf[:4] = n.to_bytes(4, byteorder="little")
        self.shm.buf[4 : 4 + n] = data
        # notify all the other ranks to read the shared memory
        for event in self.event:
            event.set()

    def call(self, method: str, *args: Any) -> Any:
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method, *args)
        method = getattr(self, method)
        return method(*args)

    def exit(self) -> None:
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def warmup_model(self) -> None:
        """
        warmup model to get the memory stats
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_num_seqs = self.config.max_num_seqs
        num_seqs = min(
            max_num_batched_tokens // self.config.max_model_len, max_num_seqs
        )
        seqs = [Sequence([0] * self.config.max_model_len) for _ in range(num_seqs)]
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self) -> None:
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        # This is the memory PyTorch explicitly knows it is using for data.
        # include the model weights + buffers(e.g., running stats in BatchNorm, if any)
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # used is roughly the model weights + buffers
        # peak - current is roughly the size of activation memory
        total_memory_available_for_kv_cache = (
            total * config.gpu_memory_utilization - used - (peak - current)
        )
        kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = hf_config.head_dim
        # number of tokens per block
        block_size = config.kv_cache_block_size
        # 2 is for k and v
        # for each token in each layer, the kv cache size in bytes kv_heads * head_dim * data_type_size(e.g bfloat16 is 2 bytes)
        # so this is the total size for each block across all layers
        kv_cache_block_bytes = (
            2
            * hf_config.num_hidden_layers
            * block_size
            * kv_heads
            * head_dim
            * hf_config.torch_dtype.itemsize
        )
        config.num_kv_cache_blocks = (
            int(total_memory_available_for_kv_cache) // kv_cache_block_bytes
        )
        assert config.num_kv_cache_blocks > 0
        self.kv_cache = torch.empty(
            2,
            hf_config.num_hidden_layers,
            config.num_kv_cache_blocks,
            block_size,
            kv_heads,
            head_dim,
        )
        layer_id = 0
        # example module name model.layers.24.self_attn.attn
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    @torch.inference_mode()
    def run_model(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, position_ids)
        logits = self.model.compute_logits(hidden_states)
        return logits

    def run(self, seqs: List[Sequence], is_prefill: bool) -> List[int]:
        input_ids, position_ids = (
            self.prepare_prefilling(seqs) if is_prefill else self.prepare_decoding(seqs)
        )
        temperatures = self.prepare_temperatures(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, position_ids)
        # shape: [n_seqs]
        token_ids = (
            self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        )
        reset_context()
        return token_ids

    def prepare_temperatures(self, seqs: List[Sequence]) -> torch.Tensor:
        temperatures = [seq.temperature for seq in seqs]
        return torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(
            non_blocking=True
        )

    def prepare_prefilling(
        self, seqs: List[Sequence]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        prepare the context for prefilling
        """
        input_ids = []
        position_ids = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq.token_ids[seq.num_cached_tokens :])
            position_ids.extend(range(seq.num_cached_tokens, seqlen))
            seqlen_q = seqlen - seq.num_cached_tokens  # prefix cache
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(max_seqlen_q, seqlen_q)
            max_seqlen_k = max(max_seqlen_k, seqlen_k)
            if not seq.block_table:  # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * seq.block_size
                if i == seq.num_blocks - 1:
                    end = start + seq.num_last_block_tokens
                else:
                    end = start + seq.block_size
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        position_ids = torch.tensor(
            position_ids, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            block_tables,
            None,
            slot_mapping,
        )
        return input_ids, position_ids

    def prepare_decoding(
        self, seqs: List[Sequence]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = []
        position_ids = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token_id)
            position_ids.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_size * seq.block_table[-1] + seq.num_last_block_tokens - 1
            )
        block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(
            non_blocking=True
        )
        position_ids = torch.tensor(
            position_ids, dtype=torch.int64, pin_memory=True
        ).cuda(non_blocking=True)
        context_lens = torch.tensor(
            context_lens, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        slot_mapping = torch.tensor(
            slot_mapping, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        set_context(
            is_prefill=False,
            block_tables=block_tables,
            context_lens=context_lens,
            slot_mapping=slot_mapping,
        )
        return input_ids, position_ids

    def prepare_block_tables(self, seqs: List[Sequence]) -> torch.Tensor:
        block_table_max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = []
        for seq in seqs:
            block_tables.append(
                seq.block_table + [-1] * (block_table_max_len - len(seq.block_table))
            )
        return torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(
            non_blocking=True
        )
