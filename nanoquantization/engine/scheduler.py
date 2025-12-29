from collections import deque
from typing import List, Tuple
from nanoquantization.config import Config
from nanoquantization.engine.block_manager import BlockManager
from nanoquantization.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.block_manager = BlockManager(
            config.num_kv_cache_blocks, config.kv_cache_block_size
        )
        self.eos_token_id = config.eos_token_id
        self.running: deque[Sequence] = deque[Sequence]()
        self.waiting: deque[Sequence] = deque[Sequence]()

    def add_sequence(self, sequence: Sequence) -> None:
        # add sequence for prefilling
        self.waiting.append(sequence)

    def is_done(self) -> bool:
        return len(self.waiting) == 0 and len(self.running) == 0

    def schedule(self) -> Tuple[List[Sequence], bool]:
        """
        schedule sequences for prefilling or decoding from the waiting or running queue.
        return sequences that will be running by the model runner
        """
        # first check prefilling
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if len(
                seq
            ) + num_batched_tokens > self.max_num_batched_tokens or not self.block_manager.can_allocate(
                seq
            ):
                break
            num_seqs += 1
            # allocate blocks for the sequence
            self.block_manager.allocate(seq)
            # change the status of the seq
            seq.status = SequenceStatus.RUNNING
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            # change the queues status
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True
        # then check decoding
        while self.running and num_seqs < self.max_num_seqs:
            # see if we can schedule this seq for decoding
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(
                        self.running.pop()
                    )  # remove the right most sequence from running queue
                else:
                    self.preempt(seq)
                    break
            else:
                # if no break
                self.block_manager.may_append(seq)
                num_seqs += 1
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)
        self.block_manager.deallocate(seq)

    def post_processing(self, sequences: List[Sequence], token_ids: List[int]) -> None:
        """
        token_ids: the autoregressive token generated for each sequence in sequences
        check whether the generation has reached the eos token or over the max generation length
        """
        for seq, token_id in zip(sequences, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos_token_id) or len(
                seq
            ) >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
