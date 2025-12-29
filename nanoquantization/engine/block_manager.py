from collections import deque
from typing import Dict, List
from nanoquantization.engine.sequence import Sequence
import xxhash
import numpy as np


class Block:
    def __init__(self, block_id: int):
        # physical block id
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1  # prefix hash and start with invalid hash
        self.token_ids = []

    def update(self, token_ids: List[int], hash: int):
        self.token_ids = token_ids
        self.hash = hash

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_kv_cache_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_kv_cache_blocks)]
        # we need it as a queue because we want to get the first free block
        self.free_blocks: deque[int] = deque[int](range(num_kv_cache_blocks))
        self.used_blocks: set[int] = set[int]()
        self.hash_to_block_id: Dict[int, int] = dict[int, int]()

    @classmethod
    def compute_hash(cls, token_ids: List[int], prefix: int) -> int:
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # for prefill to check whether there is enough space to allocate for this sequence
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_blocks) >= seq.num_blocks

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "no existing tokens in this block"
        block.reset()
        self.free_blocks.remove(block_id)
        self.used_blocks.add(block_id)
        return block

    def allocate(self, seq: Sequence) -> None:
        """
        Loop through block of token ids in sequence.
        For each, first check whether those token_ids have already been cached in some physical block.
        If not, allocate a new physical block.
        If yes, check if the block is currently in use.
            If yes, increment the reference count.
            If no, update the metadata of the physical block.
        Update the block table with the physical block id.
        """
        assert not seq.block_table, "Block table should be empty before allocation"
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = (
                self.compute_hash(token_ids, h)
                if len(token_ids) == self.block_size
                else -1
            )
            # see if there is already a block that has the same prefix hash
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_blocks[0]  # get the first free block
                block = self._allocate_block(block_id)
            else:
                # the token ids have already been cached
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_blocks:
                    # the block is currently in use
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # when the block is deallocated before
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(token_ids, h)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def can_append(self, seq: Sequence) -> bool:
        """
        If the new token(lask token) in sequence needs a new physical block, need to check whether there is at least one free block.
        If not need a new physical block, just reuse the last physical block of the sequence.
        """
        need_new_block = (len(seq) % self.block_size) == 1
        return len(self.free_blocks) >= need_new_block

    def may_append(self, seq: Sequence) -> None:
        """
        Allocate for the new token.
        If the new token needs a new physical block, allocate a new physical block. Otherwise use the last physical block.
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            # need a new physical block
            block_id = self.free_blocks[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # this is the last token of the block, need to update the hash
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix_hash = (
                self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            )
            h = self.compute_hash(token_ids, prefix_hash)
            last_block.update(token_ids, h)
            self.hash_to_block_id[h] = block_table[-1]
        else:
            # do nothing
            assert last_block.hash == -1

    def deallocate(self, seq: Sequence) -> None:
        block_table = seq.block_table
        for block_id in block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def _deallocate_block(self, block_id: int) -> None:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "reference count should be 0 when deallocating"
        self.free_blocks.append(block_id)
        self.used_blocks.remove(block_id)
