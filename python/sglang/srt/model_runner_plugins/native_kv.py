"""The built-in runner's KV pools as the API describes them: ``KVLayout``
sub-pools at startup and ``KVTables`` per step.

Described: slot-row MHA full attention, alone, beside a paged sliding-window
sub-pool (``SWAKVPool``), or beside recurrent state (``HybridLinearKVPool``).
Every other pool has no description yet, and the built-in runner runs such a
model's steps on its native forward, with no Step (``native_kv_refusal``).

Every table is a reference to what SGLang already keeps (the full-to-window
slot map, the request-to-state map) or to the step's own ForwardBatch fields,
so describing a step adds no copy; the window write slots are the iteration
plan's, which the attention backend reads from the same cache.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Optional

from sglang.srt.model_runner_plugins.api import (
    FullAttentionLayer,
    KVBuffer,
    KVKind,
    KVSubPool,
    StateTables,
    WindowTables,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


class NativeKV:
    """One model runner's pools, as described: built once, after startup."""

    def __init__(self, model_runner: ModelRunner):
        # Imported here: the launcher imports this package for describe() alone.
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        self._model_runner = model_runner
        pool = model_runner.token_to_kv_pool
        self.full_pool = pool
        self.window_pool = None
        self.state_pool = None
        if isinstance(pool, SWAKVPool):
            self.full_pool, self.window_pool = pool.full_kv_pool, pool.swa_kv_pool
            self._window_map = pool.full_to_swa_index_mapping
        elif isinstance(pool, HybridLinearKVPool):
            self.full_pool, self.state_pool = pool.full_kv_pool, pool.mamba_pool
        self.refusal = _refusal(model_runner=model_runner, pool=pool, native_kv=self)

    def full_attention(self) -> tuple[FullAttentionLayer, ...]:
        if self.refusal is not None:
            return ()
        return _mha_layers(self.full_pool)

    def sub_pools(self) -> tuple[KVSubPool, ...]:
        if self.refusal is not None:
            return ()
        pools = []
        if self.window_pool is not None:
            pools.append(
                KVSubPool(
                    kind=KVKind.SLIDING_WINDOW,
                    num_items=self.window_pool.size + self.window_pool.page_size,
                    buffers=tuple(
                        buffer
                        for layer in _mha_layers(self.window_pool)
                        for buffer in (
                            KVBuffer(
                                layer_id=layer.layer_id,
                                name="k",
                                ptr=layer.k_ptr,
                                item_bytes=layer.k_slot_bytes,
                            ),
                            KVBuffer(
                                layer_id=layer.layer_id,
                                name="v",
                                ptr=layer.v_ptr,
                                item_bytes=layer.v_slot_bytes,
                            ),
                        )
                    ),
                    window=self._model_runner.sliding_window_size,
                )
            )
        if self.state_pool is not None:
            pools.append(_state_sub_pool(self.state_pool))
        return tuple(pools)

    def window_tables(self, forward_batch: ForwardBatch) -> Optional[WindowTables]:
        if self.window_pool is None:
            return None
        from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind

        # The plan translates once per sub-pool; the backend reads the same ids.
        write_slots = self._model_runner.runner_ids.write_ids(
            forward_batch, kind=IdSpaceKind.SLIDING_WINDOW
        )
        if write_slots is None:
            raise RuntimeError("a sliding-window step without window write slots")
        return WindowTables(write_slots=write_slots, slot_map=self._window_map)

    def state_tables(self, forward_batch: ForwardBatch) -> Optional[StateTables]:
        if self.state_pool is None:
            return None
        req_pool = self._model_runner.req_to_token_pool
        return StateTables(
            req_to_state=req_pool.req_index_to_mamba_index_mapping,
            clear=forward_batch.mamba_clear_indices,
            copy_src=forward_batch.mamba_cow_src_indices,
            copy_dst=forward_batch.mamba_cow_dst_indices,
            checkpoint_slots=forward_batch.mamba_track_indices,
            checkpoint_mask=forward_batch.mamba_track_mask,
            checkpoint_lens=forward_batch.mamba_track_seqlens,
        )


def _refusal(*, model_runner, pool, native_kv: NativeKV) -> Optional[str]:
    """Why the pools have no description; None when they have one."""
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MHATokenToKVPool
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    if type(pool) not in (MHATokenToKVPool, SWAKVPool, HybridLinearKVPool):
        return f"KV pool {type(pool).__name__}"
    for sub in (native_kv.full_pool, native_kv.window_pool):
        if sub is not None and (type(sub) is not MHATokenToKVPool or sub.use_hnd):
            return f"KV pool {type(pool).__name__} over {type(sub).__name__}"
    if native_kv.window_pool is not None and native_kv._window_map is None:
        # The per-request window ring keeps no full-to-window map.
        return "a sliding window kept per request"
    if native_kv.state_pool is not None:
        return _state_refusal(model_runner=model_runner, pool=pool)
    return None


def _state_refusal(*, model_runner, pool) -> Optional[str]:
    allocator = model_runner.token_to_kv_pool_allocator
    req_pool = model_runner.req_to_token_pool
    state = allocator.state
    if pool.use_mla or state.ckpt_pool is not None or state.own_state is not None:
        return "recurrent state with quantized checkpoints or side state"
    if model_runner.runner_ids.state_ids(req_pool.state_sub_pool).write is not None:
        # State slots go through a virtual-to-physical table.
        return "remapped recurrent-state slots"
    ptrs, lens, items = pool.mamba_pool.get_contiguous_buf_infos()
    if any(length % item for length, item in zip(lens, items)):
        return "recurrent state not addressable per slot"
    return None


def _mha_layers(pool) -> tuple[FullAttentionLayer, ...]:
    return tuple(
        FullAttentionLayer(
            layer_id=layer_id,
            k_ptr=k.data_ptr(),
            v_ptr=v.data_ptr(),
            k_slot_bytes=k.stride(0) * k.element_size(),
            v_slot_bytes=v.stride(0) * v.element_size(),
        )
        for layer_id, k, v in zip(pool.model_layer_ids, pool.k_buffer, pool.v_buffer)
    )


def _state_sub_pool(mamba_pool) -> KVSubPool:
    # The buffers PD registers for transfer, in the same order.
    ptrs, lens, items = mamba_pool.get_contiguous_buf_infos()
    layer_ids = mamba_pool.get_state_layer_ids()
    seen = Counter()
    buffers = []
    for ptr, item, layer_id in zip(ptrs, items, layer_ids):
        buffers.append(
            KVBuffer(
                layer_id=layer_id,
                name=f"state{seen[layer_id]}",
                ptr=ptr,
                item_bytes=item,
            )
        )
        seen[layer_id] += 1
    num_items = {length // item for length, item in zip(lens, items)}
    if len(num_items) != 1:
        raise ValueError(f"recurrent-state buffers of unequal lengths: {num_items}")
    return KVSubPool(
        kind=KVKind.RECURRENT_STATE,
        num_items=num_items.pop(),
        buffers=tuple(buffers),
    )
