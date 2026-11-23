"""SGLang's view of an external runner's full-attention KV bytes.

Host copies (HiCache) and transfers (PD) read KV through SGLang's pool
interface. ``PluginKVView`` is that interface over the bytes the runner's
``KVLayout`` lists: one uint8 row of ``slot_bytes`` per slot and layer, at the
layout's addresses, with no copy and no allocation. Those consumers move bytes
and never interpret them, so a byte view serves any per-slot layout; SGLang's
attention never reads it, since the runner attends itself.

The view is valid for the layout's epoch: the runner keeps the addresses fixed
until SGLang ends the epoch, which in this release it never does.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_runner_plugins.api import KVLayout


class _DeviceBytes:
    """A runner's device memory as a CUDA array interface, for a zero-copy view."""

    def __init__(self, *, ptr: int, num_bytes: int):
        self.__cuda_array_interface__ = {
            "shape": (num_bytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def device_bytes(*, ptr: int, rows: int, row_bytes: int, device) -> torch.Tensor:
    """[rows, 1, row_bytes] uint8 over the runner's memory at ``ptr``."""
    flat = torch.as_tensor(
        _DeviceBytes(ptr=ptr, num_bytes=rows * row_bytes), device=device
    )
    if flat.data_ptr() != ptr:
        raise RuntimeError("the KV byte view does not alias the runner's memory")
    return flat.view(rows, 1, row_bytes)


def kv_view_refusal(layout: KVLayout) -> Optional[str]:
    """Why SGLang cannot view ``layout``'s KV as slot rows; None when it can."""
    layers = layout.full_attention
    if not layers or layout.sub_pools:
        return "KV beyond full-attention layers"
    if (
        len({layer.k_slot_bytes for layer in layers}) != 1
        or len({layer.v_slot_bytes for layer in layers}) != 1
    ):
        return "full-attention layers with unequal slot sizes"
    return None


class PluginKVView(MHATokenToKVPool):
    """The runner's full-attention layers as an MHA pool of byte rows: one head
    of ``slot_bytes`` uint8 per slot, K and V per layer, at the layout's
    addresses."""

    def __init__(self, *, layout: KVLayout, device: str):
        refusal = kv_view_refusal(layout)
        if refusal is not None:
            raise ValueError(f"SGLang cannot view the runner's KV: {refusal}")
        self._layers = layout.full_attention
        first = self._layers[0]
        super().__init__(
            size=layout.num_slots,
            page_size=layout.page_size,
            dtype=torch.uint8,
            head_num=1,
            head_dim=first.k_slot_bytes,
            v_head_dim=first.v_slot_bytes,
            layer_num=len(self._layers),
            device=device,
            enable_memory_saver=False,
            enable_alt_stream=False,
            kv_cache_layout="nhd",
        )
        self.model_layer_ids = tuple(layer.layer_id for layer in self._layers)

    def _create_buffers_normal(self):
        rows = self.size + self.page_size
        self.k_buffer = [
            device_bytes(
                ptr=layer.k_ptr,
                rows=rows,
                row_bytes=layer.k_slot_bytes,
                device=self.device,
            )
            for layer in self._layers
        ]
        self.v_buffer = [
            device_bytes(
                ptr=layer.v_ptr,
                rows=rows,
                row_bytes=layer.v_slot_bytes,
                device=self.device,
            )
            for layer in self._layers
        ]

    def wait_for_loads(self) -> None:
        """Order the runner stream after every layer's pending HiCache load: the
        runner reads all its layers inside execute(), so SGLang waits for the
        last layer's load (loads run in layer order) before it."""
        counter = self.layer_transfer_counter
        if counter is not None:
            counter.wait_until(self.layer_num - 1)

    def _attention_is_the_runners(self, *args, **kwargs):
        raise RuntimeError(
            "The model runner plugin attends over its own KV; SGLang only copies it"
        )

    get_key_buffer = _attention_is_the_runners
    get_value_buffer = _attention_is_the_runners
    get_kv_buffer = _attention_is_the_runners
    set_kv_buffer = _attention_is_the_runners
