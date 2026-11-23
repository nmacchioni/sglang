"""Startup of an installed (external) model runner on one rank.

The plugin loads its own weights and backs its own KV bytes. SGLang keeps what
it decides: its ``TpModelWorker`` (tokenizer, seed, sampling) runs over a
``PluginOwnedModelRunner`` that loads no model, and SGLang's request and slot
allocators are sized from the slot count of the plugin's ``KVLayout``, so page
and slot IDs, the prefix cache and admission stay SGLang's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import msgspec

from sglang.srt.model_runner_plugins.api import (
    Capacity,
    GraphBuckets,
    KVLayout,
    LoadContext,
    LoadResult,
    ModelRunnerExecutor,
    RuntimeConfig,
)
from sglang.srt.model_runner_plugins.native import NativeStartupHost
from sglang.srt.model_runner_plugins.native_results import admission_limits

if TYPE_CHECKING:
    from sglang.srt.model_runner_plugins.external_worker import (
        ExternalTpModelWorker,
    )


class ExternalRunner:
    """Drives an external plugin's lifecycle beside SGLang's own startup phases.

    Each phase calls the plugin first, then the scheduler phase that builds
    SGLang's side over the plugin's result.
    """

    def __init__(
        self,
        *,
        name: str,
        plugin: ModelRunnerExecutor,
        host: NativeStartupHost,
        agree_min: Optional[Callable[[int], int]] = None,
    ):
        self.name = name
        self.plugin = plugin
        self._host = host
        # The minimum of a value over the ranks that must size alike (the TP
        # group): a startup collective, as native memory sizing runs one.
        self._agree_min = agree_min or _min_over_tp_group
        self.load_result: Optional[LoadResult] = None
        # The layout SGLang sizes against: the plugin's, at the agreed slot count.
        self.layout: Optional[KVLayout] = None
        self._allocated_slots = 0

    def load(self, ctx: LoadContext) -> LoadResult:
        self.load_result = self._checked("load", self.plugin.load(ctx), LoadResult)
        self._host.load_model_workers()
        return self.load_result

    def allocate(self, cfg: RuntimeConfig) -> KVLayout:
        layout = self._checked("allocate", self.plugin.allocate(cfg), KVLayout)
        if layout.page_size != cfg.page_size:
            raise ValueError(
                f"Model runner {self.name!r} allocated KV with page size "
                f"{layout.page_size}; SGLang pages are {cfg.page_size} slots"
            )
        self._allocated_slots = layout.num_slots
        agreed = self._agree_min(layout.num_slots)
        if agreed != layout.num_slots:
            # Each rank sized its backing from its own free memory; SGLang's
            # replicated allocators hand out the same IDs on every rank, so they
            # use the smallest backing, which every rank serves.
            agreed -= agreed % layout.page_size
            layout = msgspec.structs.replace(layout, num_slots=agreed)
        self.layout = layout
        self._host.alloc_model_memory_pools()
        return layout

    def capture(self, buckets: GraphBuckets) -> None:
        self.plugin.capture(buckets)
        self._host.capture_model_graphs()

    def finalize(self, capacity: Capacity) -> KVLayout:
        layout = self._checked("finalize", self.plugin.finalize(capacity), KVLayout)
        if layout.num_slots != self._allocated_slots:
            # SGLang's allocators were sized at allocate(); they do not resize yet.
            raise ValueError(
                f"Model runner {self.name!r} resized its KV from "
                f"{self._allocated_slots} to {layout.num_slots} slots in "
                "finalize(); external runners keep their allocated size."
            )
        self._host.finalize_model_capacity()
        # Admission is SGLang's, computed from the pools it sized.
        return msgspec.structs.replace(
            layout,
            num_slots=self.layout.num_slots,
            limits=admission_limits(self._host.tp_worker),
        )

    def close(self) -> None:
        self.plugin.close()

    def create_tp_worker(self, **worker_kwargs) -> ExternalTpModelWorker:
        from sglang.srt.model_runner_plugins.external_worker import (
            ExternalTpModelWorker,
        )

        return ExternalTpModelWorker(runner=self, **worker_kwargs)

    def _checked(self, method: str, result, expected: type):
        if not isinstance(result, expected):
            raise TypeError(
                f"Model runner {self.name!r} {method}() returned "
                f"{type(result).__name__}; expected {expected.__name__}."
            )
        return result


def _min_over_tp_group(value: int) -> int:
    import torch

    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group
    if group.world_size == 1:
        return value
    tensor = torch.tensor([value], dtype=torch.int64)
    torch.distributed.all_reduce(
        tensor, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    return int(tensor.item())
