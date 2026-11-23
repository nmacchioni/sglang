"""Startup of an installed (external) model runner on one rank.

The plugin loads its own weights and backs its own KV bytes. SGLang keeps what
it decides: its ``TpModelWorker`` (tokenizer, seed, sampling) runs over a
``PluginOwnedModelRunner`` that loads no model, and SGLang's request and slot
allocators are sized from the slot count of the plugin's ``KVLayout``, so page
and slot IDs, the prefix cache and admission stay SGLang's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

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
        self, *, name: str, plugin: ModelRunnerExecutor, host: NativeStartupHost
    ):
        self.name = name
        self.plugin = plugin
        self._host = host
        self.load_result: Optional[LoadResult] = None
        self.layout: Optional[KVLayout] = None

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
        self.layout = layout
        self._host.alloc_model_memory_pools()
        return layout

    def capture(self, buckets: GraphBuckets) -> None:
        self.plugin.capture(buckets)
        self._host.capture_model_graphs()

    def finalize(self, capacity: Capacity) -> KVLayout:
        layout = self._checked("finalize", self.plugin.finalize(capacity), KVLayout)
        if layout.num_slots != self.layout.num_slots:
            # SGLang's allocators were sized at allocate(); they do not resize yet.
            raise ValueError(
                f"Model runner {self.name!r} resized its KV from "
                f"{self.layout.num_slots} to {layout.num_slots} slots in "
                "finalize(); external runners keep their allocated size."
            )
        self._host.finalize_model_capacity()
        # Admission is SGLang's, computed from the pools it sized.
        return msgspec.structs.replace(
            layout, limits=admission_limits(self._host.tp_worker)
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
