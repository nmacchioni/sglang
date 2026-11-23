"""Drive a model runner through startup in native order, and close it on failure.

SGLang calls ``load``, ``allocate``, ``capture`` and ``finalize`` in turn and
validates each typed result. An exception in any phase, a malformed result
included, closes the runner before it propagates.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import msgspec

from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.srt.model_runner_plugins.api import (
    Capacity,
    GraphBuckets,
    KVLayout,
    LoadContext,
    LoadResult,
    ModelRunnerLifecycle,
    RuntimeConfig,
)
from sglang.srt.model_runner_plugins.native import (
    NativeStartupHost,
    SGLangRunnerPlugin,
)
from sglang.srt.model_runner_plugins.registry import load_model_runner
from sglang.srt.model_runner_plugins.selection import (
    external_runner_unavailable_message,
)
from sglang.srt.runtime_context import (
    get_device,
    get_exec,
    get_model,
    get_parallel,
    get_schedule,
)

logger = logging.getLogger(__name__)


class StartupPhase(str, Enum):
    NEW = "new"
    LOADED = "loaded"
    ALLOCATED = "allocated"
    CAPTURED = "captured"
    FINALIZED = "finalized"
    CLOSED = "closed"


class StartupInputs(msgspec.Struct, frozen=True, kw_only=True):
    """What SGLang passes to each startup phase."""

    load: LoadContext
    runtime: RuntimeConfig
    buckets: GraphBuckets
    capacity: Capacity

    @classmethod
    def from_published(cls) -> StartupInputs:
        parallel = get_parallel()
        schedule = get_schedule()
        graphs = get_exec().graph.cuda_graph_config
        return cls(
            load=LoadContext(
                model_path=get_model().model_path,
                device=get_device().device,
                gpu_id=get_device().gpu_id,
                tp_rank=parallel.tp_rank,
                tp_size=parallel.tp_size,
                pp_rank=parallel.pp_rank,
                pp_size=parallel.pp_size,
            ),
            runtime=RuntimeConfig(
                page_size=schedule.page_size,
                mem_fraction_static=schedule.mem_fraction_static,
                max_total_tokens=schedule.max_total_tokens,
                max_running_requests=schedule.max_running_requests,
            ),
            buckets=GraphBuckets(
                decode=_buckets(graphs, Phase.DECODE),
                prefill=_buckets(graphs, Phase.PREFILL),
            ),
            capacity=Capacity(
                max_total_tokens=schedule.max_total_tokens,
                max_running_requests=schedule.max_running_requests,
            ),
        )


class ModelRunnerStartup:
    """One rank's runner lifecycle: startup in order, then ``close`` once."""

    @classmethod
    def maybe_create(
        cls, *, native_host: NativeStartupHost
    ) -> Optional[ModelRunnerStartup]:
        """None without ``--model-runner``, so the default path runs no lifecycle."""
        name = get_exec().kernel.model_runner
        if name is None:
            return None
        if use_mlx():
            # The MLX worker keeps its own startup path in v1.
            return None
        if not load_model_runner(name).is_native:
            raise RuntimeError(external_runner_unavailable_message(name))
        return cls(runner_name=name, plugin=SGLangRunnerPlugin(host=native_host))

    def __init__(self, *, runner_name: str, plugin: ModelRunnerLifecycle):
        self.runner_name = runner_name
        self._plugin = plugin
        self.phase = StartupPhase.NEW
        self.load_result: Optional[LoadResult] = None
        self.layout: Optional[KVLayout] = None

    def run(self, inputs: StartupInputs) -> KVLayout:
        """Run every startup phase; the final layout carries admission limits."""
        if self.phase is not StartupPhase.NEW:
            raise RuntimeError(f"startup already ran (phase {self.phase.value})")
        try:
            self.load_result = self._checked(
                "load", self._plugin.load(inputs.load), LoadResult
            )
            self.phase = StartupPhase.LOADED
            allocated = self._checked(
                "allocate", self._plugin.allocate(inputs.runtime), KVLayout
            )
            self.layout = allocated
            self.phase = StartupPhase.ALLOCATED
            self._plugin.capture(inputs.buckets)
            self.phase = StartupPhase.CAPTURED
            final = self._checked(
                "finalize", self._plugin.finalize(inputs.capacity), KVLayout
            )
            _check_final_layout(allocated, final)
            self.layout = final
            self.phase = StartupPhase.FINALIZED
        except BaseException:
            self._close_after_failure()
            raise
        logger.info(
            "Model runner %r started: %d KV slots, %d full-attention layers, %s",
            self.runner_name,
            final.num_slots,
            len(final.full_attention),
            final.limits,
        )
        return final

    def close(self) -> None:
        """Close the runner; a no-op before ``load`` or when already closed."""
        if self.phase in (StartupPhase.NEW, StartupPhase.CLOSED):
            self.phase = StartupPhase.CLOSED
            return
        self.phase = StartupPhase.CLOSED
        self._plugin.close()

    def _close_after_failure(self) -> None:
        # load() may have created state before raising, so close from LOADED on.
        if self.phase is StartupPhase.NEW:
            self.phase = StartupPhase.LOADED
        try:
            self.close()
        except Exception:
            logger.exception(
                "Model runner %r failed to close after a startup failure",
                self.runner_name,
            )

    def _checked(self, method: str, result, expected: type):
        if not isinstance(result, expected):
            raise TypeError(
                f"Model runner {self.runner_name!r} {method}() returned "
                f"{type(result).__name__}; expected {expected.__name__}."
            )
        return result


def _check_final_layout(allocated: KVLayout, final: KVLayout) -> None:
    if final.limits is None:
        raise ValueError("finalize() must return a KVLayout with limits set")
    if final.epoch != allocated.epoch or final.page_size != allocated.page_size:
        raise ValueError(
            "finalize() changed the KV epoch or page size: "
            f"{allocated.epoch}/{allocated.page_size} -> "
            f"{final.epoch}/{final.page_size}"
        )
    if final.full_attention != allocated.full_attention:
        # Sizes may change in finalize; addresses and strides may not.
        raise ValueError("finalize() moved full-attention KV layers")


def _buckets(graphs, phase: str) -> tuple[int, ...]:
    if graphs is None:
        return ()
    config = graphs[phase]
    if config.backend == Backend.DISABLED or not config.bs:
        return ()
    return tuple(config.bs)
