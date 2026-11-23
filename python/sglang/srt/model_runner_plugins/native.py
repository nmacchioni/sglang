"""The built-in ``sglang`` model runner plugin.

It declares everything SGLang's own runner supports. Its lifecycle delegates to
the scheduler's native startup phases, in the native order; execution stays on
the direct path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

from sglang.srt.model_runner_plugins.api import (
    MODEL_RUNNER_API_VERSION,
    Capacity,
    Feature,
    GraphBuckets,
    KVLayout,
    LoadContext,
    LoadResult,
    RefusedCombination,
    ResolvedConfig,
    RunnerCapabilities,
    RuntimeConfig,
)
from sglang.srt.model_runner_plugins.native_results import (
    native_kv_layout,
    native_load_result,
)

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker

_NATIVE_CAPABILITIES = RunnerCapabilities.all_supported(
    refused_combinations=(
        # The tokenizer manager refuses the same pair for the Python frontend.
        RefusedCombination(
            features=(Feature.INPUT_EMBEDS, Feature.RADIX_CACHE),
            hint="pass --disable-radix-cache",
        ),
    ),
)


class NativeStartupHost(Protocol):
    """Today's startup, split at the lifecycle boundaries (the ``Scheduler``).

    Target and draft workers stay together here: native startup interleaves
    their loading, pool allocation, attention setup and graph capture.
    """

    tp_worker: TpModelWorker
    # A speculative draft worker, or None.
    draft_worker: object

    def load_model_workers(self) -> None: ...

    def alloc_model_memory_pools(self) -> None: ...

    def capture_model_graphs(self) -> None: ...

    def finalize_model_capacity(self) -> None: ...


class SGLangRunnerPlugin:
    api_version = MODEL_RUNNER_API_VERSION

    def __init__(self, host: Optional[NativeStartupHost] = None):
        # None when built for describe() alone (the launcher, admission).
        self._host = host

    def describe(self, cfg: ResolvedConfig) -> RunnerCapabilities:
        return _NATIVE_CAPABILITIES

    def load(self, ctx: LoadContext) -> LoadResult:
        host = self._require_host()
        host.load_model_workers()
        return native_load_result(
            tp_worker=host.tp_worker, draft_worker=host.draft_worker
        )

    def allocate(self, cfg: RuntimeConfig) -> KVLayout:
        # The native pool configurator sizes from the same published bags cfg
        # is built from, so delegating honors it.
        host = self._require_host()
        host.alloc_model_memory_pools()
        return native_kv_layout(tp_worker=host.tp_worker, with_limits=False)

    def capture(self, buckets: GraphBuckets) -> None:
        # Native derives its capture buckets from the same graph config.
        self._require_host().capture_model_graphs()

    def finalize(self, capacity: Capacity) -> KVLayout:
        # Resizes only when the post-capture backing is active. PP stages agree
        # on capacity inside this phase, at its native position.
        host = self._require_host()
        host.finalize_model_capacity()
        return native_kv_layout(tp_worker=host.tp_worker, with_limits=True)

    def close(self) -> None:
        # Native workers own no teardown beyond process exit; drop the handle.
        self._host = None

    def _require_host(self) -> NativeStartupHost:
        if self._host is None:
            raise RuntimeError(
                "The built-in model runner's lifecycle runs only inside the "
                "scheduler that created it (or after close())."
            )
        return self._host
