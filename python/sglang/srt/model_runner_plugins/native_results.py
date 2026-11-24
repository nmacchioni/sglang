"""Typed lifecycle results for the built-in runner, read off its live workers.

The conversion from native worker and runner fields to the plugin contract
lives here, so callers that migrate to the typed results stop reading those
fields themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.model_runner_plugins.api import (
    AdmissionLimits,
    KVLayout,
    LoadResult,
    MemoryNeeds,
    ModelInfo,
)

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker

_GIB = 1 << 30


def native_load_result(
    *, tp_worker: TpModelWorker, draft_worker: Optional[object]
) -> LoadResult:
    model_config = tp_worker.model_config
    preloaded = tp_worker.preloaded_weights_bytes
    if draft_worker is not None:
        preloaded += draft_worker.preloaded_weights_bytes
    return LoadResult(
        model_info=ModelInfo(
            context_len=model_config.context_len,
            vocab_size=model_config.vocab_size,
            is_generation=model_config.is_generation,
        ),
        memory=MemoryNeeds(
            # Measured as a GB difference in free memory; noise can dip below 0.
            weight_bytes=max(
                0, int(tp_worker.model_runner.weight_load_mem_usage * _GIB)
            ),
            preloaded_weight_bytes=preloaded,
        ),
    )


def native_kv_layout(*, tp_worker: TpModelWorker, with_limits: bool) -> KVLayout:
    from sglang.srt.model_runner_plugins.native_kv import NativeKV

    model_runner = tp_worker.model_runner
    kv = NativeKV(model_runner)
    return KVLayout(
        page_size=model_runner.token_to_kv_pool.page_size,
        num_slots=model_runner.token_to_kv_pool.size,
        full_attention=kv.full_attention(),
        sub_pools=kv.sub_pools(),
        limits=admission_limits(tp_worker) if with_limits else None,
    )


def admission_limits(tp_worker: TpModelWorker) -> AdmissionLimits:
    # get_worker_info is the native source of these limits (and logs the
    # short-pool warning once); this reads it instead of recomputing.
    info = tp_worker.get_worker_info()
    return AdmissionLimits(
        max_total_num_tokens=info[0],
        max_running_requests=info[2],
        max_req_len=info[4],
        max_req_input_len=info[5],
    )
