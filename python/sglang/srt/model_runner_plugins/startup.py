"""Drive a model runner through startup in native order, and close it on failure.

SGLang calls ``load``, ``allocate``, ``capture`` and ``finalize`` in turn and
validates each typed result. An exception in any phase, a malformed result
included, closes the runner before it propagates.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Callable, Optional

import msgspec

from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.srt.model_runner_plugins.api import (
    Capacity,
    GraphBuckets,
    GraphChoice,
    KVLayout,
    LoadContext,
    LoadResult,
    ModelRunnerExecutor,
    ModelRunnerLifecycle,
    ParallelLayout,
    RankGroup,
    RunnerCapabilities,
    RuntimeConfig,
)
from sglang.srt.model_runner_plugins.external import ExternalRunner
from sglang.srt.model_runner_plugins.native import NativeStartupHost, SGLangRunnerPlugin
from sglang.srt.model_runner_plugins.native_results import admission_limits
from sglang.srt.model_runner_plugins.registry import load_model_runner
from sglang.srt.model_runner_plugins.selection import describe_runner, resolved_config
from sglang.srt.runtime_context import (
    get_device,
    get_exec,
    get_model,
    get_parallel,
    get_schedule,
)
from sglang.srt.utils.device import is_confidential_compute

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_runner_plugins.step_forward import StepForward

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
                parallel=parallel_layout(),
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
        cls, *, native_host: NativeStartupHost, model_config: ModelConfig
    ) -> Optional[ModelRunnerStartup]:
        """None without ``--model-runner``, so the default path runs no lifecycle."""
        name = get_exec().kernel.model_runner
        if name is None:
            return None
        if use_mlx():
            # The MLX worker keeps its own startup path in v1.
            return None
        runner = load_model_runner(name)
        capabilities = describe_runner(runner, resolved_config(model_config))
        if runner.is_native:
            plugin = SGLangRunnerPlugin(host=native_host)
        else:
            if not isinstance(runner.plugin, ModelRunnerExecutor):
                raise TypeError(
                    f"Model runner {name!r} lacks the lifecycle and step methods "
                    "(load, allocate, capture, finalize, plan, execute, project, "
                    "release, close)."
                )
            plugin = ExternalRunner(name=name, plugin=runner.plugin, host=native_host)
        return cls(
            runner_name=name,
            plugin=plugin,
            capabilities=capabilities,
            settle=lambda final: settle_capacity(native_host, final),
        )

    def __init__(
        self,
        *,
        runner_name: str,
        plugin: ModelRunnerLifecycle,
        capabilities: RunnerCapabilities = RunnerCapabilities(),
        settle: Optional[Callable[[KVLayout], KVLayout]] = None,
    ):
        self.runner_name = runner_name
        self._plugin = plugin
        # SGLang's step after finalize: PP stages agree on capacity, then
        # admission limits are read off the settled pools. None leaves the
        # runner's own limits.
        self._settle = settle
        self._capabilities = capabilities
        # The buckets passed to capture(); SGLang replays external steps in them.
        self._buckets = GraphBuckets()
        self.phase = StartupPhase.NEW
        self.load_result: Optional[LoadResult] = None
        self.layout: Optional[KVLayout] = None
        self._step_forward: Optional[StepForward] = None

    @property
    def is_external(self) -> bool:
        return isinstance(self._plugin, ExternalRunner)

    def create_tp_worker(self, **worker_kwargs) -> TpModelWorker:
        """The target worker over an external runner (``is_external`` only)."""
        return self._plugin.create_tp_worker(**worker_kwargs)

    def step_forward(self, *, tp_worker: TpModelWorker) -> Optional[StepForward]:
        """The worker's forwards through the runner's steps, after startup; None
        when the built-in runner keeps every step on the direct path."""
        from sglang.srt.model_runner_plugins.native_kv import NativeKV
        from sglang.srt.model_runner_plugins.native_steps import (
            native_graph_choice,
            native_step_refusals,
        )
        from sglang.srt.model_runner_plugins.step_forward import StepForward

        model_runner = tp_worker.model_runner
        overlap = not get_schedule().disable_overlap_schedule
        if self.is_external:
            if overlap and is_confidential_compute():
                raise ValueError(
                    f"--model-runner {self.runner_name} cannot run overlap "
                    "scheduling with asynchronous host copies (confidential "
                    "computing); pass --disable-overlap-schedule"
                )
            executor, lowering, native_kv = self._plugin.plugin, None, None
            buckets = self._buckets
            choose_graph = lambda batch: _external_graph(batch, buckets=buckets)
        else:
            native_kv = NativeKV(model_runner)
            refusals = native_step_refusals(model_runner, kv_refusal=native_kv.refusal)
            if refusals:
                logger.info(
                    "Model runner %r keeps the direct path for every step: %s",
                    self.runner_name,
                    ", ".join(refusals),
                )
                return None
            executor, lowering = self._plugin, self._plugin.lowering
            choose_graph = lambda batch: native_graph_choice(
                model_runner=model_runner, forward_batch=batch
            )
        self._step_forward = StepForward(
            runner_name=self.runner_name,
            executor=executor,
            page_table=model_runner.req_to_token_pool.req_to_token,
            vocab_size=model_runner.model_config.vocab_size,
            native_prompt_logprobs=self._capabilities.native_prompt_logprobs,
            overlap=overlap,
            shared_outputs=self._capabilities.shared_outputs,
            choose_graph=choose_graph,
            lowering=lowering,
            direct_fallback=not self.is_external,
            tp_group=_tp_group_or_none(),
            last_stage=_is_last_stage(),
            native_kv=native_kv,
        )
        logger.info("Model runner %r runs steps through its API", self.runner_name)
        return self._step_forward

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
            self._buckets = inputs.buckets
            self._plugin.capture(inputs.buckets)
            self.phase = StartupPhase.CAPTURED
            final = self._checked(
                "finalize", self._plugin.finalize(inputs.capacity), KVLayout
            )
            if self._settle is not None:
                final = self._settle(final)
            _check_final_layout(allocated, final)
            self.layout = final
            self.phase = StartupPhase.FINALIZED
        except BaseException:
            self._close_after_failure()
            raise
        logger.info(
            "Model runner %r started: %d KV slots, %d full-attention layers, "
            "sub-pools %s, %s",
            self.runner_name,
            final.num_slots,
            len(final.full_attention),
            [(p.kind.value, p.num_items, len(p.buffers)) for p in final.sub_pools],
            final.limits,
        )
        return final

    def close(self) -> None:
        """Close the runner; a no-op before ``load`` or when already closed."""
        if self.phase in (StartupPhase.NEW, StartupPhase.CLOSED):
            self.phase = StartupPhase.CLOSED
            return
        self.phase = StartupPhase.CLOSED
        if self._step_forward is not None:
            self._step_forward.close()
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


def settle_capacity(host: NativeStartupHost, final: KVLayout) -> KVLayout:
    """Agree on capacity across pipeline stages after ``finalize``, then set the
    admission limits SGLang admits against from the agreed pools."""
    host.agree_stage_capacity()
    return msgspec.structs.replace(final, limits=admission_limits(host.tp_worker))


def parallel_layout() -> ParallelLayout:
    """The groups SGLang built for this rank (``init_parallel_runtime`` has run)."""
    parallel = get_parallel()
    if parallel.tp_size == 1 and parallel.pp_size == 1:
        # One rank: no group carries a collective.
        return ParallelLayout()
    return ParallelLayout(
        tp=rank_group(parallel.tp_group),
        pp=rank_group(parallel.pp_group),
        attn_tp=rank_group(parallel.attn_tp_group),
        attn_dp_rank=parallel.attn_dp_rank,
        attn_dp_size=parallel.attn_dp_size,
    )


def _is_last_stage() -> bool:
    parallel = get_parallel()
    return parallel.pp_size == 1 or parallel.pp_group.is_last_rank


def _tp_group_or_none():
    """The TP ``GroupCoordinator``; None at TP1, where none need be built."""
    parallel = get_parallel()
    return parallel.tp_group if parallel.tp_size > 1 else None


def rank_group(coordinator) -> RankGroup:
    """A ``GroupCoordinator`` as the API's ``RankGroup``."""
    if coordinator.world_size == 1:
        return RankGroup(ranks=tuple(coordinator.ranks))
    return RankGroup(
        rank=coordinator.rank_in_group,
        size=coordinator.world_size,
        ranks=tuple(coordinator.ranks),
        device=coordinator.device_group,
        cpu=coordinator.cpu_group,
    )


def _check_final_layout(allocated: KVLayout, final: KVLayout) -> None:
    if final.limits is None:
        raise ValueError("finalize() must return a KVLayout with limits set")
    if final.epoch != allocated.epoch or final.page_size != allocated.page_size:
        raise ValueError(
            "finalize() changed the KV epoch or page size: "
            f"{allocated.epoch}/{allocated.page_size} -> "
            f"{final.epoch}/{final.page_size}"
        )
    if final.placement() != allocated.placement():
        # Sizes may change in finalize; addresses and strides may not.
        raise ValueError("finalize() moved KV buffers within an epoch")


def _buckets(graphs, phase: str) -> tuple[int, ...]:
    if graphs is None:
        return ()
    config = graphs[phase]
    if config.backend == Backend.DISABLED or not config.bs:
        return ()
    return tuple(sorted(set(config.bs)))


def _external_graph(forward_batch, *, buckets: GraphBuckets) -> GraphChoice:
    from sglang.srt.model_runner_plugins.graphs import bucket_graph_choice
    from sglang.srt.model_runner_plugins.steps import step_mode

    return bucket_graph_choice(
        mode=step_mode(forward_batch),
        batch_size=forward_batch.batch_size,
        num_tokens=forward_batch.input_ids.shape[0],
        buckets=buckets,
    )
