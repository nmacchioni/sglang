"""The worker boundary through the model runner API: one forward as
``plan`` -> ``execute`` -> outputs, with SGLang keeping sampling.

SGLang decides each step's graph and, under overlap scheduling, records
``Step.inputs_ready`` on the runner stream. When outputs are released is
``OutputLifetimes``'s: at the next step without overlap, or when the scheduler
has enqueued the last consumer with it (``take_lease``).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.layers.logprob_processor import InputLogprobProcessor
from sglang.srt.model_runner_plugins.api import (
    EAGER,
    GraphChoice,
    HiddenRows,
    KVTables,
    ModelRunnerExecutor,
    OutputBuffer,
    ProjectRequest,
    SamplingOwner,
    Step,
    StepMode,
    StepOutput,
)
from sglang.srt.model_runner_plugins.lifetimes import OutputLease, OutputLifetimes
from sglang.srt.model_runner_plugins.steps import (
    build_step,
    logits_output_from_step,
    runner_may_sample,
    step_refusal,
)

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        PPProxyTensors,
    )
    from sglang.srt.model_executor.model_runner import ModelRunnerOutput
    from sglang.srt.model_runner_plugins.native_kv import NativeKV
    from sglang.srt.model_runner_plugins.native_steps import NativeStepLowering

logger = logging.getLogger(__name__)


class StepForward:
    """Runs the forwards a ``TpModelWorker`` hands it through ``executor``."""

    def __init__(
        self,
        *,
        runner_name: str,
        executor: ModelRunnerExecutor,
        page_table: torch.Tensor,
        vocab_size: int,
        native_prompt_logprobs: bool,
        overlap: bool = False,
        shared_outputs: tuple[OutputBuffer, ...] = (),
        choose_graph: Optional[Callable[[ForwardBatch], GraphChoice]] = None,
        lowering: Optional[NativeStepLowering] = None,
        direct_fallback: bool = False,
        tp_group: Optional[Any] = None,
        last_stage: bool = True,
        native_kv: Optional[NativeKV] = None,
        wait_for_kv_loads: Optional[Callable[[], None]] = None,
        speculative: bool = False,
        pooling: bool = False,
        adapter_slots: Optional[
            Callable[[ForwardBatch], Optional[torch.Tensor]]
        ] = None,
        fused_sampling: bool = False,
        stream_ordered: bool = False,
        traced: bool = False,
        builtin: bool = False,
    ):
        self.runner_name = runner_name
        self._executor = executor
        self._page_table = page_table
        self._native_prompt_logprobs = native_prompt_logprobs
        # Under overlap scheduling, record Step.inputs_ready: not for a runner
        # that executes on the stream SGLang wrote the inputs on (native).
        self._record_ready = overlap and not stream_ordered
        # Host time inside execute() and the step's fields, for the step trace.
        self._traced = traced
        # Native: execute() runs on the runner stream and returns the native
        # forward's own outputs, which the lowering keeps.
        self._stream_ordered = stream_ordered
        # The built-in runner, untraced: steps without a rendezvous run
        # unmaterialized, through its plan_batch/run_plan (the native forward).
        self._unmaterialized = builtin and not traced
        self._choose_graph = choose_graph
        # Native only: the batch's own lowering, which its plan() reuses.
        self._lowering = lowering
        # Native only: batches outside the Step subset take the direct path.
        self._direct_fallback = direct_fallback
        self._vocab_size = vocab_size
        # SGLang's TP GroupCoordinator, when the runner spans several ranks.
        self._tp_group = tp_group
        # False on pipeline stages before the last, which return pp_out.
        self._last_stage = last_stage
        # Native only: the window and state tables beside the full-attention ones.
        self._native_kv = (
            native_kv
            if native_kv is not None
            and (native_kv.window_pool is not None or native_kv.state_pool is not None)
            else None
        )
        # External runners with HiCache: the runner stream waits for host loads
        # into the runner's KV before the step (and its inputs_ready) is built.
        self._wait_for_kv_loads = wait_for_kv_loads
        # A pooling model: steps return pooled outputs, no logits.
        self._pooling = pooling
        # Native only, with LoRA: each request's adapter slot in this step.
        self._adapter_slots = adapter_slots
        # The runner samples greedy batches itself (RunnerCapabilities).
        self._fused_sampling = fused_sampling
        # The last runner-sampled step's batch and StepOutput.next_ids, until
        # SGLang's sampling of that batch takes them.
        self._next_ids: Optional[tuple[ForwardBatch, torch.Tensor]] = None
        self.last_sampling_owner = SamplingOwner.SGLANG
        # A denoising pass's batch, step and plan, executed again by the next
        # pass over the same batch; how many times the last plan ran.
        self._denoise_plan: Optional[tuple[ForwardBatch, Step, Any]] = None
        self.last_plan_runs = 0
        self._input_logprobs = InputLogprobProcessor(vocab_size)
        self._lifetimes = OutputLifetimes(
            runner_name=runner_name,
            release=executor.release,
            shared=frozenset(shared_outputs),
            # Under speculative decoding the speculative worker consumes every
            # target output (verify logits, hidden states) on the runner stream
            # before it returns, so each output is released when the next step
            # starts, with or without overlap scheduling.
            # A pooling model's outputs are copied to the host on the runner
            # stream inside the forward call, and no lease reaches the scheduler.
            # A stream-ordered runner (native) orders every overwrite of its
            # outputs behind their consumers on the runner stream itself, and
            # its release() reads no event: outputs go back at the next step.
            deferred=overlap and not speculative and not pooling and not stream_ordered,
        )
        self._lease: Optional[OutputLease] = None
        # Host seconds of the last step inside execute(), and its graph, for
        # the step trace.
        self.last_execute_seconds = 0.0
        self.last_graph: GraphChoice = EAGER
        # The last step's KV tables and adapter slots, for the step trace.
        self.last_kv: Optional[KVTables] = None
        self.last_adapters: Optional[torch.Tensor] = None

    def takes(self, forward_batch: ForwardBatch) -> bool:
        """Whether ``forward_batch`` runs through the API; False leaves it to
        the direct path, which only the built-in runner has."""
        refusal = step_refusal(forward_batch)
        if refusal is None:
            return True
        if self._direct_fallback:
            return False
        raise ValueError(
            f"--model-runner {self.runner_name} cannot run a step with {refusal}"
        )

    def try_forward(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
    ) -> Optional[ModelRunnerOutput]:
        """``forward`` when the batch runs through the API, in one call; None
        when it takes the direct path (see ``takes``)."""
        if self._unmaterialized and forward_batch.global_num_tokens_cpu is None:
            # forward()'s unmaterialized lane, inlined. It needs no refusal
            # check: with no Step built, a batch outside the Step subset runs
            # the same native forward here as on the direct path.
            executor = self._executor
            return executor.run_plan(
                executor.plan_batch(
                    forward_batch=forward_batch, pp_proxy_tensors=pp_proxy_tensors
                )
            )
        refusal = step_refusal(forward_batch)
        if refusal is not None:
            if self._direct_fallback:
                return None
            raise ValueError(
                f"--model-runner {self.runner_name} cannot run a step with {refusal}"
            )
        return self.forward(forward_batch, pp_proxy_tensors)

    def forward(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> ModelRunnerOutput:
        if self._unmaterialized and forward_batch.global_num_tokens_cpu is None:
            # The built-in runner, with nothing to read its Step (no step
            # trace) and no rendezvous to check it against (no DP attention):
            # its plan() would read only the batch back, so neither the Step
            # nor the StepOutput is materialized, the native forward decides
            # the graph as on the direct path, and its own output comes back.
            # Releasing that output would be a no-op (stream-ordered).
            executor = self._executor
            return executor.run_plan(
                executor.plan_batch(
                    forward_batch=forward_batch, pp_proxy_tensors=pp_proxy_tensors
                )
            )
        self._lifetimes.step_started()
        if self._wait_for_kv_loads is not None:
            self._wait_for_kv_loads()
        step, plan = self._planned(forward_batch, pp_proxy_tensors)
        graph, owner = step.graph, step.sampling_owner
        self._lifetimes.before_execute()
        if self._traced:
            tic = time.perf_counter()
            out = self._executor.execute(plan)
            self.last_execute_seconds = time.perf_counter() - tic
            self.last_graph = graph
            self.last_kv = step.kv
            self.last_adapters = step.adapters
            self.last_sampling_owner = owner
        else:
            out = self._executor.execute(plan)
        if not isinstance(out, StepOutput):
            raise TypeError(
                f"Model runner {self.runner_name!r} execute() returned "
                f"{type(out).__name__}; expected StepOutput."
            )
        if out.ready is not None:
            # Every consumer runs on, or behind, the runner stream.
            device = torch.get_device_module(forward_batch.input_ids.device)
            device.current_stream().wait_event(out.ready)
        lease = self._lifetimes.produced(out)
        if lease is not None:
            self._lease = lease
        return self._outputs(step=step, plan=plan, out=out, forward_batch=forward_batch)

    def _outputs(
        self, *, step: Step, plan: Any, out: StepOutput, forward_batch: ForwardBatch
    ) -> ModelRunnerOutput:
        """The scheduler's view of ``out``, after checking it against ``step``."""
        graph, owner = step.graph, step.sampling_owner
        if not self._last_stage:
            return _model_runner_output_type()(
                logits_output=self._pp_out(out), can_run_graph=not graph.is_eager
            )
        if owner is SamplingOwner.PLUGIN:
            if out.next_ids is None or out.next_ids.shape[0] != step.rows.batch_size:
                raise ValueError(
                    f"Model runner {self.runner_name!r} owns this step's sampling "
                    f"but returned no next_ids for its {step.rows.batch_size} requests"
                )
            self._next_ids = (forward_batch, out.next_ids)
        if step.outputs.pooled and out.pooled is None:
            raise ValueError(
                f"Model runner {self.runner_name!r} returned no pooled outputs for "
                "a pooling model's step"
            )
        if step.outputs.hidden is not HiddenRows.NONE and out.hidden is None:
            raise ValueError(
                f"Model runner {self.runner_name!r} returned no hidden states for "
                f"a step that asked for them ({step.outputs.hidden.value})"
            )
        if self._lowering is not None:
            # The built-in runner's own forward output, when it is this
            # StepOutput's source: zero copies, nothing rebuilt.
            native = self._lowering.native_output_of(out)
            if native is not None:
                return native
        logits_output = logits_output_from_step(out, gather_vocab=self._gather_vocab)
        if step.mode is StepMode.DENOISE:
            # The diffusion algorithm reads every row's logits.
            logits_output.full_logits = logits_output.next_token_logits
            logits_output.next_token_logits = None
        if step.outputs.prompt_logprobs and not self._native_prompt_logprobs:
            self._project_prompt_logprobs(
                plan=plan, forward_batch=forward_batch, logits_output=logits_output
            )
        return _model_runner_output_type()(
            logits_output=logits_output, can_run_graph=not graph.is_eager
        )

    def _planned(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
    ) -> tuple[Step, Any]:
        """The step ``forward_batch`` describes and its plan. A denoising pass
        over a batch already planned executes that plan again: SGLang rewrote
        its input_ids in place."""
        reused = self._denoise_plan
        if reused is not None and reused[0] is forward_batch:
            self.last_plan_runs += 1
            return reused[1], reused[2]
        graph = (
            EAGER if self._choose_graph is None else self._choose_graph(forward_batch)
        )
        owner = (
            SamplingOwner.PLUGIN
            if self._fused_sampling
            and self._last_stage
            and runner_may_sample(forward_batch)
            else SamplingOwner.SGLANG
        )
        step = build_step(
            forward_batch=forward_batch,
            page_table=self._page_table,
            graph=graph,
            inputs_ready=self._record_inputs_ready(forward_batch),
            pp_in=None if pp_proxy_tensors is None else pp_proxy_tensors.tensors,
            last_stage=self._last_stage,
            pooling=self._pooling,
            adapters=(
                None
                if self._adapter_slots is None
                else self._adapter_slots(forward_batch)
            ),
            sampling_owner=owner,
            **self._sub_pool_tables(forward_batch),
        )
        if self._lowering is None:
            plan = self._executor.plan(step)
        else:
            self._lowering.bind(step=step, forward_batch=forward_batch)
            try:
                plan = self._executor.plan(step)
            finally:
                self._lowering.unbind()
        self._denoise_plan = (
            (forward_batch, step, plan) if step.mode is StepMode.DENOISE else None
        )
        self.last_plan_runs = 1
        return step, plan

    def _sub_pool_tables(self, forward_batch: ForwardBatch) -> dict:
        kv = self._native_kv
        if kv is None:
            return {}
        return {
            "window": kv.window_tables(forward_batch),
            "state": kv.state_tables(forward_batch),
        }

    def take_next_ids(self, forward_batch: ForwardBatch) -> Optional[torch.Tensor]:
        """The tokens the runner sampled for ``forward_batch``, once; None when
        SGLang owns that step's sampling."""
        if self._next_ids is None or self._next_ids[0] is not forward_batch:
            return None
        next_ids, self._next_ids = self._next_ids[1], None
        return next_ids

    def take_lease(self) -> Optional[OutputLease]:
        """The last step's output for the scheduler to mark consumed (overlap
        scheduling only); None when the last forward went direct."""
        lease, self._lease = self._lease, None
        return lease

    def consumers_enqueued(self, lease: OutputLease, after: Optional[Any]) -> None:
        self._lifetimes.consumers_enqueued(lease, after)

    def close(self) -> None:
        if self._lifetimes.pending and self._lifetimes.deferred:
            # Consumers may still be reading on other streams.
            torch.get_device_module().synchronize()
        self._lease = None
        self._lifetimes.close()

    def _pp_out(self, out: StepOutput) -> PPProxyTensors:
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

        if out.pp_out is None or out.logits is not None:
            raise ValueError(
                f"Model runner {self.runner_name!r} returned no pp_out (or logits) "
                "for a pipeline stage before the last"
            )
        return PPProxyTensors(out.pp_out)

    def _gather_vocab(self, shard: torch.Tensor) -> torch.Tensor:
        """Full logits from this rank's vocabulary shard: a gather every TP
        rank runs at the same point of the same step."""
        if self._tp_group is None or self._tp_group.world_size == 1:
            full = shard
        else:
            full = self._tp_group.all_gather(shard.contiguous(), dim=-1)
        if full.shape[-1] < self._vocab_size:
            raise ValueError(
                f"Model runner {self.runner_name!r} returned vocabulary shards of "
                f"{shard.shape[-1]} columns, {full.shape[-1]} in all; the "
                f"vocabulary has {self._vocab_size}"
            )
        if full.shape[-1] == self._vocab_size:
            return full
        return full[..., : self._vocab_size].contiguous()

    def _record_inputs_ready(self, forward_batch: ForwardBatch) -> Optional[Any]:
        if not self._record_ready:
            return None
        # The scheduler wrote this step's inputs on the current (runner) stream.
        event = torch.get_device_module(forward_batch.input_ids.device).Event()
        event.record()
        return event

    def _project_prompt_logprobs(
        self,
        *,
        plan: Any,
        forward_batch: ForwardBatch,
        logits_output: LogitsProcessorOutput,
    ) -> None:
        """Prompt logprobs through project(), selecting and chunking rows as the
        native LogitsProcessor does."""
        metadata = LogitsMetadata.from_forward_batch(forward_batch)
        num_rows = forward_batch.input_ids.shape[0]
        input_rows = torch.arange(
            num_rows, dtype=torch.int64, device=forward_batch.input_ids.device
        )
        # Row selection reads no LogitsProcessor state, only the metadata.
        (pruned_rows, _, _, sample_indices, logprob_indices, token_to_seq) = (
            LogitsProcessor._get_pruned_states(None, input_rows, None, None, metadata)
        )

        def project(rows, lm_head, logits_metadata, use_logits_buffer=True):
            return self._executor.project(ProjectRequest(plan=plan, rows=rows))

        result, _ = self._input_logprobs.forward(
            pruned_states=pruned_rows,
            sample_indices=sample_indices,
            input_logprob_indices=logprob_indices,
            token_to_seq_idx=token_to_seq,
            lm_head=None,
            get_logits_fn=project,
            logits_metadata=metadata,
        )
        result.write_input_to(logits_output)


class DenoiseRunner:
    """The model runner a diffusion algorithm drives, with its forwards at the
    worker boundary: each denoising pass is an execution of the batch's plan.
    Everything else the algorithm reads comes from the built-in runner."""

    def __init__(self, *, model_runner: Any, forward_model: Callable):
        self._model_runner = model_runner
        self._forward_model = forward_model

    def forward(self, forward_batch: ForwardBatch, pp_proxy_tensors=None):
        return self._forward_model(forward_batch, pp_proxy_tensors)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model_runner, name)


def _model_runner_output_type():
    global _MODEL_RUNNER_OUTPUT
    if _MODEL_RUNNER_OUTPUT is None:
        from sglang.srt.model_executor.model_runner import ModelRunnerOutput

        _MODEL_RUNNER_OUTPUT = ModelRunnerOutput
    return _MODEL_RUNNER_OUTPUT


# Imported on first use: the model runner module imports this package.
_MODEL_RUNNER_OUTPUT = None
