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
    ):
        self.runner_name = runner_name
        self._executor = executor
        self._page_table = page_table
        self._native_prompt_logprobs = native_prompt_logprobs
        self._overlap = overlap
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
        self._native_kv = native_kv
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
            deferred=overlap and not speculative and not pooling,
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

    def forward(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> ModelRunnerOutput:
        from sglang.srt.model_executor.model_runner import ModelRunnerOutput

        self._lifetimes.step_started()
        if self._wait_for_kv_loads is not None:
            self._wait_for_kv_loads()
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
            with self._lowering.bind(step=step, forward_batch=forward_batch):
                plan = self._executor.plan(step)
        self._lifetimes.before_execute()
        tic = time.perf_counter()
        out = self._executor.execute(plan)
        self.last_execute_seconds = time.perf_counter() - tic
        self.last_graph = graph
        self.last_kv = step.kv
        self.last_adapters = step.adapters
        self.last_sampling_owner = owner
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
        if self._lifetimes.deferred:
            self._lease = lease
        if not self._last_stage:
            return ModelRunnerOutput(
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
        logits_output = logits_output_from_step(out, gather_vocab=self._gather_vocab)
        if step.outputs.prompt_logprobs and not self._native_prompt_logprobs:
            self._project_prompt_logprobs(
                plan=plan, forward_batch=forward_batch, logits_output=logits_output
            )
        return ModelRunnerOutput(
            logits_output=logits_output, can_run_graph=not graph.is_eager
        )

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
        if not self._overlap:
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
