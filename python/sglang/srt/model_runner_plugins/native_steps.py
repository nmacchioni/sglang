"""The built-in runner's steps: ``plan`` takes back the ``ForwardBatch`` the
``Step`` is a view of, and ``execute`` runs today's ``ModelRunner.forward`` on it.

Rebuilding a ``ForwardBatch`` from ``Step`` would re-derive the sampling,
logprob and attention metadata ``init_new`` already made, and copy the inputs a
second time; reusing the lowering keeps the step path at zero device copies.

Graphs stay the native graph runners': they capture as today and decide replay
inside ``ModelRunner.forward``. ``native_graph_choice`` asks them the same
question first, so ``Step.graph`` reports the decision, and ``native_execute``
checks the forward took it.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Iterator, Optional

import msgspec

from sglang.srt.hardware_backend.mlx.runtime import use_mlx
from sglang.srt.model_runner_plugins.api import (
    EAGER,
    GraphChoice,
    GraphPhase,
    Logits,
    PromptLogprobs,
    Step,
    StepOutput,
)
from sglang.srt.runtime_context import (
    get_exec,
    get_lora,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.utils.device import is_confidential_compute

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


class NativeStepLowering:
    """Hands ``plan`` the ``ForwardBatch`` of the step being planned."""

    def __init__(self):
        self._step: Optional[Step] = None
        self._forward_batch: Optional[ForwardBatch] = None

    @contextlib.contextmanager
    def bind(self, *, step: Step, forward_batch: ForwardBatch) -> Iterator[None]:
        self._step, self._forward_batch = step, forward_batch
        try:
            yield
        finally:
            self._step = self._forward_batch = None

    def forward_batch_of(self, step: Step) -> ForwardBatch:
        forward_batch = self._forward_batch
        if step is not self._step or forward_batch is None:
            raise RuntimeError("the built-in runner plans only the step SGLang binds")
        # The zero-copy invariant: the Step views the batch, it holds no copies.
        if (
            step.input_ids is not forward_batch.input_ids
            or step.kv.write_slots is not forward_batch.out_cache_loc
        ):
            raise RuntimeError("Step does not view the ForwardBatch it was built from")
        return forward_batch


class NativePlan(msgspec.Struct, frozen=True, kw_only=True):
    forward_batch: object
    graph: GraphChoice = EAGER


def native_step_refusals(model_runner: ModelRunner) -> list[str]:
    """Why the built-in runner keeps today's path for every step; empty when
    its TP1 steps run through the API. Fixed for the runner's lifetime."""
    reasons = []
    parallel = get_parallel()
    if (
        parallel.tp_size > 1
        or parallel.pp_size > 1
        or parallel.attn_dp_size > 1
        or parallel.attn_cp_size > 1
        or parallel.dcp_size > 1
    ):
        reasons.append("model parallelism")
    if not get_schedule().disable_overlap_schedule and is_confidential_compute():
        # Host copies then run on a worker thread, unordered with the streams
        # a release event could be recorded on.
        reasons.append("overlap scheduling with asynchronous host copies")
    if get_spec().speculative_algorithm is not None:
        reasons.append("speculative decoding")
    if get_lora().enable_lora:
        reasons.append("LoRA")
    features = get_exec().features
    if (
        features.aux_hidden_state_capture is not None
        or features.enable_return_routed_experts
        or features.enable_prefill_weight_versions
        or features.enable_encoder_swa_bounded_replay
        or get_exec().moe.expert_distribution_recorder_mode is not None
    ):
        reasons.append("per-forward captures")
    if get_exec().dllm.dllm_algorithm is not None:
        reasons.append("diffusion LLMs")
    if (
        not model_runner.is_generation
        or model_runner.is_multimodal
        or model_runner.is_draft_worker
        or use_mlx()
    ):
        reasons.append("a non-text or draft model")
    return reasons


def native_graph_choice(
    *, model_runner: ModelRunner, forward_batch: ForwardBatch
) -> GraphChoice:
    """The graph ``ModelRunner.forward`` will replay for ``forward_batch``: the
    same eligibility checks, in the same order, as ``_forward_raw``."""
    from sglang.srt.model_executor.runner.eager_runner import EagerRunner

    mode = forward_batch.forward_mode
    decode = model_runner.decode_cuda_graph_runner
    if mode.is_cuda_graph() and decode and decode.can_run_graph(forward_batch):
        return GraphChoice(
            phase=GraphPhase.DECODE,
            bucket=_bucket(decode, forward_batch.batch_size, decode.capture_bs),
        )
    prefill = model_runner.prefill_cuda_graph_runner
    if (
        mode.is_extend()
        and prefill is not None
        and not isinstance(prefill, EagerRunner)
        and prefill.can_run_graph(forward_batch)
    ):
        return GraphChoice(
            phase=GraphPhase.PREFILL,
            bucket=_bucket(
                prefill, forward_batch.input_ids.shape[0], prefill.capture_num_tokens
            ),
        )
    return EAGER


def _bucket(runner, rows: int, captured) -> int:
    if rows > captured[-1]:
        # Unpadded replay (disable_padding) of a shape the backend admits.
        return rows
    return runner._pad_to_bucket(rows, captured)


def native_execute(*, model_runner: ModelRunner, plan: NativePlan) -> StepOutput:
    out = model_runner.forward(plan.forward_batch)
    if out.can_run_graph == plan.graph.is_eager:
        raise RuntimeError(
            f"Step.graph said {_describe(plan.graph)} but the native forward "
            f"{'replayed a graph' if out.can_run_graph else 'ran eager'}"
        )
    if (
        out.expert_distribution_metrics is not None
        or out.routed_experts_output is not None
        or out.indexer_topk_output is not None
        or out.kv_weight_version_record is not None
    ):
        raise RuntimeError("a native forward on the Step path produced captures")
    return step_output_of(out.logits_output)


def _describe(graph: GraphChoice) -> str:
    if graph.is_eager:
        return "eager"
    return f"{graph.phase.value} graph of {graph.bucket}"


def step_output_of(logits_output: LogitsProcessorOutput) -> StepOutput:
    """The native output as a StepOutput; the Step subset sets no other field."""
    if (
        logits_output.hidden_states is not None
        or logits_output.last_hidden_states is not None
        or logits_output.next_token_logits_vocab_shard is not None
        or logits_output.customized_info
        or logits_output.beam_normalizer is not None
        or logits_output.mm_input_embeds is not None
    ):
        raise RuntimeError(
            "a native forward on the Step path produced outputs StepOutput lacks"
        )
    logprobs = None
    if logits_output.input_token_logprobs is not None:
        logprobs = PromptLogprobs(
            token_logprobs=logits_output.input_token_logprobs,
            top_logprobs_val=logits_output.input_top_logprobs_val,
            top_logprobs_idx=logits_output.input_top_logprobs_idx,
            top_logprobs_weights=logits_output.input_top_logprobs_weights,
            token_ids_logprobs_val=logits_output.input_token_ids_logprobs_val,
            token_ids_logprobs_idx=logits_output.input_token_ids_logprobs_idx,
            copy_done=logits_output.input_logprobs_copy_done,
        )
    logits = logits_output.next_token_logits
    return StepOutput(
        logits=None if logits is None else Logits(tensor=logits), logprobs=logprobs
    )
