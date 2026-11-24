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
from sglang.srt.model_runner_plugins.steps import dp_prefill_graph_bucket
from sglang.srt.runtime_context import (
    get_exec,
    get_lora,
    get_parallel,
    get_schedule,
    get_spec,
    reports_expert_balancedness,
)
from sglang.srt.utils.device import is_confidential_compute

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


# Speculative algorithms whose forwards run through the API. The target's verify
# steps are VERIFY Steps; n-gram drafts on the host, with no draft model; the
# EAGLE family's draft model runs DRAFT and DRAFT_EXTEND Steps (native_draft).
# DSpark is native-only: its target's steps are Steps, with its fused accept
# kept inside the native verify, and its draft stays inside its worker.
# DFLASH, UNO and frozen-KV MTP keep the direct path (see the 7d commit).
STEP_SPECULATIVE_ALGORITHMS = frozenset(
    # NEXTN names EAGLE over a model's own MTP layers; STANDALONE drafts with a
    # separate small model through the same draft worker.
    {"NGRAM", "EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "DSPARK"}
)


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
    # The PPProxyTensors over Step.pp_in, on every stage but the first.
    pp_proxy_tensors: Optional[object] = None


def native_plan(*, step: Step, forward_batch: ForwardBatch) -> NativePlan:
    pp_proxy_tensors = None
    if step.pp_in is not None:
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

        # The same dict the stage received: the forward reads it in place.
        pp_proxy_tensors = PPProxyTensors(step.pp_in)
    return NativePlan(
        forward_batch=forward_batch, graph=step.graph, pp_proxy_tensors=pp_proxy_tensors
    )


def native_step_refusals(
    model_runner: ModelRunner, *, kv_refusal: Optional[str] = None
) -> list[str]:
    """Why the built-in runner keeps today's path for every step; empty when
    its steps run through the API. Fixed for the runner's lifetime.

    Tensor parallelism runs through the API: every TP rank lowers the same
    batch from the same replicated schedule, so each member reaches the same
    graph decision and SGLang adds no collective to agree on it. Prefill context
    parallelism and expert parallelism are the same: whether an extend step is
    sharded across the CP group, and how tokens are dispatched to experts, are
    functions of that batch (and, under DP attention, of the MLP-sync gather).
    Under DP attention and pipeline parallelism the decisions come from the
    existing MLP-sync gather and the stage schedule, which ``Step.dp`` (with
    the two-batch-overlap split) and ``pp_in`` carry. EPLB rebalances inside
    the native forward, on the same forward count on every rank as today.

    Speculative decoding runs through the API for the algorithms in
    ``STEP_SPECULATIVE_ALGORITHMS``; the others keep the direct path.

    Elastic EP keeps the direct path: on a rank fault the forward rebalances
    and runs again, and scale-up rewrites the DP layout and recaptures graphs,
    decisions taken inside the forward rather than at SGLang's rendezvous.
    Decode context parallelism keeps it too, until a DCP configuration (MLA or
    DSA models) proves the step path bitwise against it. So do KV pools that
    ``KVLayout`` and ``KVTables`` do not describe yet (``kv_refusal``): MLA,
    index caches, packed and quantized layouts, per-request windows."""
    reasons = []
    if kv_refusal is not None:
        reasons.append(f"{kv_refusal} (no KVLayout description)")
    if get_parallel().dcp_size > 1:
        reasons.append("decode context parallelism")
    if get_exec().moe.elastic_ep_backend is not None:
        reasons.append("elastic EP")
    if not get_schedule().disable_overlap_schedule and is_confidential_compute():
        # Host copies then run on a worker thread, unordered with the streams
        # a release event could be recorded on.
        reasons.append("overlap scheduling with asynchronous host copies")
    algorithm = get_spec().speculative_algorithm
    if algorithm is not None and algorithm.upper() not in STEP_SPECULATIVE_ALGORITHMS:
        reasons.append(f"speculative decoding with {algorithm}")
    elif algorithm is not None and get_spec().enable_multi_layer_eagle:
        # Its own draft worker, whose draft calls are not Steps yet.
        reasons.append("multi-layer EAGLE")
    elif (
        algorithm is not None
        and algorithm.upper() == "DSPARK"
        and (
            "Inkling2ForConditionalGeneration"
            in (model_runner.model_config.hf_config.architectures or ())
        )
    ):
        # Served by the Inkling2 MTP worker, not the DSpark worker.
        reasons.append("DSPARK on Inkling2")
    if get_lora().enable_lora:
        reasons.append("LoRA")
    features = get_exec().features
    if (
        features.aux_hidden_state_capture is not None
        or features.enable_return_routed_experts
        or features.enable_prefill_weight_versions
        or features.enable_encoder_swa_bounded_replay
        # The recorder alone (EPLB statistics, dumps on request) returns
        # nothing per forward; the balancedness report does.
        or reports_expert_balancedness()
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
    from sglang.srt.model_executor.model_runner import (
        _prefill_cuda_graph_allows_context_parallel,
    )
    from sglang.srt.model_executor.runner.eager_runner import EagerRunner

    mode = forward_batch.forward_mode
    decode = model_runner.decode_cuda_graph_runner
    if mode.is_cuda_graph() and decode and decode.can_run_graph(forward_batch):
        # Under DP attention every rank replays the bucket of the largest
        # rank's batch, as gathered at the rendezvous.
        rows = (
            decode._max_dp_batch_size(forward_batch)
            if decode.require_mlp_tp_gather
            else forward_batch.batch_size
        )
        return GraphChoice(
            phase=GraphPhase.DECODE, bucket=_bucket(decode, rows, decode.capture_bs)
        )
    prefill = model_runner.prefill_cuda_graph_runner
    if prefill is None or isinstance(prefill, EagerRunner):
        return EAGER
    dp_bucket = dp_prefill_graph_bucket(forward_batch)
    if dp_bucket is not None:
        # Under DP attention the forward pads every rank, idle and decode ranks
        # included, to the shared bucket before it asks the graph runner; the
        # rendezvous's vote is that every rank can replay it.
        return GraphChoice(phase=GraphPhase.PREFILL, bucket=dp_bucket)
    if (
        mode.is_extend()
        and prefill.can_run_graph(forward_batch)
        and _prefill_cuda_graph_allows_context_parallel(prefill, forward_batch)
    ):
        global_num_tokens = forward_batch.global_num_tokens_cpu
        rows = (
            forward_batch.input_ids.shape[0]
            if global_num_tokens is None
            else max(global_num_tokens)
        )
        return GraphChoice(
            phase=GraphPhase.PREFILL,
            bucket=_bucket(prefill, rows, prefill.capture_num_tokens),
        )
    return EAGER


def _bucket(runner, rows: int, captured) -> int:
    if rows > captured[-1]:
        # Unpadded replay (disable_padding) of a shape the backend admits.
        return rows
    return runner._pad_to_bucket(rows, captured)


def native_execute(*, model_runner: ModelRunner, plan: NativePlan) -> StepOutput:
    forward_batch = plan.forward_batch
    out = model_runner.forward(forward_batch, pp_proxy_tensors=plan.pp_proxy_tensors)
    if out.can_run_graph == plan.graph.is_eager:
        raise RuntimeError(
            f"Step.graph said {_describe(plan.graph)} but the native forward "
            f"{'replayed a graph' if out.can_run_graph else 'ran eager'}"
        )
    if (
        plan.graph.phase is GraphPhase.PREFILL
        and forward_batch.global_num_tokens_cpu is not None
    ):
        # The forward padded every rank to one bucket (MAX_LEN), as the Step said.
        padding, padded = (
            forward_batch.dp_padding_mode,
            forward_batch.global_num_tokens_cpu,
        )
        if (
            padding is None
            or not padding.is_max_len()
            or set(padded) != {plan.graph.bucket}
        ):
            raise RuntimeError(
                f"Step.graph said {_describe(plan.graph)} on every DP rank but "
                f"the native forward padded the ranks to {padded} ({padding})"
            )
    if (
        out.expert_distribution_metrics is not None
        or out.routed_experts_output is not None
        or out.indexer_topk_output is not None
        or out.kv_weight_version_record is not None
    ):
        raise RuntimeError("a native forward on the Step path produced captures")
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

    if isinstance(out.logits_output, PPProxyTensors):
        # A stage before the last: what goes to the next stage.
        return StepOutput(pp_out=out.logits_output.tensors)
    return step_output_of(out.logits_output)


def _describe(graph: GraphChoice) -> str:
    if graph.is_eager:
        return "eager"
    return f"{graph.phase.value} graph of {graph.bucket}"


def step_output_of(logits_output: LogitsProcessorOutput) -> StepOutput:
    """The native output as a StepOutput; the Step subset sets no other field."""
    if (
        logits_output.hidden_states_token_indices is not None
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
        logits=None if logits is None else Logits(tensor=logits),
        logprobs=logprobs,
        hidden=logits_output.hidden_states,
    )
