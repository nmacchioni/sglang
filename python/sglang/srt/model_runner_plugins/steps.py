"""SGLang's side of one step: ``Step`` as a view of the step's ``ForwardBatch``,
and ``StepOutput`` back to the ``LogitsProcessorOutput`` the scheduler reads.

``ForwardBatch.init_new`` is where SGLang lowers a ``ScheduleBatch`` for any
runner (positions, extend lengths, physical write slots, logprob metadata).
``Step`` references those tensors instead of copying them, so building it costs
a few attribute reads and no device work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from sglang.srt.layers.logits_processor import (
    LogitsProcessorOutput,
    extend_reads_input_logprobs,
)
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_runner_plugins.api import (
    EAGER,
    DPInfo,
    GraphChoice,
    KVTables,
    LogitRows,
    LogitsLayout,
    OutputSpec,
    PromptLogprobs,
    Rows,
    StateTables,
    Step,
    StepMode,
    StepOutput,
    WindowTables,
)

if TYPE_CHECKING:
    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_STEP_MODES = {
    ForwardMode.EXTEND: StepMode.EXTEND,
    ForwardMode.DECODE: StepMode.DECODE,
    ForwardMode.MIXED: StepMode.MIXED,
    ForwardMode.IDLE: StepMode.IDLE,
}


def step_refusal(forward_batch: ForwardBatch) -> Optional[str]:
    """Why ``forward_batch`` cannot run as a ``Step`` yet; None when it can."""
    if forward_batch.forward_mode not in _STEP_MODES:
        return f"forward mode {forward_batch.forward_mode.name}"
    if forward_batch.capture_hidden_mode != CaptureHiddenMode.NULL:
        return "hidden-state capture"
    if (
        forward_batch.input_embeds is not None
        or forward_batch.replace_embeds is not None
    ):
        return "input embeddings"
    if forward_batch.mm_inputs is not None and any(forward_batch.mm_inputs):
        return "multimodal inputs"
    if any(lora_id is not None for lora_id in forward_batch.lora_ids or ()):
        return "LoRA adapters"
    if forward_batch.beam is not None:
        return "beam search"
    if (
        forward_batch.token_indices_to_pool is not None
        or forward_batch.multi_item_delimiter_indices is not None
    ):
        return "position scoring"
    if forward_batch.input_candidate_selectors is not None:
        return "input candidate selectors"
    if forward_batch.input_ids_from_graph:
        return "input ids read inside the decode graph"
    if (
        forward_batch.global_num_tokens_cpu is not None
        and forward_batch.is_extend_in_batch
        and forward_batch.can_run_dp_prefill_cuda_graph
        and dp_prefill_graph_bucket(forward_batch) is None
    ):
        # The vote stands but the ranks' tokens fit no shared bucket: each rank
        # then decides its own prefill graph after padding to its own count.
        # The inputs are the rendezvous's, so every rank falls back alike.
        return "a prefill graph step under DP attention without a shared bucket"
    return None


def dp_prefill_graph_bucket(forward_batch: ForwardBatch) -> Optional[int]:
    """The prefill graph bucket every DP rank pads this step to, from the
    rendezvous's token counts; None when the step has no shared bucket.

    The same decision ``ForwardBatch.prepare_mlp_sync_batch`` takes in the
    forward (MAX_LEN padding to one captured bucket); the built-in runner
    checks after its forward that it was taken. None as well under context
    parallelism, where replay also depends on whether the padded batch is
    CP-sharded."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.cp.base import get_cp_strategy
    from sglang.srt.model_executor.forward_batch_info import (
        _mega_moe_materializes_idle_rank,
        prefill_graph_tolerates_sum_len,
    )
    from sglang.srt.runtime_context import get_exec, get_parallel
    from sglang.srt.utils.common import ceil_align

    global_num_tokens = forward_batch.global_num_tokens_cpu
    if (
        global_num_tokens is None
        or not forward_batch.is_extend_in_batch
        or not forward_batch.can_run_dp_prefill_cuda_graph
        or forward_batch.dp_spec_prefill_coordination_applied
        or get_cp_strategy() is not None
    ):
        return None
    if _mega_moe_materializes_idle_rank(forward_batch):
        global_num_tokens = [1] * len(global_num_tokens)
    attn_tp_size = get_parallel().attn_tp_size
    rows = [ceil_align(n, attn_tp_size) for n in global_num_tokens]
    buckets = get_exec().graph.cuda_graph_config.prefill.bs
    idle_allowed = (
        min(rows) > 0
        if envs.SGLANG_ENABLE_PREFILL_GRAPH_IDLE_RANK_EAGER.get()
        else True
    )
    if (
        not buckets
        or max(rows) > max(buckets)
        or not idle_allowed
        or prefill_graph_tolerates_sum_len()
    ):
        return None
    return min(b for b in buckets if b >= max(rows))


def step_mode(forward_batch: ForwardBatch) -> StepMode:
    return _STEP_MODES[forward_batch.forward_mode]


def reads_prompt_logprobs(forward_batch: ForwardBatch) -> bool:
    return (
        forward_batch.forward_mode.is_extend()
        and forward_batch.return_logprob
        and extend_reads_input_logprobs(forward_batch)
    )


def build_step(
    *,
    forward_batch: ForwardBatch,
    page_table: torch.Tensor,
    graph: GraphChoice = EAGER,
    inputs_ready: Optional[Any] = None,
    pp_in: Optional[dict[str, torch.Tensor]] = None,
    last_stage: bool = True,
    window: Optional[WindowTables] = None,
    state: Optional[StateTables] = None,
) -> Step:
    """The step ``forward_batch`` describes; every tensor is one of its own.

    Raises ``RendezvousMismatch`` when the batch is not the one this rank
    published at the MLP-sync rendezvous, before any collective runs."""
    mode = _STEP_MODES[forward_batch.forward_mode]
    extend = mode in (StepMode.EXTEND, StepMode.MIXED)
    return Step(
        mode=mode,
        input_ids=forward_batch.input_ids,
        positions=forward_batch.positions,
        rows=Rows(
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            query_start=forward_batch.extend_start_loc if extend else None,
            query_lens=forward_batch.extend_seq_lens if extend else None,
            prefix_lens=forward_batch.extend_prefix_lens if extend else None,
            query_lens_cpu=forward_batch.extend_seq_lens_cpu if extend else None,
            prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu if extend else None,
        ),
        kv=KVTables(
            write_slots=forward_batch.out_cache_loc,
            page_table=page_table,
            window=window,
            state=state,
        ),
        outputs=OutputSpec(
            logit_rows=(
                LogitRows.LAST
                if last_stage and mode is not StepMode.IDLE
                else LogitRows.NONE
            ),
            prompt_logprobs=last_stage and reads_prompt_logprobs(forward_batch),
            pp_out=not last_stage,
        ),
        graph=graph,
        inputs_ready=inputs_ready,
        dp=dp_info(forward_batch),
        pp_in=pp_in,
    )


def dp_info(forward_batch: ForwardBatch) -> Optional[DPInfo]:
    """The MLP-sync rendezvous's decisions for this step; None without one."""
    from sglang.srt.layers.dp_attention import dp_slot_in

    global_num_tokens = forward_batch.global_num_tokens_cpu
    if global_num_tokens is None:
        return None
    return DPInfo(
        rank=dp_slot_in(global_num_tokens),
        global_num_tokens=tuple(global_num_tokens),
        global_num_tokens_for_logprob=tuple(
            forward_batch.global_num_tokens_for_logprob_cpu
        ),
        extend_in_batch=forward_batch.is_extend_in_batch,
        decode_graph=forward_batch.can_run_decode_cuda_graph,
        prefill_graph=forward_batch.can_run_dp_prefill_cuda_graph,
        tbo_split_seq_index=forward_batch.tbo_split_seq_index,
    )


def logits_output_from_step(
    out: StepOutput, *, gather_vocab: Optional[Callable[[torch.Tensor], torch.Tensor]]
) -> LogitsProcessorOutput:
    """The scheduler's view of a step's outputs; ``gather_vocab`` turns
    vocabulary-sharded logits into full ones (None: the runner has one rank)."""
    logits = None if out.logits is None else out.logits.tensor
    if logits is not None and out.logits.layout is LogitsLayout.VOCAB_SHARDED:
        if gather_vocab is None:
            raise ValueError("vocabulary-sharded logits need a TP group to gather")
        logits = gather_vocab(logits)
    logits_output = LogitsProcessorOutput(next_token_logits=logits)
    if out.pp_out is not None:
        raise ValueError("a last-stage step returned pp_out")
    if out.logprobs is not None:
        write_prompt_logprobs(out.logprobs, logits_output)
    return logits_output


def write_prompt_logprobs(
    logprobs: PromptLogprobs, logits_output: LogitsProcessorOutput
) -> None:
    logits_output.input_token_logprobs = logprobs.token_logprobs
    if logprobs.top_logprobs_val is not None:
        logits_output.input_top_logprobs_val = logprobs.top_logprobs_val
        logits_output.input_top_logprobs_idx = logprobs.top_logprobs_idx
        logits_output.input_top_logprobs_weights = logprobs.top_logprobs_weights
    if logprobs.token_ids_logprobs_val is not None:
        logits_output.input_token_ids_logprobs_val = logprobs.token_ids_logprobs_val
        logits_output.input_token_ids_logprobs_idx = logprobs.token_ids_logprobs_idx
    logits_output.input_logprobs_copy_done = logprobs.copy_done
