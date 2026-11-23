"""SGLang's side of one step: ``Step`` as a view of the step's ``ForwardBatch``,
and ``StepOutput`` back to the ``LogitsProcessorOutput`` the scheduler reads.

``ForwardBatch.init_new`` is where SGLang lowers a ``ScheduleBatch`` for any
runner (positions, extend lengths, physical write slots, logprob metadata).
``Step`` references those tensors instead of copying them, so building it costs
a few attribute reads and no device work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.layers.logits_processor import (
    LogitsProcessorOutput,
    extend_reads_input_logprobs,
)
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_runner_plugins.api import (
    EAGER,
    GraphChoice,
    KVTables,
    LogitRows,
    OutputSpec,
    PromptLogprobs,
    Rows,
    Step,
    StepMode,
    StepOutput,
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
    if forward_batch.global_num_tokens_cpu is not None:
        return "MLP-sync padding"
    if forward_batch.input_ids_from_graph:
        return "input ids read inside the decode graph"
    return None


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
) -> Step:
    """The step ``forward_batch`` describes; every tensor is one of its own."""
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
        kv=KVTables(write_slots=forward_batch.out_cache_loc, page_table=page_table),
        outputs=OutputSpec(
            logit_rows=LogitRows.NONE if mode is StepMode.IDLE else LogitRows.LAST,
            prompt_logprobs=reads_prompt_logprobs(forward_batch),
        ),
        graph=graph,
        inputs_ready=inputs_ready,
    )


def logits_output_from_step(out: StepOutput) -> LogitsProcessorOutput:
    """The scheduler's view of a step's outputs."""
    logits_output = LogitsProcessorOutput(
        next_token_logits=None if out.logits is None else out.logits.tensor
    )
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
