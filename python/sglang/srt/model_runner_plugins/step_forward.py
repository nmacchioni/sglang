"""The worker boundary through the model runner API: one forward as
``plan`` -> ``execute`` -> outputs, with SGLang keeping sampling.

Block 3 runs with overlap scheduling off, on one stream: every consumer of a
step's outputs (sampling, the host copy) is enqueued before the next step is
planned, so a step's output is released when the next step starts, or at close.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.layers.logprob_processor import InputLogprobProcessor
from sglang.srt.model_runner_plugins.api import (
    ModelRunnerExecutor,
    ProjectRequest,
    StepOutput,
)
from sglang.srt.model_runner_plugins.steps import (
    build_step,
    logits_output_from_step,
    step_refusal,
)

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunnerOutput
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
        lowering: Optional[NativeStepLowering] = None,
        direct_fallback: bool = False,
    ):
        self.runner_name = runner_name
        self._executor = executor
        self._page_table = page_table
        self._native_prompt_logprobs = native_prompt_logprobs
        # Native only: the batch's own lowering, which its plan() reuses.
        self._lowering = lowering
        # Native only: batches outside the Step subset take the direct path.
        self._direct_fallback = direct_fallback
        self._input_logprobs = InputLogprobProcessor(vocab_size)
        self._unreleased: Optional[StepOutput] = None
        # Host seconds of the last step inside execute(), for the step trace.
        self.last_execute_seconds = 0.0

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

    def forward(self, forward_batch: ForwardBatch) -> ModelRunnerOutput:
        from sglang.srt.model_executor.model_runner import ModelRunnerOutput

        self._release_previous()
        step = build_step(forward_batch=forward_batch, page_table=self._page_table)
        if self._lowering is None:
            plan = self._executor.plan(step)
        else:
            with self._lowering.bind(step=step, forward_batch=forward_batch):
                plan = self._executor.plan(step)
        tic = time.perf_counter()
        out = self._executor.execute(plan)
        self.last_execute_seconds = time.perf_counter() - tic
        if not isinstance(out, StepOutput):
            raise TypeError(
                f"Model runner {self.runner_name!r} execute() returned "
                f"{type(out).__name__}; expected StepOutput."
            )
        self._unreleased = out
        logits_output = logits_output_from_step(out)
        if step.outputs.prompt_logprobs and not self._native_prompt_logprobs:
            self._project_prompt_logprobs(
                plan=plan, forward_batch=forward_batch, logits_output=logits_output
            )
        return ModelRunnerOutput(logits_output=logits_output, can_run_graph=False)

    def close(self) -> None:
        self._release_previous()

    def _release_previous(self) -> None:
        # Its consumers were enqueued on this stream before this step started.
        if self._unreleased is not None:
            out, self._unreleased = self._unreleased, None
            self._executor.release(out, None)

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
