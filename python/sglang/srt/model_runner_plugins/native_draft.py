"""The built-in runner's draft model on the Step path (EAGLE family).

The target and the draft are two runner instances: the target's steps run
through ``StepForward``, the draft's through ``NativeDraftSteps``. The draft is
native: SGLang's EAGLE worker loads it beside its target (embeddings and LM
head shared, read-only target KV through the shared page table: the target
handle), so it serves only the built-in target. Each draft call becomes a Step:

- ``DRAFT``: the k-step top-k expansion, one call (one graph when replayed),
  returning a ``DraftTree``. SGLang builds the verify tree from it.
- ``DRAFT_EXTEND``: the draft's forward over the target's accepted tokens (or a
  prompt), fed the target's hidden states (``SpecInputs.hidden``).

``plan`` takes back the draft batch the Step views, as the target's native
lowering does, and ``execute`` runs the worker's own draft code on it, so the
draft graphs, the fused top-1 chain and the draft sampling stay native.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

import msgspec

from sglang.srt.model_runner_plugins.api import (
    EAGER,
    DraftTree,
    GraphChoice,
    GraphPhase,
    HiddenRows,
    KVTables,
    LogitRows,
    Logits,
    OutputSpec,
    Rows,
    SpecInputs,
    Step,
    StepMode,
    StepOutput,
)

if TYPE_CHECKING:
    import torch

    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_runner_plugins.step_trace import StepTrace


class NativeDraftPlan(msgspec.Struct, frozen=True, kw_only=True):
    # The worker's own draft code for the bound batch.
    run: Any
    graph: GraphChoice = EAGER


class SGLangDraftPlugin:
    """The executor half of the built-in draft runner: its lifecycle is the
    target's (native startup loads, allocates and captures both together)."""

    def __init__(self):
        self._bound: Optional[tuple[Step, Callable[[], Any]]] = None

    @contextlib.contextmanager
    def bind(self, *, step: Step, run: Callable[[], Any]) -> Iterator[None]:
        self._bound = (step, run)
        try:
            yield
        finally:
            self._bound = None

    def plan(self, step: Step) -> NativeDraftPlan:
        if self._bound is None or self._bound[0] is not step:
            raise RuntimeError("the built-in draft plans only the step SGLang binds")
        return NativeDraftPlan(run=self._bound[1], graph=step.graph)

    def execute(self, plan: NativeDraftPlan) -> StepOutput:
        out = plan.run()
        if isinstance(out, tuple):
            parents, scores, tokens, probs = out
            return StepOutput(
                draft=DraftTree(
                    tokens=tokens, parents=parents, scores=scores, probs=probs
                )
            )
        if (
            out.hidden_states_token_indices is not None
            or out.last_hidden_states is not None
        ):
            raise RuntimeError("a native draft step produced outputs StepOutput lacks")
        return StepOutput(
            logits=Logits(tensor=out.next_token_logits), hidden=out.hidden_states
        )

    def release(self, out: StepOutput, after: Optional[Any]) -> None:
        # Draft outputs are consumed on the runner stream within the same
        # speculative round, before the next draft step is planned.
        pass


class NativeDraftSteps:
    """Runs the EAGLE draft worker's draft calls as DRAFT and DRAFT_EXTEND Steps."""

    def __init__(
        self,
        *,
        runner_name: str,
        page_table: torch.Tensor,
        topk: int,
        num_steps: int,
        num_draft_tokens: int,
        trace: Optional[StepTrace] = None,
    ):
        self.runner_name = runner_name
        self._executor = SGLangDraftPlugin()
        self._page_table = page_table
        self._topk = topk
        self._num_steps = num_steps
        self._num_draft_tokens = num_draft_tokens
        self._trace = trace
        self.steps_executed = 0

    def draft(
        self,
        forward_batch: ForwardBatch,
        *,
        graph_runner: Optional[Any],
        run: Callable[[], tuple],
    ) -> tuple:
        """The expansion: ``run`` is the worker's native draft of the batch
        (one graph replay, or the eager loop); returns its
        ``(parents, scores, tokens, probs)``."""
        spec_info = forward_batch.spec_info
        step = Step(
            mode=StepMode.DRAFT,
            # The seeds of the expansion: each request's top-k draft tokens.
            input_ids=spec_info.topk_index.reshape(-1),
            positions=forward_batch.positions,
            rows=Rows(
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
            ),
            kv=KVTables(
                write_slots=forward_batch.out_cache_loc, page_table=self._page_table
            ),
            outputs=OutputSpec(logit_rows=LogitRows.NONE),
            graph=draft_graph_choice(graph_runner, forward_batch),
            spec=SpecInputs(
                num_draft_tokens=self._num_draft_tokens,
                num_steps=self._num_steps,
                topk=self._topk,
                hidden=spec_info.hidden_states,
            ),
        )
        out = self._run(step, forward_batch, run)
        tree = out.draft
        return tree.parents, tree.scores, tree.tokens, tree.probs

    def draft_extend(
        self,
        forward_batch: ForwardBatch,
        *,
        run: Callable[[], LogitsProcessorOutput],
        selected_rows: bool,
        graph_runner: Optional[Any] = None,
        model_runner: Optional[Any] = None,
    ) -> LogitsProcessorOutput:
        """One draft forward over accepted tokens or a prompt, fed the
        target's hidden states; ``run`` is the worker's native forward, which
        replays ``graph_runner`` (a draft-extend graph, returning only each
        request's last accepted row: ``selected_rows``) or runs
        ``model_runner.forward``, whose own graph choice the Step reports."""
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.model_executor.forward_mode import CaptureHiddenMode
        from sglang.srt.model_runner_plugins.native_steps import native_graph_choice

        extend = forward_batch.extend_start_loc is not None
        step = Step(
            mode=StepMode.DRAFT_EXTEND,
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
                prefix_lens_cpu=(
                    forward_batch.extend_prefix_lens_cpu if extend else None
                ),
            ),
            kv=KVTables(
                write_slots=forward_batch.out_cache_loc, page_table=self._page_table
            ),
            outputs=OutputSpec(
                logit_rows=(
                    LogitRows.LAST if extend or selected_rows else LogitRows.ALL
                ),
                hidden=(
                    HiddenRows.NONE
                    if forward_batch.capture_hidden_mode == CaptureHiddenMode.NULL
                    else HiddenRows.LAST
                    if forward_batch.capture_hidden_mode == CaptureHiddenMode.LAST
                    else HiddenRows.ALL
                ),
            ),
            graph=(
                native_graph_choice(
                    model_runner=model_runner, forward_batch=forward_batch
                )
                if model_runner is not None
                else draft_graph_choice(graph_runner, forward_batch)
            ),
            spec=SpecInputs(
                num_draft_tokens=self._num_draft_tokens,
                hidden=forward_batch.spec_info.hidden_states,
            ),
        )
        out = self._run(step, forward_batch, run)
        return LogitsProcessorOutput(
            next_token_logits=out.logits.tensor, hidden_states=out.hidden
        )

    def _run(self, step: Step, forward_batch: ForwardBatch, run) -> StepOutput:
        trace = self._trace
        tic = time.perf_counter()
        with self._executor.bind(step=step, run=run):
            plan = self._executor.plan(step)
        execute_tic = time.perf_counter()
        out = self._executor.execute(plan)
        execute_seconds = time.perf_counter() - execute_tic
        self.steps_executed += 1
        self._executor.release(out, None)
        if trace is not None:
            trace.record_draft(
                tic=tic,
                runner=self.runner_name,
                execute_seconds=execute_seconds,
                step=step,
                forward_batch=forward_batch,
                out=out,
            )
        return out


def draft_graph_choice(runner, forward_batch: ForwardBatch) -> GraphChoice:
    """The draft graph ``runner`` replays for ``forward_batch``: its padded
    bucket of requests; eager without a runner."""
    if runner is None:
        return EAGER
    rows = forward_batch.batch_size
    captured = getattr(runner, "capture_bs", None)
    if captured and rows <= captured[-1] and hasattr(runner, "_pad_to_bucket"):
        rows = runner._pad_to_bucket(rows, captured)
    return GraphChoice(phase=GraphPhase.DECODE, bucket=rows)
