"""SGLang's graph decision for a step: which captured bucket replays it, or
eager. External runners replay the buckets SGLang passed to ``capture``; the
built-in runner's own graph runners decide, and ``Step.graph`` reports it."""

from __future__ import annotations

import bisect

from sglang.srt.model_runner_plugins.api import (
    EAGER,
    GraphBuckets,
    GraphChoice,
    GraphPhase,
    StepMode,
)


def bucket_graph_choice(
    *, mode: StepMode, batch_size: int, num_tokens: int, buckets: GraphBuckets
) -> GraphChoice:
    """The smallest captured bucket that holds the step, else eager.

    Decode buckets count requests and prefill buckets input tokens; idle steps
    run eager, since they produce nothing on one GPU, and so do verify steps:
    the buckets SGLang passes to ``capture`` hold one token per request.
    """
    if mode is StepMode.DECODE:
        phase, rows = GraphPhase.DECODE, batch_size
    elif mode in (StepMode.EXTEND, StepMode.MIXED):
        phase, rows = GraphPhase.PREFILL, num_tokens
    else:
        return EAGER
    captured = buckets.of(phase)
    if rows <= 0 or not captured or rows > captured[-1]:
        return EAGER
    return GraphChoice(phase=phase, bucket=captured[bisect.bisect_left(captured, rows)])
