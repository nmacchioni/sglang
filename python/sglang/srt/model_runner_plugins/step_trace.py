"""Per-forward trace at the worker boundary, for tests and step-overhead runs.

Off unless ``SGLANG_MODEL_RUNNER_STEP_TRACE_DIR`` is set. Each record says which
path ran (direct, or the runner's steps), the step's shape, and the host time
spent at the boundary; optionally the inputs and raw logits before sampling.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class StepTrace:
    @classmethod
    def maybe_create(cls) -> Optional[StepTrace]:
        directory = envs.SGLANG_MODEL_RUNNER_STEP_TRACE_DIR.get()
        if directory is None:
            return None
        return cls(
            directory=directory,
            save_logits=envs.SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS.get(),
        )

    def __init__(self, *, directory: str, save_logits: bool):
        os.makedirs(directory, exist_ok=True)
        self._prefix = os.path.join(directory, f"steps-{os.getpid()}")
        self._save_logits = save_logits
        self._count = 0

    def start(self) -> float:
        return time.perf_counter()

    def record(
        self,
        *,
        tic: float,
        runner: Optional[str],
        execute_seconds: Optional[float],
        forward_batch: ForwardBatch,
        logits_output: LogitsProcessorOutput,
    ) -> None:
        host_seconds = time.perf_counter() - tic
        record = {
            "index": self._count,
            "path": "direct" if runner is None else "step",
            "runner": runner,
            "mode": forward_batch.forward_mode.name.lower(),
            "batch_size": forward_batch.batch_size,
            "num_tokens": int(forward_batch.input_ids.shape[0]),
            "prefix_lens": forward_batch.extend_prefix_lens_cpu,
            "host_us": host_seconds * 1e6,
            "execute_us": None if execute_seconds is None else execute_seconds * 1e6,
            "prompt_logprobs": logits_output.input_token_logprobs is not None,
        }
        with open(f"{self._prefix}.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        if self._save_logits and logits_output.next_token_logits is not None:
            torch.save(
                {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in (
                        ("input_ids", forward_batch.input_ids),
                        ("positions", forward_batch.positions),
                        ("out_cache_loc", forward_batch.out_cache_loc),
                        ("seq_lens", forward_batch.seq_lens),
                        ("req_pool_indices", forward_batch.req_pool_indices),
                        ("logits", logits_output.next_token_logits),
                    )
                },
                f"{self._prefix}-{self._count:05d}.pt",
            )
        self._count += 1
