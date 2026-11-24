"""Fused greedy sampling through the model runner API.

The reference runner declares fused sampling: its decode graphs take the
greedy argmax, and SGLang makes it the sampling owner of every all-greedy
batch with nothing that reads or edits the distribution. Served with overlap
scheduling and decode graphs, a greedy batch runs with the runner as owner,
then the same batch with logprobs requested runs with SGLang as owner: same
tokens, and the logits of every step the two runs share bitwise; a batch
mixing a sampled request keeps SGLang the owner of each of its steps."""

import os
import tempfile
import unittest

import requests
import torch
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.model_runner_reference import (
    REFERENCE_RUNNER,
    install_reference_runner,
    read_step_trace,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=300, stage="extra-a", runner_config="1-gpu-large")

_MODEL = "Qwen/Qwen3-0.6B"
_PROMPTS = [
    "The capital of France is",
    "Write a haiku about the sea:",
    "def fib(n):\n    if n < 2:\n        return n\n",
]
_KEY = ("input_ids", "positions", "seq_lens", "req_pool_indices")


def _generate(input_ids, **extra):
    response = requests.post(
        DEFAULT_URL_FOR_TEST + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": [
                {"temperature": 0, "max_new_tokens": 32} for _ in input_ids
            ],
            **extra,
        },
        timeout=300,
    )
    response.raise_for_status()
    return [r["output_ids"] for r in response.json()]


def _by_key(steps):
    by_key, repeated = {}, set()
    for r in steps:
        if "tensors" not in r:
            continue
        key = tuple(r["tensors"][k].numpy().tobytes() for k in _KEY) + (r["mode"],)
        if key in by_key:
            repeated.add(key)
        by_key[key] = r
    return {k: r for k, r in by_key.items() if k not in repeated}


class TestFusedSampling(CustomTestCase):
    def test_runner_sampled_greedy_batches_match_sglang_sampling(self):
        tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        ids = [tokenizer.encode(p) for p in _PROMPTS]
        trace_dir = tempfile.mkdtemp()
        process = popen_launch_server(
            _MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--model-runner",
                REFERENCE_RUNNER,
                "--disable-flashinfer-autotune",
                "--disable-prefill-cuda-graph",
                "--cuda-graph-max-bs-decode",
                "4",
                "--max-total-tokens",
                "20000",
                # Each batch starts cold, so the two runs build the same steps.
                "--disable-radix-cache",
            ],
            env={
                **os.environ,
                **install_reference_runner(tempfile.mkdtemp()),
                "SGLANG_MODEL_RUNNER_STEP_TRACE_DIR": trace_dir,
                "SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS": "1",
            },
        )
        try:
            fused = _generate(ids)
            n_fused = len(read_step_trace(trace_dir))
            sglang = _generate(ids, return_logprob=True)
            n_sglang = len(read_step_trace(trace_dir))
            mixed = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "input_ids": ids[:2],
                    "sampling_params": [
                        {"temperature": 0, "max_new_tokens": 8},
                        {"temperature": 0.8, "max_new_tokens": 8, "sampling_seed": 1},
                    ],
                },
                timeout=300,
            )
            mixed.raise_for_status()
        finally:
            terminate_and_kill_process_tree(process)
        steps = sorted(read_step_trace(trace_dir), key=lambda r: r["index"])
        runs = {
            "fused": steps[:n_fused],
            "sglang": steps[n_fused:n_sglang],
            "mixed": steps[n_sglang:],
        }
        self.assertEqual(fused, sglang)
        self.assertEqual({r["runner"] for r in steps}, {REFERENCE_RUNNER})
        owners = {
            name: {r["sampling_owner"] for r in run if r["mode"] != "idle"}
            for name, run in runs.items()
        }
        self.assertEqual(
            owners, {"fused": {"plugin"}, "sglang": {"sglang"}, "mixed": {"sglang"}}
        )
        replayed = sum(r["replayed_graph"] for r in runs["fused"])
        self.assertGreater(replayed, 0)
        want = _by_key(runs["sglang"])
        matched = 0
        for key, got in _by_key(runs["fused"]).items():
            if key in want:
                matched += 1
                self.assertTrue(
                    torch.equal(
                        got["tensors"]["logits"], want[key]["tensors"]["logits"]
                    ),
                    f"step {got['index']} ({got['mode']}) logits differ",
                )
        self.assertGreater(matched, 0)
        outside = sorted(
            r["host_us"] - r["execute_us"]
            for r in runs["fused"]
            if r["mode"] == "decode"
        )
        print(
            f"[fused sampling] steps: runner-sampled {len(runs['fused'])} "
            f"({replayed} graph replays), SGLang-sampled {len(runs['sglang'])}, "
            f"mixed {len(runs['mixed'])}; {matched} shared steps bitwise; median "
            f"decode host time outside execute {outside[len(outside) // 2]:.0f} us"
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
