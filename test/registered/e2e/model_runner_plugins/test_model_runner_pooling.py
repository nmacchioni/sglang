"""Pooling models and position scoring through the model runner API.

Each configuration serves the same requests direct, direct again, then with
``--model-runner sglang``: identical responses, the pooled outputs (or
logits) bitwise for every step whose inputs the other run also has, and every
step of the API run on the step path. Configurations: an embedding model, a reward model (a sequence
classification head), a cross-encoder (token type ids, an encoder-only model),
scores from a generation model, and multi-item scoring (delimiter rows)."""

import os
import tempfile
import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.model_runner_reference import read_step_trace, with_model_runner
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=900, stage="extra-a", runner_config="1-gpu-large")

_COMMON = ["--disable-flashinfer-autotune", "--mem-fraction-static", "0.4"]
_TEXTS = [
    "The capital of France is Paris.",
    "A haiku about the sea, written at dawn.",
    "def fib(n): return n if n < 2 else fib(n - 1) + fib(n - 2)",
    "Norway's fjords were carved by glaciers.",
]
_SIGNATURE = ("input_ids", "positions", "seq_lens")


def _by_inputs(steps, output):
    """Steps keyed by their inputs: those whose inputs occur once in the run,
    and every key seen. The server's health checks repeat their inputs a
    number of times that varies between runs."""
    by_key, seen = {}, {}
    for r in steps:
        if "tensors" not in r or output not in r["tensors"]:
            continue
        key = tuple(r["tensors"][k].numpy().tobytes() for k in _SIGNATURE) + (
            r["mode"],
        )
        seen[key] = seen.get(key, 0) + 1
        by_key[key] = r
    return {key: r for key, r in by_key.items() if seen[key] == 1}, set(seen)


def _post(path, payload):
    response = requests.post(DEFAULT_URL_FOR_TEST + path, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def _embed(model):
    one_by_one = [_post("/v1/embeddings", {"model": model, "input": t}) for t in _TEXTS]
    batched = _post("/v1/embeddings", {"model": model, "input": _TEXTS})
    return [[d["embedding"] for d in r["data"]] for r in [*one_by_one, batched]]


def _classify(model):
    one_by_one = [_post("/classify", {"text": t}) for t in _TEXTS]
    return [r["embedding"] for r in one_by_one] + [
        [r["embedding"] for r in _post("/classify", {"text": _TEXTS})]
    ]


def _rerank(model):
    ranked = _post(
        "/v1/rerank",
        {"query": "Where are fjords?", "documents": _TEXTS, "model": model},
    )
    # Each result also carries a fresh request id.
    return [(r["index"], r["score"]) for r in ranked]


def _score(model):
    return [
        _post(
            "/v1/score",
            {
                "query": "Is this sentence about geography? ",
                "items": items,
                "label_token_ids": [9454, 2753],
                "apply_softmax": True,
                "model": model,
            },
        )["scores"]
        for items in (_TEXTS[:1], _TEXTS, _TEXTS[1:3])
    ]


_CONFIGS = {
    "embedding": (
        "Qwen/Qwen3-Embedding-0.6B",
        ["--is-embedding"],
        _embed,
    ),
    "reward": (
        "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
        ["--is-embedding", "--disable-radix-cache"],
        _classify,
    ),
    "cross_encoder": (
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        ["--is-embedding", "--disable-radix-cache", "--attention-backend", "triton"],
        _rerank,
    ),
    "score": ("Qwen/Qwen3-0.6B", [], _score),
    "multi_item_score": (
        "Qwen/Qwen3-0.6B",
        [
            "--disable-radix-cache",
            "--chunked-prefill-size",
            "-1",
            "--enable-mis",
            "--attention-backend",
            "flashinfer",
        ],
        _score,
    ),
}


def _serve(model, args, workload):
    trace_dir = tempfile.mkdtemp()
    process = popen_launch_server(
        model,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=with_model_runner([*_COMMON, *args]),
        env={
            **os.environ,
            "SGLANG_MODEL_RUNNER_STEP_TRACE_DIR": trace_dir,
            "SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS": "1",
        },
    )
    try:
        # The launch's health checks run a number of steps that varies.
        launch_steps = len(read_step_trace(trace_dir))
        out = workload(model)
    finally:
        terminate_and_kill_process_tree(process)
    steps = sorted(read_step_trace(trace_dir), key=lambda r: r["index"])
    return out, steps[launch_steps:]


class TestModelRunnerPooling(CustomTestCase):
    def _assert_steps_bitwise(self, want, got, *, output="logits"):
        """Steps matched by inputs bitwise: every step of one run whose inputs
        the other run has."""
        (want, want_seen), (got, got_seen) = (
            _by_inputs(want, output),
            _by_inputs(got, output),
        )
        matched = [key for key in got if key in want]
        for key in matched:
            a, b = want[key], got[key]
            self.assertTrue(
                torch.equal(a["tensors"][output], b["tensors"][output]),
                f"step {b['index']} ({b['mode']}): {output} differ",
            )
        # A batch request's items reach the scheduler one by one, so how they
        # batch depends on timing, on the direct path too, and a run can share
        # no step with another; the responses are compared regardless.
        # Multi-item scoring steps return no next-token logits to compare
        # (their scores are the prompt logprobs at the delimiters).
        return f"{len(matched)}/{len(got_seen)}"

    def _case(self, name):
        model, args, workload = _CONFIGS[name]
        direct, direct_steps = _serve(model, args, workload)
        again, again_steps = _serve(model, args, workload)
        api, api_steps = _serve(model, [*args, "--model-runner", "sglang"], workload)
        self.assertEqual(again, direct)
        self.assertEqual(api, direct)
        self.assertEqual({r["path"] for r in api_steps}, {"step"})
        pooled = all(r["pooled"] for r in api_steps)
        output = "pooled" if pooled else "logits"
        repeat = self._assert_steps_bitwise(direct_steps, again_steps, output=output)
        through_api = self._assert_steps_bitwise(direct_steps, api_steps, output=output)
        # Launch steps are already left out, so no warmup step remains.
        outside = sorted(r["host_us"] - r["execute_us"] for r in api_steps)
        boundary = {
            label: sorted(r["host_us"] for r in run)
            for label, run in (("direct", direct_steps), ("API", api_steps))
        }
        print(
            f"[{name}] steps bitwise ({output}): direct again {repeat}, API "
            f"{through_api}; median host time outside execute "
            f"{outside[len(outside) // 2]:.0f} us (n={len(outside)}); at the worker "
            f"boundary {', '.join(f'{k} {v[len(v) // 2]:.0f} us' for k, v in boundary.items())}"
        )
        return api_steps

    def test_embedding(self):
        steps = self._case("embedding")
        self.assertTrue(all(r["pooled"] for r in steps))

    def test_reward_model(self):
        steps = self._case("reward")
        self.assertTrue(all(r["pooled"] for r in steps))

    def test_cross_encoder(self):
        steps = self._case("cross_encoder")
        self.assertTrue(all(r["pooled"] for r in steps))

    def test_scores_from_a_generation_model(self):
        # Each item is a request scored from its next-token logits.
        self._case("score")

    def test_multi_item_scoring(self):
        steps = self._case("multi_item_score")
        self.assertTrue(any(r["prompt_logprobs"] for r in steps))


if __name__ == "__main__":
    unittest.main(verbosity=3)
