"""Diffusion LLMs through the model runner API.

SDAR-1.7B-Chat (blocks of 4) with the LowConfidence algorithm, in synchronous
mode (every denoising pass of a block within one forward call) and in
first-done-first-out mode (one pass per scheduler step), eager (its decode
graphs hit an illegal instruction on the direct path here, with the Triton and
FlashInfer backends alike): direct, direct again, then with
``--model-runner sglang``. Identical
outputs, raw logits of every pass bitwise in order, every forward on the step
path as DENOISE steps (context encoding as EXTEND), and in synchronous mode
the later passes of a block re-executing its plan."""

import os
import tempfile
import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.model_runner_reference import read_step_trace
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=900, stage="extra-a", runner_config="1-gpu-large")

_MODEL = "JetLM/SDAR-1.7B-Chat"
_COMMON = [
    "--trust-remote-code",
    "--disable-flashinfer-autotune",
    "--mem-fraction-static",
    "0.5",
    "--max-running-requests",
    "4",
    "--dllm-algorithm",
    "LowConfidence",
    # On the direct path here, the default Blackwell backend has no kernel for
    # this block layout.
    "--attention-backend",
    "triton",
]
_EAGER = ["--disable-decode-cuda-graph", "--disable-prefill-cuda-graph"]
_PROMPTS = [
    "<|im_start|>user\nWrite the numbers from 1 to 20.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
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


def _workload():
    out = []
    for prompts in (_PROMPTS[:1], _PROMPTS):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": prompts,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
            timeout=600,
        )
        response.raise_for_status()
        out.append([r["output_ids"] for r in response.json()])
    return out


def _serve(args):
    trace_dir = tempfile.mkdtemp()
    process = popen_launch_server(
        _MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=[*_COMMON, *args],
        env={
            **os.environ,
            "SGLANG_MODEL_RUNNER_STEP_TRACE_DIR": trace_dir,
            "SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS": "1",
        },
    )
    try:
        out = _workload()
    finally:
        terminate_and_kill_process_tree(process)
    return out, read_step_trace(trace_dir)


class TestModelRunnerDiffusion(CustomTestCase):
    def _assert_steps_bitwise(self, want, got, *, output="logits"):
        """Steps matched by inputs bitwise; every step of one run whose inputs
        the other run has is compared, and few have none."""
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
        self.assertLessEqual(
            len(got_seen - want_seen), 0.1 * len(got_seen), "steps without a match"
        )
        self.assertTrue(matched)
        return len(matched)

    def _case(self, name, args):
        direct, direct_steps = _serve(args)
        again, again_steps = _serve(args)
        api, api_steps = _serve([*args, "--model-runner", "sglang"])
        self.assertEqual(again, direct)
        self.assertEqual(api, direct)
        repeat = self._assert_steps_bitwise(direct_steps, again_steps)
        through_api = self._assert_steps_bitwise(direct_steps, api_steps)
        self.assertEqual({r["path"] for r in api_steps}, {"step"})
        passes = [r for r in api_steps if r["mode"] == "dllm_extend"]
        self.assertTrue(passes)
        reruns = sum(r["plan_runs"] > 1 for r in passes)
        outside = sorted(r["host_us"] - r["execute_us"] for r in passes)
        boundary = {
            label: sorted(r["host_us"] for r in run if r["mode"] == "dllm_extend")
            for label, run in (("direct", direct_steps), ("API", api_steps))
        }
        print(
            f"[diffusion {name}] steps bitwise: direct again {repeat}, API "
            f"{through_api}; denoising passes {len(passes)}, of them plan "
            f"re-executions {reruns}; graph replays "
            f"{sum(r['replayed_graph'] for r in passes)}; median host time "
            f"outside execute {outside[len(outside) // 2]:.0f} us; at the worker "
            f"boundary {', '.join(f'{k} {v[len(v) // 2]:.0f} us' for k, v in boundary.items())}"
        )
        return reruns

    def test_synchronous_eager(self):
        self.assertGreater(self._case("sync eager", [*_EAGER, "--no-dllm-fdfo"]), 0)

    def test_first_done_first_out_eager(self):
        self._case("fdfo eager", _EAGER)


if __name__ == "__main__":
    unittest.main(verbosity=3)
