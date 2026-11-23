"""The controlled bitwise fixture for --model-runner sglang steps (TP1, eager,
overlap off): raw logits before sampling are bitwise equal to the direct path's
at identical batches, row order, positions and KV slots; prompt logprobs, which
the built-in runner returns natively, are equal too. Each comparison is first
made direct against direct, to show the configuration repeats. Also reports the
host time each path spends at the worker boundary."""

import os
import statistics
import tempfile
import unittest

import requests
import torch
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.model_runner_reference import read_step_trace
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-small")

_MODEL = "Qwen/Qwen3-0.6B"
_EAGER = [
    "--disable-overlap-schedule",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
    # Startup autotuning takes minutes on GB300 and changes nothing compared here.
    "--disable-flashinfer-autotune",
]
# Longer than two KV pages (64 tokens on Blackwell backends), so a follow-up
# reuses whole cached pages.
_PREAMBLE = "Answer in one short sentence, plainly and without lists. " * 12
_PROMPTS = [
    _PREAMBLE + "The capital of France is",
    "Write a haiku about the sea:",
    _PREAMBLE + "1, 2, 3, 4,",
    "In 1969, the first person to walk on the moon was",
]
# Each shares a cached prefix with a prompt of the first batch.
_FOLLOW_UPS = [
    _PREAMBLE + "The capital of France is a city that",
    _PREAMBLE + "1, 2, 3, 4, 5, 6,",
]
_SIGNATURE = ("input_ids", "positions", "out_cache_loc", "seq_lens", "req_pool_indices")


def _generate(input_ids, **extra):
    response = requests.post(
        DEFAULT_URL_FOR_TEST + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            **extra,
        },
    )
    response.raise_for_status()
    return response.json()


def _serve(other_args, *, save_logits, workload):
    """Launch, run ``workload``, shut down; return its result and the trace."""
    trace_dir = tempfile.mkdtemp()
    env = {
        **os.environ,
        "SGLANG_MODEL_RUNNER_STEP_TRACE_DIR": trace_dir,
        "SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS": "1" if save_logits else "0",
    }
    process = popen_launch_server(
        _MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
        env=env,
    )
    try:
        result = workload()
    finally:
        terminate_and_kill_process_tree(process)
    return result, read_step_trace(trace_dir)


class TestModelRunnerStepBitwise(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        cls.prompts = [tokenizer.encode(p) for p in _PROMPTS]
        cls.follow_ups = [tokenizer.encode(p) for p in _FOLLOW_UPS]

    def _workload(self):
        # Pre-tokenized batches reach the scheduler as one message each, so the
        # batches, their row order and their KV slots repeat across launches.
        first = _generate(self.prompts)
        second = _generate(self.follow_ups)
        logprobs = _generate(
            self.follow_ups[0],
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=3,
        )
        return first, second, logprobs

    def _assert_same_steps(self, expected, actual):
        self.assertEqual(len(actual), len(expected))
        compared = 0
        for want, got in zip(expected, actual):
            self.assertEqual(got["mode"], want["mode"])
            if "tensors" not in want:
                continue
            compared += 1
            for name in _SIGNATURE:
                torch.testing.assert_close(
                    got["tensors"][name], want["tensors"][name], rtol=0, atol=0
                )
            want_logits, got_logits = (
                want["tensors"]["logits"],
                got["tensors"]["logits"],
            )
            self.assertEqual(got_logits.shape, want_logits.shape)
            self.assertTrue(
                torch.equal(got_logits, want_logits),
                f"step {want['index']} ({want['mode']}): logits differ, max abs "
                f"{(got_logits.float() - want_logits.float()).abs().max().item()}",
            )
        return compared

    def test_step_logits_are_bitwise_equal_to_the_direct_path(self):
        runs = {
            name: _serve(args, save_logits=True, workload=self._workload)
            for name, args in (
                ("direct", _EAGER),
                ("direct_again", _EAGER),
                ("steps", [*_EAGER, "--model-runner", "sglang"]),
            )
        }
        (direct_out, direct), (again_out, again), (steps_out, steps) = runs.values()

        self.assertEqual({r["path"] for r in direct + again}, {"direct"})
        self.assertEqual(
            {(r["path"], r["runner"]) for r in steps}, {("step", "sglang")}
        )
        modes = {r["mode"] for r in steps}
        self.assertTrue({"extend", "decode"} <= modes, modes)
        self.assertTrue(any(any(r["prefix_lens"] or ()) for r in steps))
        self.assertTrue(any(r["prompt_logprobs"] for r in steps))

        # Direct against direct first: the fixture repeats bitwise.
        repeat = self._assert_same_steps(direct, again)
        self._assert_same_responses(direct_out, again_out)
        through_steps = self._assert_same_steps(direct, steps)
        self._assert_same_responses(direct_out, steps_out)
        print(
            f"bitwise logits: {repeat} steps direct vs direct, "
            f"{through_steps} steps direct vs steps"
        )
        self.assertGreater(through_steps, 0)

    def _assert_same_responses(self, expected, actual):
        # Request ids and timings differ per launch; tokens and logprobs must not.
        self.assertEqual(len(actual), len(expected))
        for want, got in zip(expected, actual):
            for key in ("output_ids", "meta_info"):
                self._assert_equal_ignoring_timing(want, got, key)

    def _assert_equal_ignoring_timing(self, want, got, key):
        if isinstance(want, list):
            for w, g in zip(want, got):
                self._assert_equal_ignoring_timing(w, g, key)
            return
        if key == "meta_info":
            fields = (
                "input_token_logprobs",
                "output_token_logprobs",
                "input_top_logprobs",
                "output_top_logprobs",
                "cached_tokens",
            )
            for field in fields:
                self.assertEqual(want[key].get(field), got[key].get(field), field)
        else:
            self.assertEqual(want[key], got[key])

    def test_step_overhead_at_the_worker_boundary(self):
        """Host time per forward at TpModelWorker.forward_model, direct vs steps.

        Reported, not gated: the step path adds building the Step views, plan
        and the output conversion around the same ModelRunner.forward.
        """

        def decode_heavy():
            for batch in (self.prompts[:1], self.prompts):
                requests.post(
                    DEFAULT_URL_FOR_TEST + "/generate",
                    json={
                        "input_ids": batch,
                        "sampling_params": {
                            "max_new_tokens": 64,
                            "temperature": 0,
                            "ignore_eos": True,
                        },
                    },
                ).raise_for_status()

        _, direct = _serve(_EAGER, save_logits=False, workload=decode_heavy)
        _, steps = _serve(
            [*_EAGER, "--model-runner", "sglang"],
            save_logits=False,
            workload=decode_heavy,
        )
        report = []
        for mode in ("extend", "decode"):
            for batch_size in (1, len(self.prompts)):
                rows = {
                    name: [
                        r
                        for r in trace[2:]  # skip warmup
                        if r["mode"] == mode and r["batch_size"] == batch_size
                    ]
                    for name, trace in (("direct", direct), ("steps", steps))
                }
                if not all(rows.values()):
                    continue
                direct_us = statistics.median(r["host_us"] for r in rows["direct"])
                steps_us = statistics.median(r["host_us"] for r in rows["steps"])
                outside = statistics.median(
                    r["host_us"] - r["execute_us"] for r in rows["steps"]
                )
                report.append(
                    f"{mode} bs={batch_size}: direct {direct_us:.0f} us, "
                    f"steps {steps_us:.0f} us (outside execute {outside:.0f} us), "
                    f"n={len(rows['direct'])}/{len(rows['steps'])}"
                )
        print("STEP OVERHEAD (median host time at forward_model)")
        print("\n".join(report))
        self.assertTrue(report)


if __name__ == "__main__":
    unittest.main(verbosity=3)
