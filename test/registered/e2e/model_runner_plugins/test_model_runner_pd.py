"""Prefill/decode disaggregation through the model runner API.

Each case launches a prefill server, a decode server and the router with the KV
checksum on (the decode digests the transferred bytes and aborts a request
whose digest differs from the prefill's), runs the same requests, and keeps
the step trace of both roles:

- the built-in runner on both roles, against the same PD launch on the direct
  path, after the direct launch is repeated: identical tokens, and the raw
  logits of every prefill and decode step bitwise where the direct repeat
  reproduces them (plain, sliding-window and recurrent-state models);
- the reference (external) runner on both roles: SGLang registers and moves the
  runner's own KV bytes, located through its KVLayout; the decode's first step
  after each transfer reads that KV, and its logits and every continuation
  are bitwise equal to the same runner serving without PD;
- an asymmetric topology (prefill TP2, decode TP1), built-in runner, the same
  way.
"""

import os
import statistics
import tempfile
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.model_runner_reference import (
    REFERENCE_RUNNER,
    install_reference_runner,
    read_step_trace,
)
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=1800, stage="extra-b", runner_config="4-gpu-h100")

_MODEL = "Qwen/Qwen3-0.6B"
_COMMON = [
    "--disaggregation-enable-kv-checksum",
    "--disable-flashinfer-autotune",
    "--mem-fraction-static",
    "0.5",
    "--disable-overlap-schedule",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
]
_PROMPTS = [
    "Answer in one short sentence, plainly and without lists. " * 12
    # Prompts of different lengths, so no two requests decode at one position.
    + "Please " * i
    + f"Question {i}: what is the capital of {country}?"
    for i, country in enumerate(("France", "Spain", "Italy", "Norway"))
]
_STEP_KEY = ("input_ids", "positions", "seq_lens")


def _key(record):
    return tuple(record["tensors"][k].numpy().tobytes() for k in _STEP_KEY) + (
        record["mode"],
    )


def _unique_by_key(trace) -> dict:
    by_key, repeated = {}, set()
    for record in trace:
        if "tensors" not in record:
            continue
        key = _key(record)
        if key in by_key:
            repeated.add(key)
        by_key[key] = record
    return {key: r for key, r in by_key.items() if key not in repeated}


def _compare_logits(want_steps, got_steps, *, modes):
    want_by_key = _unique_by_key(want_steps)
    matched, differ = 0, []
    for key, got in _unique_by_key(got_steps).items():
        want = want_by_key.get(key)
        if want is None or got["mode"] not in modes:
            continue
        matched += 1
        if not torch.equal(got["tensors"]["logits"], want["tensors"]["logits"]):
            differ.append((got["index"], got["mode"]))
    return matched, differ


def _roles(trace):
    """The trace split by role: the decode's file is the one with decode steps."""
    by_pid = {}
    for record in trace:
        by_pid.setdefault(record["source"], []).append(record)
    decode = [r for r in by_pid.values() if any(x["mode"] == "decode" for x in r)]
    prefill = [r for r in by_pid.values() if all(x["mode"] != "decode" for x in r)]
    # Every prefill TP rank runs the same steps; rank 0's stand for them.
    prefill = [x for r in prefill for x in r if x["tp_rank"] == 0]
    return prefill, [x for r in decode for x in r]


class _PDRuns(PDDisaggregationServerBase):
    """Launches PD several times in one test: setUpClass sets the fixture up
    without launching, ``_serve_pd`` runs one launch."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = _MODEL
        cls.reference_env = install_reference_runner(tempfile.mkdtemp())

    def _generate(self, url, prompt, max_new_tokens=24):
        response = requests.post(
            url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
                "return_logprob": True,
            },
            timeout=300,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _workload(self, url):
        # One request at a time, so every launch builds the same batches.
        return [self._generate(url, p) for p in _PROMPTS]

    def _serve_pd(self, *, prefill_args, decode_args, env=None):
        cls = type(self)
        trace_dir = tempfile.mkdtemp()
        saved = dict(os.environ)
        os.environ.update(
            {
                **(env or {}),
                "SGLANG_MODEL_RUNNER_STEP_TRACE_DIR": trace_dir,
                "SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS": "1",
            }
        )
        cls.extra_prefill_args = [*_COMMON, *prefill_args]
        cls.extra_decode_args = [*_COMMON, *decode_args]
        try:
            cls.launch_all()
            out = self._workload(cls.lb_url)
        finally:
            self._stop()
            os.environ.clear()
            os.environ.update(saved)
        return out, read_step_trace(trace_dir)

    def _serve_alone(self, other_args, env=None):
        trace_dir = tempfile.mkdtemp()
        process = popen_launch_server(
            _MODEL,
            self.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[*_COMMON[1:], *other_args],
            env={
                **os.environ,
                **(env or {}),
                "SGLANG_MODEL_RUNNER_STEP_TRACE_DIR": trace_dir,
                "SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS": "1",
            },
        )
        try:
            out = self._workload(self.prefill_url)
        finally:
            terminate_and_kill_process_tree(process)
        return out, read_step_trace(trace_dir)

    @classmethod
    def _stop(cls):
        if cls._fail_fast_stop is not None:
            cls._fail_fast_stop.set()
            cls._fail_fast_stop = None
        if cls.process_lb:
            kill_process_tree(cls.process_lb.pid, wait_timeout=60)
        for process in (cls.process_decode, cls.process_prefill):
            if process:
                terminate_and_kill_process_tree(process, wait_timeout=60)
        cls.process_lb = cls.process_decode = cls.process_prefill = None

    def _assert_same_tokens(self, want, got):
        self.assertEqual(
            [o["output_ids"] for o in got], [o["output_ids"] for o in want]
        )
        for o in got:
            self.assertNotEqual(o["meta_info"]["finish_reason"]["type"], "abort")

    def _assert_matched_logits(self, want_steps, got_steps, *, modes):
        """Bitwise logits of the steps of ``modes`` whose inputs occur once in
        each run (different requests can share a token at the same position
        and length); returns how many were compared."""
        matched, differ = _compare_logits(want_steps, got_steps, modes=modes)
        self.assertEqual(differ, [], "steps whose logits differ")
        return matched

    def _builtin_case(self, name, *, model=_MODEL, args=()):
        """Direct PD, direct PD again, then PD through the API. Steps are
        compared bitwise when the direct repeat reproduces every step;
        otherwise the configuration does not repeat, and only tokens are."""
        type(self).model = model
        runner = [*args, "--model-runner", "sglang"]
        try:
            direct_out, direct = self._serve_pd(prefill_args=args, decode_args=args)
            again_out, again = self._serve_pd(prefill_args=args, decode_args=args)
            steps_out, steps = self._serve_pd(prefill_args=runner, decode_args=runner)
        finally:
            type(self).model = _MODEL
        prefill, decode = _roles(steps)
        self.assertTrue(prefill and decode)
        self.assertEqual({r["path"] for r in prefill + decode}, {"step"})
        self._assert_same_tokens(direct_out, again_out)
        self._assert_same_tokens(direct_out, steps_out)
        want = dict(zip(("extend", "decode"), _roles(direct)))
        report = {}
        for label, roles in (
            ("direct again", _roles(again)),
            ("steps", (prefill, decode)),
        ):
            for mode, got in zip(("extend", "decode"), roles):
                report[(label, mode)] = _compare_logits(want[mode], got, modes={mode})
        print(f"[{name}] (matched, differing) steps vs direct: {report}")
        for role, records in (("prefill", prefill), ("decode", decode)):
            outside = [r["host_us"] - r["execute_us"] for r in records[2:]]
            print(
                f"[{name}] {role}: median host time outside execute "
                f"{statistics.median(outside):.0f} us (n={len(outside)})"
            )
        for mode in ("extend", "decode"):
            matched, differ = report[("steps", mode)]
            self.assertGreater(matched, 0)
            if not report[("direct again", mode)][1]:
                self.assertEqual(differ, [], f"{mode} steps differ through the API")
        return prefill, decode


class TestModelRunnerPD(_PDRuns):
    def test_builtin_runner_pd_matches_the_direct_path(self):
        self._builtin_case("builtin PD")

    def test_builtin_runner_pd_with_a_sliding_window(self):
        # The window sub-pool is transferred beside the full-attention layers.
        _, decode = self._builtin_case(
            "builtin PD, sliding window", model="unsloth/gemma-3-1b-it"
        )
        self.assertTrue(all(r["kv_window"] for r in decode))

    def test_builtin_runner_pd_with_recurrent_state(self):
        # The request's state slot is transferred beside the attention layers.
        _, decode = self._builtin_case(
            "builtin PD, recurrent state", model="ibm-granite/granite-4.0-h-350m"
        )
        self.assertTrue(all(r["kv_state"] is not None for r in decode))

    def test_reference_runner_kv_moves_through_its_kv_layout(self):
        runner = ["--model-runner", REFERENCE_RUNNER]
        alone_out, alone = self._serve_alone(runner, env=self.reference_env)
        pd_out, pd = self._serve_pd(
            prefill_args=runner, decode_args=runner, env=self.reference_env
        )
        prefill, decode = _roles(pd)
        self.assertEqual({r["runner"] for r in prefill + decode}, {REFERENCE_RUNNER})
        # Resumed decode: every continuation token for token, after the decode
        # read the KV the prefill's runner wrote and SGLang moved.
        self._assert_same_tokens(alone_out, pd_out)
        extends = self._assert_matched_logits(alone, prefill, modes={"extend"})
        decodes = self._assert_matched_logits(alone, decode, modes={"decode"})
        print(
            f"[reference PD] steps bitwise vs serving alone: prefill {extends}, "
            f"decode {decodes}"
        )
        self.assertGreater(extends, 0)
        self.assertGreater(decodes, 0)


class TestModelRunnerPDAsymmetric(_PDRuns):
    prefill_tp_size = 2
    decode_tp_size = 1
    decode_base_gpu_id = 2

    def test_prefill_tp2_to_decode_tp1(self):
        self._builtin_case("asymmetric PD, prefill TP2 to decode TP1")


if __name__ == "__main__":
    unittest.main(verbosity=3)
