"""The controlled bitwise fixture for --model-runner sglang steps (TP1): raw
logits before sampling are bitwise equal to the native lane's at identical
batches, row order, positions and KV slots; prompt logprobs, which the built-in
runner returns natively, are equal too. Eager with overlap off, CUDA graphs,
overlap scheduling, and both are compared separately, each with the same
configuration on both paths, and each first native against native, to show the
configuration repeats. Also reports the host time each path spends at the
worker boundary."""

import json
import os
import statistics
import tempfile
import threading
import time
import unittest

import requests
import torch
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.model_runner_reference import native_lane, read_step_trace
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=1500, stage="base-b", runner_config="1-gpu-small")

_MODEL = "Qwen/Qwen3-0.6B"
# Startup autotuning takes minutes on GB300 and changes nothing compared here.
_COMMON = ["--disable-flashinfer-autotune"]
_NO_OVERLAP = ["--disable-overlap-schedule"]
_NO_GRAPHS = ["--disable-decode-cuda-graph", "--disable-prefill-cuda-graph"]
# Small buckets, so the batches below run both replayed and eager: decode of
# four requests and the four-prompt prefill fall outside them.
_GRAPHS = [
    "--cuda-graph-max-bs-decode",
    "2",
    "--cuda-graph-max-bs-prefill",
    "256",
    "--enable-mixed-chunk",
]
_EAGER = [*_NO_OVERLAP, *_NO_GRAPHS, *_COMMON]
_CONFIGS = {
    "eager": _EAGER,
    "graphs": [*_NO_OVERLAP, *_GRAPHS, *_COMMON],
    "overlap": [*_NO_GRAPHS, *_COMMON],
    "overlap_graphs": [*_GRAPHS, *_COMMON],
}
_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"city": {"type": "string"}, "rank": {"type": "integer"}},
        "required": ["city", "rank"],
    }
)
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


def _generate(input_ids, *, max_new_tokens=8, sampling=None, **extra):
    response = requests.post(
        DEFAULT_URL_FOR_TEST + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0,
                **(sampling or {}),
            },
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
        env={**env, **native_lane(other_args)},
    )
    try:
        result = workload()
    finally:
        terminate_and_kill_process_tree(process)
    return result, read_step_trace(trace_dir)


def _serve_untraced(other_args, *, workload, log=False):
    """Launch without the step trace, run ``workload``, shut down; with
    ``log``, also return the server's output."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("SGLANG_MODEL_RUNNER_STEP_TRACE")
    }
    with tempfile.TemporaryFile("w+") as out:
        process = popen_launch_server(
            _MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={**env, **native_lane(other_args)},
            return_stdout_stderr=(out, out) if log else None,
        )
        try:
            result = workload()
        finally:
            terminate_and_kill_process_tree(process)
        if not log:
            return result
        out.seek(0)
        return result, out.read()


def _unique_by_inputs(trace) -> dict:
    """Steps keyed by their inputs, without keys that repeat in the run: a
    freed partial page can put another request's token at the same position
    and slot, over different earlier KV."""
    by_inputs, repeated = {}, set()
    for record in trace:
        if "tensors" not in record:
            continue
        key = _inputs_key(record)
        if key in by_inputs:
            repeated.add(key)
        by_inputs[key] = record
    return {key: r for key, r in by_inputs.items() if key not in repeated}


def _inputs_key(record):
    return tuple(record["tensors"][name].numpy().tobytes() for name in _SIGNATURE) + (
        record["mode"],
    )


def _assert_bitwise(test, want, got):
    want_logits, got_logits = want["tensors"]["logits"], got["tensors"]["logits"]
    test.assertEqual(got_logits.shape, want_logits.shape)
    test.assertTrue(
        torch.equal(got_logits, want_logits),
        f"step {want['index']} ({want['mode']}): logits differ, max abs "
        f"{(got_logits.float() - want_logits.float()).abs().max().item()}",
    )


class TestModelRunnerStepBitwise(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        cls.prompts = [tokenizer.encode(p) for p in _PROMPTS]
        cls.follow_ups = [tokenizer.encode(p) for p in _FOLLOW_UPS]
        cls.json_prompt = tokenizer.encode("Give a city and its rank as JSON:")

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
        # One request at a time: decode batches inside the graph buckets.
        single = _generate(self.prompts[1], max_new_tokens=12)
        # A grammar: under overlap its sampling is deferred until the bitmask
        # of the previous step is ready, so the next replay runs first.
        grammar = _generate(
            self.json_prompt,
            max_new_tokens=64,
            sampling={"json_schema": _SCHEMA},
            return_logprob=True,
        )
        return first, second, logprobs, single, grammar

    def _check_config(self, name):
        args = _CONFIGS[name]
        runs = {
            run: _serve(run_args, save_logits=True, workload=self._workload)
            for run, run_args in (
                ("native", args),
                ("native_again", args),
                ("steps", [*args, "--model-runner", "sglang"]),
            )
        }
        (native_out, native), (again_out, again), (steps_out, steps) = runs.values()

        self.assertEqual({r["path"] for r in native + again}, {"native"})
        self.assertEqual(
            {(r["path"], r["runner"]) for r in steps}, {("step", "sglang")}
        )
        modes = {r["mode"] for r in steps}
        self.assertTrue({"extend", "decode"} <= modes, modes)
        self.assertTrue(any(any(r["prefix_lens"] or ()) for r in steps))
        self.assertTrue(any(r["prompt_logprobs"] for r in steps))
        json.loads(steps_out[4]["text"])
        # Step.graph reports exactly what the native forward replayed, and the
        # two paths make the same graph decision at every step.
        for r in steps:
            self.assertEqual(r["step_graph"] is not None, r["replayed_graph"], r)
        self.assertEqual(
            [(r["mode"], r["replayed_graph"]) for r in native],
            [(r["mode"], r["replayed_graph"]) for r in steps],
        )
        if "graphs" in name:
            replays = {(r["mode"], r["replayed_graph"]) for r in steps}
            # Replayed and eager-fallback steps in both phases.
            for mode in ("extend", "decode"):
                for replayed in (True, False):
                    self.assertIn((mode, replayed), replays)
        else:
            self.assertFalse(any(r["replayed_graph"] for r in steps))

        # Native against native first: the fixture repeats bitwise.
        repeat = self._assert_same_steps(native, again)
        self._assert_same_responses(native_out, again_out)
        through_steps = self._assert_same_steps(native, steps)
        self._assert_same_responses(native_out, steps_out)
        print(
            f"[{name}] bitwise logits: {repeat} steps native vs native, "
            f"{through_steps} steps native vs steps"
        )
        self.assertGreater(through_steps, 0)

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
            _assert_bitwise(self, want, got)
        return compared

    def test_eager_steps_are_bitwise_equal_to_the_native_lane(self):
        self._check_config("eager")

    def test_graph_steps_are_bitwise_equal_to_the_native_lane(self):
        self._check_config("graphs")

    def test_overlap_steps_are_bitwise_equal_to_the_native_lane(self):
        self._check_config("overlap")

    def test_overlap_graph_steps_are_bitwise_equal_to_the_native_lane(self):
        self._check_config("overlap_graphs")

    def test_mixed_prefill_and_decode_under_overlap_and_graphs(self):
        """A prefill arriving while a decode runs makes mixed steps, whose
        composition depends on timing: steps are matched by their inputs, and
        every matched step's logits must be bitwise equal."""
        args = _CONFIGS["overlap_graphs"]
        _, native = _serve(args, save_logits=True, workload=self._mixed_workload)
        _, steps = _serve(
            [*args, "--model-runner", "sglang"],
            save_logits=True,
            workload=self._mixed_workload,
        )
        self.assertTrue(any(r["mode"] == "mixed" for r in steps))
        self.assertTrue(all(r["path"] == "step" for r in steps if r["mode"] == "mixed"))
        by_inputs = _unique_by_inputs(native)
        matched = {"mixed": 0, "other": 0}
        for key, got in _unique_by_inputs(steps).items():
            want = by_inputs.get(key)
            if want is None:
                continue
            _assert_bitwise(self, want, got)
            matched["mixed" if got["mode"] == "mixed" else "other"] += 1
        print(f"[mixed] bitwise logits on matched steps: {matched}")
        self.assertGreater(matched["other"], 0)

    def _mixed_workload(self):
        long = threading.Thread(
            target=_generate,
            args=(self.prompts[0],),
            # Long enough to still decode when the prefills arrive, with graphs.
            kwargs={"max_new_tokens": 512, "sampling": {"ignore_eos": True}},
        )
        long.start()
        time.sleep(0.5)
        late = [_generate(p, max_new_tokens=4) for p in self.follow_ups]
        long.join()
        return late

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

    def test_untraced_steps_serve_as_the_native_lane(self):
        """Without the step trace the built-in runner does not materialize its
        Steps (nothing reads them); its responses must still be the native
        path's, under overlap and graphs, with every step through the runner."""
        args = _CONFIGS["overlap_graphs"]
        native = _serve_untraced(args, workload=self._workload)
        steps, log = _serve_untraced(
            [*args, "--model-runner", "sglang"], workload=self._workload, log=True
        )
        self.assertIn("Model runner 'sglang' runs steps through its API", log)
        self._assert_same_responses(native[:4], steps[:4])
        self.assertEqual(native[4]["text"], steps[4]["text"])

    def test_step_overhead_at_the_worker_boundary(self):
        """Host time per forward at TpModelWorker.forward_model, and the wall
        time of a decode-heavy workload, native vs steps, per configuration.

        Reported, not gated: the step path adds building the Step views, the
        graph decision, plan, the output conversion and, under overlap, the
        release bookkeeping around the same ModelRunner.forward. Responses are
        compared too: without saved logits nothing synchronizes the overlap
        schedule, so this is the run where consumers truly lag the next replay.
        """

        def decode_heavy():
            outputs = []
            tic = time.perf_counter()
            for batch in (self.prompts[:1], self.prompts[:2], self.prompts):
                outputs.append(
                    _generate(
                        batch,
                        max_new_tokens=64,
                        sampling={"ignore_eos": True},
                        return_logprob=True,
                    )
                )
            return outputs, time.perf_counter() - tic

        report = []
        for config in ("eager", "graphs", "overlap_graphs"):
            args = _CONFIGS[config]
            (native_out, native_s), native = _serve(
                args, save_logits=False, workload=decode_heavy
            )
            (steps_out, steps_s), steps = _serve(
                [*args, "--model-runner", "sglang"],
                save_logits=False,
                workload=decode_heavy,
            )
            self._assert_same_responses(native_out, steps_out)
            report.append(
                f"[{config}] workload wall time: native {native_s:.3f} s, "
                f"steps {steps_s:.3f} s"
            )
            report += _host_time_report(native, steps, batch_sizes=(1, 2, 4))
        print("STEP OVERHEAD (median host time at forward_model)")
        print("\n".join(report))
        self.assertTrue(report)


def _host_time_report(native, steps, *, batch_sizes):
    report = []
    for mode in ("extend", "decode"):
        for batch_size in batch_sizes:
            rows = {
                name: [
                    r
                    for r in trace[2:]  # skip warmup
                    if r["mode"] == mode and r["batch_size"] == batch_size
                ]
                for name, trace in (("native", native), ("steps", steps))
            }
            if not all(rows.values()):
                continue
            native_us = statistics.median(r["host_us"] for r in rows["native"])
            steps_us = statistics.median(r["host_us"] for r in rows["steps"])
            outside = statistics.median(
                r["host_us"] - r["execute_us"] for r in rows["steps"]
            )
            replayed = sum(r["replayed_graph"] for r in rows["steps"])
            report.append(
                f"  {mode} bs={batch_size}: native {native_us:.0f} us, "
                f"steps {steps_us:.0f} us (outside execute {outside:.0f} us), "
                f"n={len(rows['native'])}/{len(rows['steps'])}, "
                f"replayed {replayed}"
            )
    return report


if __name__ == "__main__":
    unittest.main(verbosity=3)
