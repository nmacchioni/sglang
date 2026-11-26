"""The reference plugin, an independent pure-PyTorch runner that owns its weights
and KV, serves Qwen3-0.6B through the Step API with prefix reuse, mixed steps
and prompt logprobs through project(), and agrees with SGLang's own runner.

The two runners share no kernel, so exact greedy agreement is not guaranteed:
a greedy mismatch must sit at a near tie of the native lane, and on the same
forced tokens the reference's logprobs must stay within a calibrated bound.

Under overlap scheduling with its decode graphs, the reference is compared with
itself eager: a decode graph whose bucket equals the batch runs the same
kernels at the same shapes, so its logits are bitwise equal.
"""

import json
import os
import tempfile
import threading
import time
import unittest

import requests
import torch
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.model_runner_reference import (
    install_reference_runner,
    native_lane,
    read_step_trace,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=720, stage="base-b", runner_config="1-gpu-small")

_MODEL = "Qwen/Qwen3-0.6B"
_ARGS = [
    "--disable-overlap-schedule",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
    # Startup autotuning takes minutes on GB300 and changes nothing compared here.
    "--disable-flashinfer-autotune",
    "--enable-mixed-chunk",
    "--mem-fraction-static",
    "0.5",
]
_MAX_NEW_TOKENS = 16
# Longer than two KV pages (64 tokens on Blackwell backends), so the second and
# fourth prompts reuse whole cached pages of the first and third.
_PREAMBLE = "Answer in one short sentence, plainly and without lists. " * 12
_PROMPTS = [
    _PREAMBLE + "The capital of France is",
    _PREAMBLE + "The capital of France is a city that",
    _PREAMBLE + "Write a short poem about the ocean.",
    _PREAMBLE + "Write a short poem about the ocean. Make it rhyme.",
    "def fibonacci(n):",
]
# Greedy may part ways only where the native lane's top two logprobs are this close.
_TIE_MARGIN = 0.5
# Calibrated on GB300: bf16 activations through different kernels.
_MAX_LOGPROB_DIFF = 0.25
_MEAN_LOGPROB_DIFF = 0.03


def _generate(payload):
    response = requests.post(DEFAULT_URL_FOR_TEST + "/generate", json=payload)
    response.raise_for_status()
    return response.json()


def _greedy(input_ids):
    # No logprobs here: prompt logprobs from position 0 would bypass the cache.
    return _generate(
        {
            "input_ids": input_ids,
            "sampling_params": {"max_new_tokens": _MAX_NEW_TOKENS, "temperature": 0},
        }
    )


def _forced_logprobs(input_ids):
    """Prompt logprobs and the top two candidates at every position."""
    out = _generate(
        {
            "input_ids": input_ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True,
            "logprob_start_len": 0,
            "top_logprobs_num": 2,
        }
    )
    return out["meta_info"]["input_token_logprobs"], out["meta_info"][
        "input_top_logprobs"
    ]


def _mixed_workload():
    """A long decode with prefills arriving while it runs."""
    long = threading.Thread(
        target=_generate,
        args=(
            {
                "text": "Count from one to one hundred:",
                "sampling_params": {
                    "max_new_tokens": 48,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
        ),
    )
    long.start()
    time.sleep(0.5)
    late = _generate(
        {
            "text": "The tallest mountain on Earth is",
            "sampling_params": {"max_new_tokens": 4, "temperature": 0},
        }
    )
    long.join()
    return late


def _serve(other_args, env, *, save_logits=False):
    trace_dir = tempfile.mkdtemp()
    process = popen_launch_server(
        _MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
        env={
            **env,
            "SGLANG_MODEL_RUNNER_STEP_TRACE_DIR": trace_dir,
            "SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS": "1" if save_logits else "0",
            **native_lane(other_args),
        },
    )
    return process, trace_dir


class TestReferenceRunner(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        cls.prompts = [tokenizer.encode(p) for p in _PROMPTS]
        cls.native = cls._run(_ARGS, dict(os.environ), forced_outputs=None)
        env = {**os.environ, **install_reference_runner(tempfile.mkdtemp())}
        # Both runners score the native lane's tokens.
        cls.reference = cls._run(
            [*_ARGS, "--model-runner", "reference"],
            env,
            forced_outputs=[g["output_ids"] for g in cls.native["greedy"]],
        )

    @classmethod
    def _run(cls, args, env, *, forced_outputs):
        process, trace_dir = _serve(args, env)
        try:
            greedy = [_greedy(p) for p in cls.prompts]
            if forced_outputs is None:
                forced_outputs = [g["output_ids"] for g in greedy]
            forced = [
                _forced_logprobs(p + out) for p, out in zip(cls.prompts, forced_outputs)
            ]
            requests.post(DEFAULT_URL_FOR_TEST + "/flush_cache").raise_for_status()
            cold = [_greedy(p) for p in cls.prompts]
            late = _mixed_workload()
        finally:
            terminate_and_kill_process_tree(process)
        return {
            "greedy": greedy,
            "cold": cold,
            "forced": forced,
            "late": late,
            "trace": read_step_trace(trace_dir),
        }

    def test_steps_ran_through_the_reference_runner(self):
        trace = self.reference["trace"]
        self.assertEqual(
            {(r["path"], r["runner"]) for r in trace}, {("step", "reference")}
        )
        modes = {r["mode"] for r in trace}
        self.assertTrue({"extend", "decode", "mixed"} <= modes, modes)
        # Prompt logprobs went through project().
        self.assertTrue(any(r["prompt_logprobs"] for r in trace))
        self.assertEqual({r["path"] for r in self.native["trace"]}, {"native"})

    def test_prefix_reuse(self):
        """Follow-up prompts hit the radix cache, and the reference reads the
        cached KV through SGLang's page table: a prefix hit answers as the same
        prompt does with an empty cache."""
        cached = [g["meta_info"]["cached_tokens"] for g in self.reference["greedy"]]
        self.assertGreater(cached[1], 0, cached)
        self.assertGreater(cached[3], 0, cached)
        self.assertEqual(
            [g["meta_info"]["cached_tokens"] for g in self.reference["cold"]][0], 0
        )
        self.assertTrue(
            any(any(r["prefix_lens"] or ()) for r in self.reference["trace"])
        )
        self.assertEqual(
            [g["output_ids"] for g in self.reference["greedy"]],
            [g["output_ids"] for g in self.reference["cold"]],
        )

    def test_greedy_outputs_match_up_to_near_ties(self):
        exact = 0
        for prompt, ids, want, got, (_, native_top) in zip(
            _PROMPTS,
            self.prompts,
            self.native["greedy"],
            self.reference["greedy"],
            self.native["forced"],
        ):
            want_ids, got_ids = want["output_ids"], got["output_ids"]
            if want_ids == got_ids:
                exact += 1
                continue
            first = next(i for i, (w, g) in enumerate(zip(want_ids, got_ids)) if w != g)
            # The native lane's candidates for output token `first`.
            top = native_top[len(ids) + first]
            margin = top[0][0] - top[1][0]
            print(f"{prompt!r}: diverges at token {first}, native margin {margin:.3f}")
            self.assertLess(margin, _TIE_MARGIN, prompt)
        print(f"greedy: {exact}/{len(_PROMPTS)} prompts match exactly")
        self.assertGreaterEqual(exact, len(_PROMPTS) - 1)
        self.assertEqual(
            self.reference["late"]["output_ids"], self.native["late"]["output_ids"]
        )

    def test_forced_token_logprobs_agree(self):
        diffs = []
        for (want, _), (got, got_top) in zip(
            self.native["forced"], self.reference["forced"]
        ):
            self.assertEqual([t for _, t, _ in got], [t for _, t, _ in want])
            diffs += [abs(w[0] - g[0]) for w, g in zip(want[1:], got[1:])]
            self.assertTrue(all(top for top in got_top[1:]))
        mean = sum(diffs) / len(diffs)
        print(f"forced-token logprob diff: max {max(diffs):.4f}, mean {mean:.5f}")
        self.assertLess(max(diffs), _MAX_LOGPROB_DIFF)
        self.assertLess(mean, _MEAN_LOGPROB_DIFF)


# Overlap on (the default) and decode graphs of one and two requests.
_OVERLAP_GRAPH_ARGS = [
    "--disable-prefill-cuda-graph",
    "--cuda-graph-max-bs-decode",
    "2",
    "--disable-flashinfer-autotune",
    "--enable-mixed-chunk",
    "--mem-fraction-static",
    "0.5",
]
_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
)
_SIGNATURE = ("input_ids", "positions", "out_cache_loc", "seq_lens", "req_pool_indices")


class TestReferenceRunnerOverlapGraphs(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        cls.prompts = [tokenizer.encode(p) for p in _PROMPTS]
        cls.env = {**os.environ, **install_reference_runner(tempfile.mkdtemp())}
        runner = ["--model-runner", "reference"]
        cls.runs = {
            name: cls._run([*args, *runner])
            for name, args in (
                ("eager", _ARGS),
                ("eager_again", _ARGS),
                ("overlap_graphs", _OVERLAP_GRAPH_ARGS),
            )
        }

    @classmethod
    def _run(cls, args):
        process, trace_dir = _serve(args, cls.env, save_logits=True)
        try:
            singles = [_greedy(p) for p in cls.prompts[:3]]
            pair = _greedy(cls.prompts[3:5])
            batch = _greedy(cls.prompts)
            grammar = _generate(
                {
                    "text": "Describe a person as JSON:",
                    "sampling_params": {
                        "max_new_tokens": 48,
                        "temperature": 0,
                        "json_schema": _SCHEMA,
                    },
                }
            )
        finally:
            terminate_and_kill_process_tree(process)
        return {
            "outputs": [*singles, *pair, *batch],
            "grammar": grammar,
            "trace": read_step_trace(trace_dir),
        }

    def test_steps_replay_decode_graphs_and_fall_back_to_eager(self):
        trace = self.runs["overlap_graphs"]["trace"]
        self.assertEqual(
            {(r["path"], r["runner"]) for r in trace}, {("step", "reference")}
        )
        for r in trace:
            self.assertEqual(r["step_graph"] is not None, r["replayed_graph"], r)
        replays = {(r["mode"], r["batch_size"], r["replayed_graph"]) for r in trace}
        self.assertIn(("decode", 1, True), replays)
        self.assertIn(("decode", 2, True), replays)
        # Five requests exceed the largest bucket.
        self.assertIn(("decode", len(_PROMPTS), False), replays)
        self.assertFalse(
            any(r["replayed_graph"] for r in trace if r["mode"] != "decode")
        )

    def test_outputs_match_the_eager_reference(self):
        eager, again, graphs = (
            self.runs[name] for name in ("eager", "eager_again", "overlap_graphs")
        )
        ids = [o["output_ids"] for o in eager["outputs"]]
        self.assertEqual([o["output_ids"] for o in again["outputs"]], ids)
        self.assertEqual([o["output_ids"] for o in graphs["outputs"]], ids)
        # Grammar sampling is deferred under overlap; the output stays valid.
        self.assertEqual(
            graphs["grammar"]["output_ids"], eager["grammar"]["output_ids"]
        )
        json.loads(graphs["grammar"]["text"])

    def test_graph_logits_are_bitwise_equal_to_eager_steps(self):
        """Steps are matched by their inputs, first eager against eager."""
        eager = self.runs["eager"]["trace"]
        repeat = _matched_bitwise(self, eager, self.runs["eager_again"]["trace"])
        counts = _matched_bitwise(self, eager, self.runs["overlap_graphs"]["trace"])
        print(f"reference bitwise logits: eager vs eager {repeat}, vs graphs {counts}")
        self.assertGreater(counts["replayed"], 0)


def _matched_bitwise(test, expected, actual) -> dict:
    expected, actual = _unique_by_inputs(expected), _unique_by_inputs(actual)
    counts = {"replayed": 0, "eager": 0}
    for key, got in actual.items():
        want = expected.get(key)
        if want is None:
            continue
        want_logits, got_logits = want["tensors"]["logits"], got["tensors"]["logits"]
        test.assertTrue(
            torch.equal(want_logits, got_logits),
            f"{got['mode']} bs={got['batch_size']} (replayed "
            f"{got['replayed_graph']}): max abs "
            f"{(want_logits - got_logits).abs().max().item()}",
        )
        counts["replayed" if got["replayed_graph"] else "eager"] += 1
    return counts


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


if __name__ == "__main__":
    unittest.main(verbosity=3)
