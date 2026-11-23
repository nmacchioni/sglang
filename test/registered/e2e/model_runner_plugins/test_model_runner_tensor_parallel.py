"""Tensor parallelism through the model runner API (TP=2).

The built-in runner: raw logits before sampling are bitwise equal to the direct
path's on every rank, eager and with overlap scheduling and CUDA graphs, each
configuration first direct against direct to show it repeats at this topology;
responses are identical, and every rank's trace shows the step path and the
same graph decision as the direct path.

The reference runner: column- and row-parallel linears over the TP group SGLang
hands to ``load``. Its full logits (gathered inside the runner) and its
vocabulary-sharded logits (gathered by SGLang) are bitwise equal, and it agrees
with the direct path within the calibrated bounds of the TP1 reference test.
"""

import os
import tempfile
import unittest
from collections import defaultdict

import requests
import torch
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.model_runner_reference import (
    REFERENCE_RUNNER,
    SHARDED_REFERENCE_RUNNER,
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

register_cuda_ci(est_time=1500, stage="extra-a", runner_config="2-gpu-large")

_MODEL = "Qwen/Qwen3-0.6B"
_TP = ["--tp-size", "2", "--disable-flashinfer-autotune", "--enable-mixed-chunk"]
_EAGER = [
    *_TP,
    "--disable-overlap-schedule",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
]
# Decode graphs of one and two requests: four requests decode eagerly.
_OVERLAP_GRAPHS = [
    *_TP,
    "--cuda-graph-max-bs-decode",
    "2",
    "--cuda-graph-max-bs-prefill",
    "256",
]
_REFERENCE_EAGER = [*_EAGER, "--mem-fraction-static", "0.5"]
_REFERENCE_GRAPHS = [
    *_TP,
    "--disable-prefill-cuda-graph",
    "--cuda-graph-max-bs-decode",
    "2",
    "--mem-fraction-static",
    "0.5",
]
_PREAMBLE = "Answer in one short sentence, plainly and without lists. " * 12
_PROMPTS = [
    _PREAMBLE + "The capital of France is",
    "Write a haiku about the sea:",
    _PREAMBLE + "1, 2, 3, 4,",
    "In 1969, the first person to walk on the moon was",
]
_FOLLOW_UPS = [
    _PREAMBLE + "The capital of France is a city that",
    _PREAMBLE + "1, 2, 3, 4, 5, 6,",
]
_SIGNATURE = ("input_ids", "positions", "out_cache_loc", "seq_lens", "req_pool_indices")
# The TP1 reference test's bounds, max widened for TP2 by the direct path's own
# TP1-to-TP2 shift. Measured on GB300 over the same sequences: direct TP1 vs
# TP2 max 0.21; vs HF fp32, direct TP1 0.18, direct TP2 0.36, reference TP2
# 0.43; reference vs direct, TP1 0.24, TP2 0.27 (mean 0.015). Each max is at
# a sequence's first or second scored token (logprob near -12).
_TIE_MARGIN = 0.5
_MAX_LOGPROB_DIFF = 0.35
_MEAN_LOGPROB_DIFF = 0.03


def _generate(input_ids, *, max_new_tokens=8, **extra):
    response = requests.post(
        DEFAULT_URL_FOR_TEST + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
            **extra,
        },
    )
    response.raise_for_status()
    return response.json()


def _serve(other_args, *, workload, env=None, save_logits=True):
    trace_dir = tempfile.mkdtemp()
    process = popen_launch_server(
        _MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
        env={
            **(env or os.environ),
            "SGLANG_MODEL_RUNNER_STEP_TRACE_DIR": trace_dir,
            "SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS": "1" if save_logits else "0",
        },
    )
    try:
        result = workload()
    finally:
        terminate_and_kill_process_tree(process)
    return result, by_tp_rank(read_step_trace(trace_dir))


def by_tp_rank(trace) -> dict[int, list[dict]]:
    ranks = defaultdict(list)
    for record in trace:
        ranks[record["tp_rank"]].append(record)
    return {rank: sorted(r, key=lambda r: r["index"]) for rank, r in ranks.items()}


def _assert_bitwise(test, want, got, what):
    want_logits, got_logits = want["tensors"]["logits"], got["tensors"]["logits"]
    test.assertEqual(got_logits.shape, want_logits.shape, what)
    test.assertTrue(
        torch.equal(got_logits, want_logits),
        f"{what}: step {want['index']} ({want['mode']}) logits differ, max abs "
        f"{(got_logits.float() - want_logits.float()).abs().max().item()}",
    )


class _Prompts:
    @classmethod
    def tokenize(cls):
        tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        cls.prompts = [tokenizer.encode(p) for p in _PROMPTS]
        cls.follow_ups = [tokenizer.encode(p) for p in _FOLLOW_UPS]

    @classmethod
    def workload(cls):
        return [
            _generate(cls.prompts),
            _generate(cls.follow_ups),
            _generate(
                cls.follow_ups[0],
                return_logprob=True,
                logprob_start_len=0,
                top_logprobs_num=3,
            ),
            _generate(cls.prompts[1], max_new_tokens=12),
        ]


class TestNativeTensorParallel(_Prompts, CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenize()

    def _check(self, args):
        (direct_out, direct), (again_out, again), (steps_out, steps) = (
            _serve(run_args, workload=self.workload)
            for run_args in (args, args, [*args, "--model-runner", "sglang"])
        )
        self.assertEqual(set(direct), {0, 1})
        self.assertEqual(set(steps), {0, 1})
        for rank in (0, 1):
            self.assertEqual({r["path"] for r in direct[rank]}, {"direct"})
            self.assertEqual(
                {(r["path"], r["runner"]) for r in steps[rank]}, {("step", "sglang")}
            )
            for r in steps[rank]:
                self.assertEqual(r["step_graph"] is not None, r["replayed_graph"], r)
            # The TP members took the same decisions as each other and as the
            # direct path: same modes, same graph replays, step by step.
            decisions = [(r["mode"], r["replayed_graph"]) for r in steps[rank]]
            self.assertEqual(
                decisions, [(r["mode"], r["replayed_graph"]) for r in direct[rank]]
            )
            self.assertEqual(
                [r["step_graph"] for r in steps[rank]],
                [r["step_graph"] for r in steps[0]],
            )
        repeat = self._same_steps(direct, again)
        through_steps = self._same_steps(direct, steps)
        self._assert_same_outputs(direct_out, again_out)
        self._assert_same_outputs(direct_out, steps_out)
        print(
            f"[tp2 {args[-2:]}] bitwise logits per rank: direct vs direct "
            f"{repeat}, direct vs steps {through_steps}"
        )
        return steps

    def _same_steps(self, expected, actual):
        compared = {}
        for rank in expected:
            self.assertEqual(len(actual[rank]), len(expected[rank]))
            compared[rank] = 0
            for want, got in zip(expected[rank], actual[rank]):
                self.assertEqual(got["mode"], want["mode"])
                if "tensors" not in want:
                    continue
                for name in _SIGNATURE:
                    torch.testing.assert_close(
                        got["tensors"][name], want["tensors"][name], rtol=0, atol=0
                    )
                _assert_bitwise(self, want, got, f"rank {rank}")
                compared[rank] += 1
            self.assertGreater(compared[rank], 0)
        return compared

    def _assert_same_outputs(self, want, got):
        for w, g in zip(want, got):
            for a, b in zip(
                w if isinstance(w, list) else [w], g if isinstance(g, list) else [g]
            ):
                self.assertEqual(a["output_ids"], b["output_ids"])
                for field in (
                    "input_token_logprobs",
                    "output_token_logprobs",
                    "input_top_logprobs",
                    "cached_tokens",
                ):
                    self.assertEqual(
                        a["meta_info"].get(field), b["meta_info"].get(field), field
                    )

    def test_eager_steps_are_bitwise_equal_on_every_rank(self):
        self._check(_EAGER)

    def test_overlap_graph_steps_are_bitwise_equal_on_every_rank(self):
        steps = self._check(_OVERLAP_GRAPHS)
        replays = {(r["mode"], r["replayed_graph"]) for r in steps[1]}
        for mode in ("extend", "decode"):
            for replayed in (True, False):
                self.assertIn((mode, replayed), replays)


class TestReferenceTensorParallel(_Prompts, CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenize()
        env = {**os.environ, **install_reference_runner(tempfile.mkdtemp())}
        cls.forced_outputs = None
        cls.runs = {"direct": _serve(_REFERENCE_EAGER, workload=cls.scored)}
        # Every runner scores the direct path's continuations.
        cls.forced_outputs = [o["output_ids"] for o in cls.runs["direct"][0]["greedy"]]
        for name, args, runner in (
            ("full", _REFERENCE_EAGER, REFERENCE_RUNNER),
            ("sharded", _REFERENCE_EAGER, SHARDED_REFERENCE_RUNNER),
            ("sharded_graphs", _REFERENCE_GRAPHS, SHARDED_REFERENCE_RUNNER),
        ):
            cls.runs[name] = _serve(
                [*args, "--model-runner", runner], workload=cls.scored, env=env
            )

    @classmethod
    def scored(cls):
        """Greedy outputs, then every prompt with the direct path's continuation
        scored (prompt logprobs through project())."""
        greedy = [_generate(p, max_new_tokens=16) for p in cls.prompts]
        pair = _generate(cls.prompts[:2], max_new_tokens=16)
        continuations = cls.forced_outputs or [o["output_ids"] for o in greedy]
        forced = []
        for prompt, continuation in zip(cls.prompts, continuations):
            meta = _generate(
                prompt + continuation,
                max_new_tokens=1,
                return_logprob=True,
                logprob_start_len=0,
                top_logprobs_num=2,
            )["meta_info"]
            forced.append((meta["input_token_logprobs"], meta["input_top_logprobs"]))
        return {"greedy": greedy, "pair": pair, "forced": forced}

    def test_every_rank_ran_the_reference_steps(self):
        for name in ("full", "sharded", "sharded_graphs"):
            _, trace = self.runs[name]
            self.assertEqual(set(trace), {0, 1}, name)
            runner = REFERENCE_RUNNER if name == "full" else SHARDED_REFERENCE_RUNNER
            for rank in (0, 1):
                self.assertEqual(
                    {(r["path"], r["runner"]) for r in trace[rank]},
                    {("step", runner)},
                )
                self.assertTrue(any(r["prompt_logprobs"] for r in trace[rank]))
        _, graphs = self.runs["sharded_graphs"]
        for rank in (0, 1):
            self.assertTrue(any(r["replayed_graph"] for r in graphs[rank]))
            self.assertEqual(
                [r["step_graph"] for r in graphs[rank]],
                [r["step_graph"] for r in graphs[0]],
            )

    def test_sglang_gathered_shards_equal_runner_gathered_logits(self):
        """Gathers are exact, so where the runner returns full logits or its
        vocabulary shard for SGLang to gather, the logits are bitwise equal; and
        a decode graph whose bucket equals the batch matches eager bitwise."""
        _, full = self.runs["full"]
        _, sharded = self.runs["sharded"]
        _, graphs = self.runs["sharded_graphs"]
        counts = defaultdict(int)
        for rank in (0, 1):
            self.assertEqual(len(sharded[rank]), len(full[rank]))
            for want, got in zip(full[rank], sharded[rank]):
                if "tensors" in want:
                    _assert_bitwise(self, want, got, f"rank {rank} sharded")
                    counts["sharded"] += 1
            eager = _unique_by_inputs(sharded[rank])
            for key, got in _unique_by_inputs(graphs[rank]).items():
                want = eager.get(key)
                if want is not None and got["replayed_graph"]:
                    if got["batch_size"] == got["step_graph"][1]:
                        _assert_bitwise(self, want, got, f"rank {rank} graph")
                        counts["graph"] += 1
        print(f"reference tp2 bitwise logits: {dict(counts)}")
        self.assertGreater(counts["sharded"], 0)
        self.assertGreater(counts["graph"], 0)
        self.assertEqual(
            [o["output_ids"] for o in self.runs["sharded"][0]["greedy"]],
            [o["output_ids"] for o in self.runs["full"][0]["greedy"]],
        )
        self.assertEqual(
            [o["output_ids"] for o in self.runs["sharded_graphs"][0]["greedy"]],
            [o["output_ids"] for o in self.runs["sharded"][0]["greedy"]],
        )

    def test_agrees_with_the_direct_path_within_calibrated_bounds(self):
        direct = self.runs["direct"][0]
        for name in ("full", "sharded"):
            got = self.runs[name][0]
            exact = 0
            for ids, want, out, (_, top) in zip(
                self.prompts, direct["greedy"], got["greedy"], direct["forced"]
            ):
                if want["output_ids"] == out["output_ids"]:
                    exact += 1
                    continue
                first = next(
                    i
                    for i, (w, g) in enumerate(
                        zip(want["output_ids"], out["output_ids"])
                    )
                    if w != g
                )
                candidates = top[len(ids) + first]
                self.assertLess(candidates[0][0] - candidates[1][0], _TIE_MARGIN)
            self.assertGreaterEqual(exact, len(self.prompts) - 1, name)
            diffs = []
            for (want, _), (logprobs, _) in zip(direct["forced"], got["forced"]):
                self.assertEqual([t for _, t, _ in logprobs], [t for _, t, _ in want])
                diffs += [abs(w[0] - g[0]) for w, g in zip(want[1:], logprobs[1:])]
            mean = sum(diffs) / len(diffs)
            print(
                f"[{name}] forced logprob diff: max {max(diffs):.4f}, mean {mean:.5f}"
            )
            self.assertLess(max(diffs), _MAX_LOGPROB_DIFF)
            self.assertLess(mean, _MEAN_LOGPROB_DIFF)


def _unique_by_inputs(trace) -> dict:
    by_inputs, repeated = {}, set()
    for record in trace:
        if "tensors" not in record:
            continue
        key = tuple(
            record["tensors"][name].numpy().tobytes() for name in _SIGNATURE
        ) + (record["mode"],)
        if key in by_inputs:
            repeated.add(key)
        by_inputs[key] = record
    return {key: r for key, r in by_inputs.items() if key not in repeated}


if __name__ == "__main__":
    unittest.main(verbosity=3)
