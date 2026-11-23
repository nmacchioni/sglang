"""Pipeline parallelism and DP attention through the model runner API.

PP=2: every stage runs its steps through the built-in runner; the first stage
returns pp_out and no logits, the last stage logits. DP attention (TP=2, two
DP ranks): each rank runs its own batch or an idle step and all join the MLP
collectives; the graph decision is the one SGLang's MLP-sync gather voted, so
one rank's oversized batch sends every rank eager on that step.

Both are compared with the direct path step by step, matched by their inputs
(batch composition under DP depends on request arrival), each configuration
first direct against direct: matched steps' logits and stage outputs must be
bitwise equal, and responses of the deterministic phases identical.
"""

import os
import tempfile
import threading
import time
import unittest
from collections import defaultdict

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

register_cuda_ci(est_time=2400, stage="extra-a", runner_config="2-gpu-large")

_MODEL = "Qwen/Qwen3-0.6B"
_COMMON = ["--disable-flashinfer-autotune", "--mem-fraction-static", "0.6"]
_NO_GRAPHS = ["--disable-decode-cuda-graph", "--disable-prefill-cuda-graph"]
# Decode graphs of one and two requests: a rank decoding four runs eager.
_SMALL_GRAPHS = ["--disable-prefill-cuda-graph", "--cuda-graph-max-bs-decode", "2"]
_PP = [*_COMMON, "--pp-size", "2"]
_DP = [*_COMMON, "--tp-size", "2", "--attn-dp-size", "2"]
_PREAMBLE = "Answer in one short sentence, plainly and without lists. " * 12
_PROMPTS = [
    _PREAMBLE + "The capital of France is",
    "Write a haiku about the sea:",
    _PREAMBLE + "1, 2, 3, 4,",
    "In 1969, the first person to walk on the moon was",
]
_SIGNATURE = ("input_ids", "positions", "out_cache_loc", "seq_lens", "req_pool_indices")


def _generate(input_ids, *, max_new_tokens=8, dp_rank=None, **extra):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
        **extra,
    }
    if dp_rank is not None:
        payload["routed_dp_rank"] = dp_rank
    response = requests.post(DEFAULT_URL_FOR_TEST + "/generate", json=payload)
    response.raise_for_status()
    return response.json()


def _serve(other_args, *, workload):
    trace_dir = tempfile.mkdtemp()
    process = popen_launch_server(
        _MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
        env={
            **os.environ,
            "SGLANG_MODEL_RUNNER_STEP_TRACE_DIR": trace_dir,
            "SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS": "1",
        },
    )
    try:
        result = workload()
    finally:
        terminate_and_kill_process_tree(process)
    return result, read_step_trace(trace_dir)


def _by(trace, key) -> dict[int, list[dict]]:
    ranks = defaultdict(list)
    for record in trace:
        ranks[record[key]].append(record)
    return {rank: sorted(r, key=lambda r: r["index"]) for rank, r in ranks.items()}


def _unique_by_inputs(trace) -> dict:
    """Steps keyed by their inputs, without keys that repeat in the run."""
    by_inputs, repeated = {}, set()
    for record in trace:
        if "tensors" not in record:
            continue
        key = tuple(
            record["tensors"][name].numpy().tobytes() for name in _SIGNATURE
        ) + (record["mode"], str(record["global_num_tokens"]))
        if key in by_inputs:
            repeated.add(key)
        by_inputs[key] = record
    return {key: r for key, r in by_inputs.items() if key not in repeated}


def _max_abs_diff(a, b):
    if a.shape != b.shape or a.numel() == 0:
        return None
    return (a.float() - b.float()).abs().max().item()


class _Compare:
    def _matched_bitwise(self, expected, actual) -> int:
        """Outputs of the steps both runs took with the same inputs."""
        expected = _unique_by_inputs(expected)
        matched = 0
        for key, got in _unique_by_inputs(actual).items():
            want = expected.get(key)
            if want is None:
                continue
            outputs = sorted(n for n in want["tensors"] if n not in _SIGNATURE)
            self.assertEqual(
                outputs, sorted(n for n in got["tensors"] if n not in _SIGNATURE)
            )
            for name in outputs:
                a, b = want["tensors"][name], got["tensors"][name]
                if not torch.equal(a, b):
                    self.fail(
                        f"{got['mode']} step {got['index']}: {name} differs, "
                        f"shapes {tuple(a.shape)} {tuple(b.shape)}, max abs "
                        f"{_max_abs_diff(a, b)}"
                    )
            matched += 1
        return matched

    def _check_paths(self, direct, steps, key, *, direct_fallback=False):
        for rank, records in _by(direct, key).items():
            self.assertEqual({r["path"] for r in records}, {"direct"}, rank)
        by_rank = _by(steps, key)
        self.assertEqual(set(by_rank), {0, 1})
        for rank, records in by_rank.items():
            if direct_fallback:
                records = [r for r in records if r["path"] != "direct"]
            self.assertEqual(
                {(r["path"], r["runner"]) for r in records}, {("step", "sglang")}
            )
            for r in records:
                self.assertEqual(r["step_graph"] is not None, r["replayed_graph"], r)
        return by_rank

    def _three_runs(self, args, workload):
        return [
            _serve(run_args, workload=workload)
            for run_args in (args, args, [*args, "--model-runner", "sglang"])
        ]

    def _compare(self, name, key, runs):
        (direct_out, direct), (again_out, again), (steps_out, steps) = runs
        counts = {}
        for rank in (0, 1):
            d = [r for r in direct if r[key] == rank]
            counts[rank] = (
                self._matched_bitwise(d, [r for r in again if r[key] == rank]),
                self._matched_bitwise(d, [r for r in steps if r[key] == rank]),
            )
            self.assertGreater(counts[rank][1], 0, rank)
        print(f"[{name}] matched bitwise steps per rank (repeat, steps): {counts}")
        self.assertEqual(_ids(again_out), _ids(direct_out))
        self.assertEqual(_ids(steps_out), _ids(direct_out))


def _ids(outputs):
    return [
        [o["output_ids"] for o in (out if isinstance(out, list) else [out])]
        for out in outputs
    ]


class TestPipelineParallel(_Compare, CustomTestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        cls.prompts = [tokenizer.encode(p) for p in _PROMPTS]

    def workload(self):
        return [
            _generate(self.prompts),
            _generate(self.prompts[1], max_new_tokens=12),
            _generate(
                self.prompts[0],
                return_logprob=True,
                logprob_start_len=0,
                top_logprobs_num=2,
            ),
        ]

    def _check(self, name, args):
        runs = self._three_runs(args, self.workload)
        by_stage = self._check_paths(runs[0][1], runs[2][1], "pp_rank")
        first, last = by_stage[0], by_stage[1]
        # The first stage hands proxy tensors on; only the last has logits.
        self.assertTrue(all(r["pp_out"] for r in first if r["mode"] != "idle"))
        self.assertFalse(any(r["pp_out"] for r in last))
        self.assertTrue(any("logits" in r.get("tensors", {}) for r in last))
        self.assertTrue(any(r["prompt_logprobs"] for r in last))
        self._compare(name, "pp_rank", runs)
        return by_stage

    def test_pipeline_stages_eager(self):
        self._check("pp2 eager", [*_PP, *_NO_GRAPHS])

    def test_pipeline_stages_with_graphs(self):
        by_stage = self._check("pp2 graphs", [*_PP, *_SMALL_GRAPHS])
        for stage in (0, 1):
            self.assertTrue(any(r["replayed_graph"] for r in by_stage[stage]))


class TestDPAttention(_Compare, CustomTestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        cls.prompts = [tokenizer.encode(p) for p in _PROMPTS]

    def workload(self):
        # Rank 0 alone: rank 1 runs idle steps and joins the MLP collectives.
        alone = _generate(self.prompts[:2], dp_rank=0)
        # Four requests decode on rank 0 while one decodes on rank 1: rank 1's
        # batch fits a graph, rank 0's does not, so both run eager.
        both = {}

        def run(key, prompts, dp_rank):
            both[key] = _generate(prompts, max_new_tokens=24, dp_rank=dp_rank)

        threads = [
            threading.Thread(target=run, args=("wide", self.prompts, 0)),
            threading.Thread(target=run, args=("narrow", self.prompts[1], 1)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # A prefill on rank 1 while rank 0 decodes: mixed local modes.
        long = threading.Thread(
            target=_generate,
            args=(self.prompts[2],),
            kwargs={"max_new_tokens": 64, "dp_rank": 0},
        )
        long.start()
        time.sleep(0.5)
        late = _generate(self.prompts[3], dp_rank=1)
        long.join()
        return [alone, late]

    def _check(self, name, args, *, direct_fallback=False):
        runs = self._three_runs(args, self.workload)
        by_rank = self._check_paths(
            runs[0][1], runs[2][1], "attn_dp_rank", direct_fallback=direct_fallback
        )
        ranks = (by_rank[0], by_rank[1])
        # Lockstep: every step runs on both DP ranks, with one graph decision
        # and, where a step falls back to the direct path, on every rank.
        self.assertEqual(len(ranks[0]), len(ranks[1]))
        for a, b in zip(*ranks):
            self.assertEqual(a["global_num_tokens"], b["global_num_tokens"])
            self.assertEqual(a["replayed_graph"], b["replayed_graph"], (a, b))
            self.assertEqual(a["path"], b["path"], (a, b))
        modes = [(a["mode"], b["mode"]) for a, b in zip(*ranks)]
        self.assertIn(("decode", "idle"), modes)
        if not direct_fallback:
            # With prefill graphs, the direct path pads a decode rank of an
            # extend step into a dummy extend, and the trace records that.
            self.assertTrue(
                any(a != b and "idle" not in (a, b) for a, b in modes), modes
            )
        self._compare(name, "attn_dp_rank", runs)
        return ranks

    def test_dp_attention_eager(self):
        self._check("dp2 eager", [*_DP, *_NO_GRAPHS])

    def test_dp_attention_coordinated_graph_fallback(self):
        rank0, rank1 = self._check("dp2 graphs", [*_DP, *_SMALL_GRAPHS])
        steps = list(zip(rank0, rank1))
        # Rank 1 decodes one request: replayed when rank 0's batch fits too,
        # eager when rank 0 decodes four.
        self.assertTrue(
            any(
                b["mode"] == "decode" and b["batch_size"] == 1 and b["replayed_graph"]
                for _, b in steps
            )
        )
        self.assertTrue(
            any(
                a["batch_size"] > 2
                and b["mode"] == "decode"
                and b["batch_size"] == 1
                and not b["replayed_graph"]
                for a, b in steps
            )
        )

    def test_dp_attention_prefill_graph_steps_keep_the_direct_path(self):
        """With prefill graphs on, the forward pads every rank of an extend
        step to one prefill graph bucket after the Step is built: those steps
        take the direct path on every rank, the others stay on the API."""
        rank0, rank1 = self._check(
            "dp2 prefill graphs",
            [*_DP, "--cuda-graph-max-bs-decode", "2"],
            direct_fallback=True,
        )
        paths = {(r["path"], r["replayed_graph"]) for r in rank0 + rank1}
        self.assertIn(("direct", True), paths)
        self.assertIn(("step", True), paths)


if __name__ == "__main__":
    unittest.main(verbosity=3)
