"""Context parallelism, two-batch overlap and expert parallelism through the
model runner API, for the built-in runner (2 GPUs).

- Context parallelism: Qwen3-0.6B at TP=2 with attention CP=2 (zigzag prefill
  CP), eager and with graphs (CP prefill graphs, decode graphs). The trace
  shows long extend steps CP-sharded on the step path.
- Expert parallelism: Qwen3-30B-A3B at TP=2 with EP=2, with graphs; and with
  EPLB rebalancing every few forwards, where the trace shows the expert
  placement change mid-run on the step path, at the same forward as direct.
- Two-batch overlap: Qwen3-30B-A3B-FP8 at TP=2 with EP=2 (DeepEP), where the
  trace shows steps split into two micro-batches.

Each configuration runs direct, direct again, and through the API; steps are
matched by their inputs (batch composition can depend on request arrival),
and a configuration's matched steps are compared bitwise only once the direct
path repeats bitwise against itself. Responses must be identical throughout.
"""

import os
import tempfile
import threading
import unittest
from collections import defaultdict

import requests
import torch
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.model_runner_reference import read_step_trace, with_model_runner
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=3600, stage="extra-a", runner_config="2-gpu-large")

_DENSE = "Qwen/Qwen3-0.6B"
_MOE = "Qwen/Qwen3-30B-A3B"
# DeepEP's normal mode needs FP8 experts (unquantized experts run only in its
# low-latency mode); two-batch overlap for Qwen3 MoE needs DeepEP.
_MOE_FP8 = "Qwen/Qwen3-30B-A3B-FP8"
_COMMON = ["--tp-size", "2", "--disable-flashinfer-autotune"]
_NO_GRAPHS = ["--disable-decode-cuda-graph", "--disable-prefill-cuda-graph"]
_CP = [
    *_COMMON,
    "--attn-cp-size",
    "2",
    "--enable-prefill-cp",
    "--cp-strategy",
    "zigzag",
    "--mem-fraction-static",
    "0.6",
]
_EP = [*_COMMON, "--ep-size", "2", "--mem-fraction-static", "0.7"]
# Two micro-batches per step, experts dispatched by DeepEP (normal mode, which
# captures no graphs). Attention stays TP: under DP attention an idle rank
# beside a peer whose batch splits fails on the direct path in DeepEP's
# dispatch (the idle rank's empty micro-batches), whatever the runner.
_TBO = [
    *_COMMON,
    "--ep-size",
    "2",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "normal",
    "--enable-two-batch-overlap",
    *_NO_GRAPHS,
    "--mem-fraction-static",
    "0.7",
]
_PREAMBLE = "Answer in one short sentence, plainly and without lists. " * 12
_PROMPTS = [
    _PREAMBLE + "The capital of France is",
    "Write a haiku about the sea:",
    _PREAMBLE + "1, 2, 3, 4,",
    "In 1969, the first person to walk on the moon was",
]
_SIGNATURE = ("input_ids", "positions", "out_cache_loc", "seq_lens", "req_pool_indices")


def _generate(input_ids, *, max_new_tokens=16, dp_rank=None, ignore_eos=False, **extra):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
            "ignore_eos": ignore_eos,
        },
        **extra,
    }
    if dp_rank is not None:
        payload["routed_dp_rank"] = dp_rank
    response = requests.post(DEFAULT_URL_FOR_TEST + "/generate", json=payload)
    response.raise_for_status()
    return response.json()


def _concurrently(calls):
    results = [None] * len(calls)

    def run(i, fn):
        try:
            results[i] = fn()
        except Exception as e:  # re-raised below, in the test's thread
            results[i] = e

    threads = [threading.Thread(target=run, args=(i, c)) for i, c in enumerate(calls)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for result in results:
        if isinstance(result, Exception):
            raise result
    return results


def _serve(model, other_args, *, workload):
    trace_dir = tempfile.mkdtemp()
    process = popen_launch_server(
        model,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=with_model_runner(other_args),
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
    by_rank = defaultdict(list)
    for record in read_step_trace(trace_dir):
        by_rank[record["tp_rank"]].append(record)
    return result, {
        rank: sorted(records, key=lambda r: r["index"])
        for rank, records in by_rank.items()
    }


def _unique_by_inputs(records) -> dict:
    """Steps keyed by their inputs, without keys that repeat in the run."""
    by_inputs, repeated = {}, set()
    for record in records:
        if "tensors" not in record:
            continue
        key = tuple(
            record["tensors"][name].numpy().tobytes() for name in _SIGNATURE
        ) + (record["mode"], str(record["global_num_tokens"]))
        if key in by_inputs:
            repeated.add(key)
        by_inputs[key] = record
    return {key: r for key, r in by_inputs.items() if key not in repeated}


def _ids(outputs):
    return [
        [o["output_ids"] for o in (out if isinstance(out, list) else [out])]
        for out in outputs
    ]


class _Compare:
    model = _DENSE

    @classmethod
    def setUpClass(cls):
        tokenizer = AutoTokenizer.from_pretrained(cls.model)
        cls.prompts = [tokenizer.encode(p) for p in _PROMPTS]

    def _matched(self, expected, actual) -> tuple[int, int]:
        """(matched steps, matched steps whose logits are bitwise equal)."""
        expected = _unique_by_inputs(expected)
        matched = equal = 0
        for key, got in _unique_by_inputs(actual).items():
            want = expected.get(key)
            if want is None:
                continue
            matched += 1
            equal += all(
                torch.equal(want["tensors"][n], got["tensors"][n])
                for n in want["tensors"]
                if n not in _SIGNATURE
            )
        return matched, equal

    def _check(self, name, args, workload):
        """Direct, direct again, then the API; returns the API run's trace."""
        runs = [
            _serve(self.model, run_args, workload=workload)
            for run_args in (args, args, [*args, "--model-runner", "sglang"])
        ]
        (direct_out, direct), (again_out, again), (steps_out, steps) = runs
        self.assertEqual(set(steps), {0, 1})
        for rank in (0, 1):
            self.assertEqual({r["path"] for r in direct[rank]}, {"direct"})
            self.assertEqual({r["path"] for r in again[rank]}, {"direct"})
            self.assertEqual(
                {(r["path"], r["runner"]) for r in steps[rank]}, {("step", "sglang")}
            )
            for r in steps[rank]:
                self.assertEqual(r["step_graph"] is not None, r["replayed_graph"], r)
        counts = {
            rank: (
                self._matched(direct[rank], again[rank]),
                self._matched(direct[rank], steps[rank]),
            )
            for rank in (0, 1)
        }
        print(f"[{name}] (matched, bitwise) per rank (repeat, steps): {counts}")
        for rank, (repeat, through_api) in counts.items():
            self.assertGreater(through_api[0], 0, rank)
            # Bitwise wherever the direct path repeats bitwise.
            if repeat[0] == repeat[1]:
                self.assertEqual(through_api[1], through_api[0], (rank, counts))
        self.assertEqual(_ids(again_out), _ids(direct_out))
        self.assertEqual(_ids(steps_out), _ids(direct_out))
        return direct, steps


class TestContextParallel(_Compare, CustomTestCase):
    def workload(self):
        return [
            _generate(self.prompts),
            _generate(self.prompts[0], max_new_tokens=8),
            _generate(
                self.prompts[2],
                max_new_tokens=4,
                return_logprob=True,
                logprob_start_len=0,
                top_logprobs_num=2,
            ),
        ]

    def _check_cp(self, name, args):
        _, steps = self._check(name, args, self.workload)
        for rank in (0, 1):
            self.assertEqual({r["attn_cp_rank"] for r in steps[rank]}, {rank})
            sharded = [r for r in steps[rank] if r["cp_sharded"]]
            self.assertTrue(sharded, f"no CP-sharded extend on rank {rank}")
            self.assertEqual({r["mode"] for r in sharded}, {"extend"})
        return steps

    def test_context_parallel_eager(self):
        self._check_cp("cp2 eager", [*_CP, *_NO_GRAPHS])

    def test_context_parallel_with_graphs(self):
        steps = self._check_cp("cp2 graphs", [*_CP, "--cuda-graph-max-bs-decode", "4"])
        for rank in (0, 1):
            graphs = {
                r["step_graph"][0] for r in steps[rank] if r["step_graph"] is not None
            }
            self.assertEqual(graphs, {"decode", "prefill"})
            # CP prefill graphs replay the CP-sharded body.
            self.assertTrue(
                any(
                    r["cp_sharded"] and r["step_graph"][0] == "prefill"
                    for r in steps[rank]
                    if r["step_graph"] is not None
                )
            )


class TestExpertParallel(_Compare, CustomTestCase):
    model = _MOE

    def workload(self):
        return [
            _generate(self.prompts),
            _generate(self.prompts[1], max_new_tokens=24),
            _generate(
                self.prompts[0],
                max_new_tokens=4,
                return_logprob=True,
                logprob_start_len=0,
                top_logprobs_num=2,
            ),
        ]

    def test_expert_parallel_with_graphs(self):
        _, steps = self._check(
            "ep2 graphs", [*_EP, "--cuda-graph-max-bs-decode", "4"], self.workload
        )
        for rank in (0, 1):
            self.assertEqual({r["moe_ep_rank"] for r in steps[rank]}, {rank})
            self.assertTrue(any(r["replayed_graph"] for r in steps[rank]))

    def test_eplb_rebalances_on_the_step_path(self):
        """EPLB counts forwards and rebalances inside the native forward: on the
        step path the placement changes at the same forward as on the direct
        path, and the logits after it still match."""
        args = [
            *_EP,
            "--cuda-graph-max-bs-decode",
            "4",
            "--enable-eplb",
            "--eplb-rebalance-num-iterations",
            "16",
            # Remapping experts needs materialized top-k routing; the default
            # Blackwell MoE kernel routes from the logits inside the kernel.
            "--moe-runner-backend",
            "triton",
        ]
        direct, steps = self._check("ep2 eplb", args, self.workload)
        for rank in (0, 1):
            changes = [
                (r["index"], r["expert_layout"])
                for prev, r in zip(steps[rank], steps[rank][1:])
                if r["expert_layout"] != prev["expert_layout"]
            ]
            self.assertTrue(changes, f"no rebalance on rank {rank}")
            direct_changes = [
                (r["index"], r["expert_layout"])
                for prev, r in zip(direct[rank], direct[rank][1:])
                if r["expert_layout"] != prev["expert_layout"]
            ]
            self.assertEqual(changes, direct_changes)


class TestTwoBatchOverlap(_Compare, CustomTestCase):
    model = _MOE_FP8

    def workload(self):
        # Batches of two and four requests, whose extend and decode steps split.
        return [
            _generate(self.prompts, max_new_tokens=24, ignore_eos=True),
            _generate(self.prompts[:2], max_new_tokens=8, ignore_eos=True),
        ]

    def test_two_batch_overlap(self):
        _, steps = self._check("tbo ep2 deepep", _TBO, self.workload)
        for rank in (0, 1):
            split = [r for r in steps[rank] if r["tbo_split_seq_index"] is not None]
            self.assertTrue(split, f"no two-batch-overlap step on rank {rank}")
            # The forward built the two micro-batches of every split step.
            self.assertTrue(all(r["tbo_children"] for r in split))
            self.assertLessEqual({"extend", "decode"}, {r["mode"] for r in split})


if __name__ == "__main__":
    unittest.main(verbosity=3)
