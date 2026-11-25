"""--model-runner sglang on models whose KV is more than full attention: a hybrid
sliding-window model (Gemma 3, window 512) and a hybrid recurrent-state model
(Granite 4.0 H, Mamba2 layers with prefix-cache checkpoints).

Each configuration runs direct, direct again, then through the API, and the
raw logits of every step are compared bitwise, after the direct-vs-direct run
shows the configuration repeats. The trace proves the step path ran with the
sub-pool tables (KVTables.window, KVTables.state) and that the features were
exercised: prompts longer than the window, prefix hits, eviction of the cached
prefix from a small pool, and state checkpoints that a later request resumes
from (a copy of the checkpoint into its state slot)."""

import os
import statistics
import tempfile
import unittest

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

register_cuda_ci(est_time=1500, stage="extra-a", runner_config="1-gpu-large")

_SWA_MODEL = "unsloth/gemma-3-1b-it"
_STATE_MODEL = "ibm-granite/granite-4.0-h-350m"
_COMMON = ["--disable-flashinfer-autotune", "--mem-fraction-static", "0.5"]
_EAGER = ["--disable-overlap-schedule", "--disable-decode-cuda-graph"]
_EAGER += ["--disable-prefill-cuda-graph"]
_GRAPHS = ["--cuda-graph-max-bs-decode", "4", "--disable-prefill-cuda-graph"]
# Small pools, so the cached prefixes of earlier requests are evicted.
_SWA_POOL = ["--max-total-tokens", "6144", "--swa-full-tokens-ratio", "0.5"]
_STATE_POOL = [
    "--max-mamba-cache-size",
    "12",
    "--mamba-radix-cache-strategy",
    "extra_buffer",
    "--mamba-track-interval",
    "256",
    "--max-total-tokens",
    "8192",
]
_CONFIGS = {
    "swa_eager": (_SWA_MODEL, [*_COMMON, *_EAGER, *_SWA_POOL]),
    "swa_overlap_graphs": (_SWA_MODEL, [*_COMMON, *_GRAPHS, *_SWA_POOL]),
    "state_eager": (_STATE_MODEL, [*_COMMON, *_EAGER, *_STATE_POOL]),
    "state_overlap_graphs": (_STATE_MODEL, [*_COMMON, *_GRAPHS, *_STATE_POOL]),
}
# About 700 tokens: past the 512-token window, and past two 256-token state
# checkpoint intervals.
_PREAMBLE = (
    "Below is a long list of facts about European capitals, rivers and "
    "mountains, written plainly, one per sentence. "
) * 30
_SIGNATURE = ("input_ids", "positions", "out_cache_loc", "seq_lens", "req_pool_indices")


def _generate(input_ids, *, max_new_tokens):
    response = requests.post(
        DEFAULT_URL_FOR_TEST + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
    )
    response.raise_for_status()
    return response.json()


def _serve(model, other_args, *, save_logits, workload):
    trace_dir = tempfile.mkdtemp()
    env = {
        **os.environ,
        "SGLANG_MODEL_RUNNER_STEP_TRACE_DIR": trace_dir,
        "SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS": "1" if save_logits else "0",
    }
    process = popen_launch_server(
        model,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=with_model_runner(other_args),
        env=env,
    )
    try:
        result = workload()
    finally:
        terminate_and_kill_process_tree(process)
    return result, read_step_trace(trace_dir)


class TestModelRunnerAdvancedKV(CustomTestCase):
    def _workload(self, model):
        def run():
            tokenizer = AutoTokenizer.from_pretrained(model)
            prompts = [
                tokenizer.encode(_PREAMBLE + f"Question {i}: name a river in {c}.")
                for i, c in enumerate(("France", "Spain", "Italy", "Norway"))
            ]
            fillers = [
                tokenizer.encode(f"Unrelated text number {i}: " + "lorem ipsum " * 600)
                for i in range(10)
            ]
            again = [p + tokenizer.encode(" Answer briefly.") for p in prompts[2:]]
            # Sequential batches: the same batches, rows and slots on every run.
            out = {"cold": _generate(prompts[:2], max_new_tokens=300)}
            out["hit"] = _generate(prompts[2:], max_new_tokens=16)
            # Prefix hits on the cached preamble; with recurrent state, resumed
            # from the checkpoints the previous batch's prefill left behind.
            out["hit_again"] = _generate(again, max_new_tokens=16)
            # Enough unrelated tokens to evict the preamble from the small pools.
            out["fill"] = [_generate(f, max_new_tokens=4) for f in fillers]
            out["after_eviction"] = _generate(again[0], max_new_tokens=16)
            return out

        return run

    def _check(self, name):
        model, args = _CONFIGS[name]
        workload = self._workload(model)
        runs = [
            _serve(model, run_args, save_logits=True, workload=workload)
            for run_args in (args, args, [*args, "--model-runner", "sglang"])
        ]
        (direct_out, direct), (again_out, again), (steps_out, steps) = runs
        self.assertEqual({r["path"] for r in direct + again}, {"direct"})
        self.assertEqual(
            {(r["path"], r["runner"]) for r in steps}, {("step", "sglang")}
        )

        cached = [r["meta_info"]["cached_tokens"] for r in steps_out["hit_again"]]
        evicted = steps_out["after_eviction"]["meta_info"]["cached_tokens"]
        print(f"[{name}] cached tokens on hits {cached}, after eviction {evicted}")
        self.assertTrue(all(c > 256 for c in cached), cached)
        self.assertLess(evicted, min(cached), "the fillers evicted nothing")
        self._check_sub_pool_tables(name, steps)

        repeat = self._assert_same_steps(direct, again)
        self._assert_same_outputs(direct_out, again_out)
        through_steps = self._assert_same_steps(direct, steps)
        self._assert_same_outputs(direct_out, steps_out)
        print(
            f"[{name}] bitwise logits: {repeat} steps direct vs direct, "
            f"{through_steps} steps direct vs steps"
        )
        self.assertGreater(through_steps, 0)
        if "graphs" in name:
            self.assertTrue(any(r["replayed_graph"] for r in steps))

    def _check_sub_pool_tables(self, name, steps):
        if name.startswith("swa"):
            self.assertTrue(all(r["kv_window"] for r in steps))
            self.assertTrue(all(r["kv_state"] is None for r in steps))
            for r in steps:
                if "tensors" not in r:
                    continue
                window = r["tensors"]["window_write_slots"]
                self.assertEqual(window.shape, r["tensors"]["out_cache_loc"].shape)
            # The window pool is half the full pool: the ids differ once it wraps.
            self.assertTrue(
                any(
                    not torch.equal(
                        r["tensors"]["window_write_slots"],
                        r["tensors"]["out_cache_loc"],
                    )
                    for r in steps
                    if "tensors" in r
                )
            )
            return
        self.assertTrue(all(r["kv_window"] is False for r in steps))
        ops = [r["kv_state"] for r in steps]
        totals = {k: sum(op[k] for op in ops) for k in ("clear", "copy", "checkpoints")}
        print(f"[{name}] state operations named by the Steps: {totals}")
        # Cold requests start from cleared state, hits resume from a
        # checkpoint copy, and long requests leave checkpoints behind.
        self.assertGreater(totals["clear"], 0)
        self.assertGreater(totals["copy"], 0)
        self.assertGreater(totals["checkpoints"], 0)
        for r in steps:
            if "tensors" in r and r["mode"] != "idle":
                slots = r["tensors"]["state_slots"].tolist()
                self.assertEqual(len(set(slots)), len(slots), "shared state slots")

    def _assert_same_steps(self, expected, actual):
        self.assertEqual([r["mode"] for r in actual], [r["mode"] for r in expected])
        compared = 0
        for want, got in zip(expected, actual):
            if "tensors" not in want:
                continue
            compared += 1
            for key in _SIGNATURE:
                self.assertTrue(
                    torch.equal(got["tensors"][key], want["tensors"][key]),
                    f"step {want['index']}: {key} differs",
                )
            want_logits, got_logits = (
                want["tensors"]["logits"],
                got["tensors"]["logits"],
            )
            self.assertTrue(
                torch.equal(got_logits, want_logits),
                f"step {want['index']} ({want['mode']}): logits differ, max abs "
                f"{(got_logits.float() - want_logits.float()).abs().max().item()}",
            )
        return compared

    def _assert_same_outputs(self, expected, actual):
        def ids(out):
            if isinstance(out, list):
                return [ids(o) for o in out]
            if isinstance(out, dict) and "output_ids" in out:
                return (out["output_ids"], out["meta_info"]["cached_tokens"])
            return {k: ids(v) for k, v in out.items()}

        self.assertEqual(ids(actual), ids(expected))

    def test_sliding_window_eager(self):
        self._check("swa_eager")

    def test_sliding_window_overlap_graphs(self):
        self._check("swa_overlap_graphs")

    def test_recurrent_state_eager(self):
        self._check("state_eager")

    def test_recurrent_state_overlap_graphs(self):
        self._check("state_overlap_graphs")

    def test_step_overhead(self):
        """Median host time outside execute per forward on the step path, and
        the direct path's host time per forward, for each model with graphs."""
        report = []
        for name in ("swa_overlap_graphs", "state_overlap_graphs"):
            model, args = _CONFIGS[name]
            workload = self._workload(model)
            _, direct = _serve(model, args, save_logits=False, workload=workload)
            _, steps = _serve(
                model,
                [*args, "--model-runner", "sglang"],
                save_logits=False,
                workload=workload,
            )
            for mode in ("extend", "decode"):
                d = [r["host_us"] for r in direct[2:] if r["mode"] == mode]
                s = [r for r in steps[2:] if r["mode"] == mode]
                if not d or not s:
                    continue
                report.append(
                    f"[{name}] {mode}: direct {statistics.median(d):.0f} us, steps "
                    f"{statistics.median(r['host_us'] for r in s):.0f} us, outside "
                    "execute "
                    f"{statistics.median(r['host_us'] - r['execute_us'] for r in s):.0f}"
                    f" us (n={len(d)}/{len(s)})"
                )
        print("STEP OVERHEAD\n" + "\n".join(report))
        self.assertTrue(report)


if __name__ == "__main__":
    unittest.main(verbosity=3)
