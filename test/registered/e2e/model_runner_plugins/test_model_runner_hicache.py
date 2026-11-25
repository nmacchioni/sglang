"""Hierarchical cache (HiCache) through the model runner API.

The built-in runner: HiCache on, a device pool small enough that cached
prefixes are evicted to host memory and loaded back; direct, direct again,
then through the API, raw logits of every step bitwise.

The reference (external) runner: SGLang backs up and loads the runner's own KV
bytes, viewed through its KVLayout. The same workload runs on the reference
runner without HiCache and a large pool (every hit served from device), then
with HiCache and a small pool (hits loaded from host). For each prefix hit,
the step trace records a CRC of the prefix's KV bytes as the runner's step
reads them: equal CRCs are the byte-equal round trip, equal logits of the
steps that read the loaded prefix are the reconstructed values, and equal
continuations are the resumed decode."""

import os
import statistics
import tempfile
import unittest

import requests
import torch
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.model_runner_reference import (
    REFERENCE_RUNNER,
    install_reference_runner,
    read_step_trace,
    with_model_runner,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=900, stage="extra-a", runner_config="1-gpu-large")

_MODEL = "Qwen/Qwen3-0.6B"
_COMMON = [
    "--disable-flashinfer-autotune",
    "--mem-fraction-static",
    "0.5",
    "--disable-overlap-schedule",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
    "--page-size",
    "64",
]
# Room for about three prompts on device, all of them on host.
_SMALL_POOL = ["--max-total-tokens", "4096"]
_HICACHE = [
    "--enable-hierarchical-cache",
    "--hicache-size",
    "2",
    "--hicache-write-policy",
    "write_through",
]
_PREAMBLE = "Here are some notes on the history of European cartography. " * 60
_STEP_KEY = ("input_ids", "positions", "seq_lens")


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


def _serve(other_args, *, workload, env=None):
    trace_dir = tempfile.mkdtemp()
    env = {
        **os.environ,
        **(env or {}),
        "SGLANG_MODEL_RUNNER_STEP_TRACE_DIR": trace_dir,
        "SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS": "1",
    }
    process = popen_launch_server(
        _MODEL,
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


def _host_tokens(out) -> int:
    details = out["meta_info"].get("cached_tokens_details") or {}
    return details.get("host", 0)


def _outputs(out):
    if isinstance(out, list):
        return [_outputs(o) for o in out]
    if isinstance(out, dict) and "output_ids" in out:
        return out["output_ids"]
    return {k: _outputs(v) for k, v in out.items()}


def _unique_by_key(trace) -> dict:
    by_key, repeated = {}, set()
    for record in trace:
        if "tensors" not in record:
            continue
        key = tuple(record["tensors"][k].numpy().tobytes() for k in _STEP_KEY) + (
            record["mode"],
        )
        if key in by_key:
            repeated.add(key)
        by_key[key] = record
    return {key: r for key, r in by_key.items() if key not in repeated}


class TestModelRunnerHiCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        cls.prompts = [
            tokenizer.encode(_PREAMBLE + f"Chapter {i}: the maps of {city}.")
            for i, city in enumerate(("Venice", "Lisbon", "Antwerp"))
        ]
        cls.fillers = [
            tokenizer.encode(f"Filler {i}: " + "unrelated words here " * 250)
            for i in range(6)
        ]
        cls.follow_ups = [
            p + tokenizer.encode(" Continue the story.") for p in cls.prompts
        ]
        cls.reference_env = install_reference_runner(tempfile.mkdtemp())

    def _workload(self):
        out = {"cold": [_generate(p, max_new_tokens=32) for p in self.prompts]}
        # Evict the prompts from the small device pool; HiCache keeps them on host.
        out["fill"] = [_generate(f, max_new_tokens=4) for f in self.fillers]
        # Each follow-up resumes a prefix: from host when HiCache is on.
        out["resume"] = [_generate(p, max_new_tokens=48) for p in self.follow_ups]
        return out

    def test_builtin_runner_with_hicache_is_bitwise_equal_to_the_direct_path(self):
        args = [*_COMMON, *_SMALL_POOL, *_HICACHE]
        direct_out, direct = _serve(args, workload=self._workload)
        again_out, again = _serve(args, workload=self._workload)
        steps_out, steps = _serve(
            [*args, "--model-runner", "sglang"], workload=self._workload
        )
        self.assertEqual({r["path"] for r in direct + again}, {"direct"})
        self.assertEqual({r["path"] for r in steps}, {"step"})
        host = [_host_tokens(o) for o in steps_out["resume"]]
        print(f"[builtin] host-loaded prefix tokens per follow-up: {host}")
        self._assert_resumed_from_host(steps_out, host)
        for want_out, want, got_out, got in (
            (direct_out, direct, again_out, again),
            (direct_out, direct, steps_out, steps),
        ):
            self.assertEqual(_outputs(got_out), _outputs(want_out))
            self.assertEqual(len(got), len(want))
            compared = 0
            for w, g in zip(want, got):
                if "tensors" not in w:
                    continue
                for key in ("input_ids", "positions", "out_cache_loc", "seq_lens"):
                    self.assertTrue(torch.equal(g["tensors"][key], w["tensors"][key]))
                self.assertTrue(
                    torch.equal(g["tensors"]["logits"], w["tensors"]["logits"]),
                    f"step {w['index']}: logits differ",
                )
                compared += 1
            print(f"[builtin] bitwise logits on {compared} steps")
            self.assertGreater(compared, 0)

    def test_reference_runner_round_trips_its_kv_through_host_memory(self):
        reference = ["--model-runner", REFERENCE_RUNNER]
        device_out, device = _serve(
            [*_COMMON, *reference], workload=self._workload, env=self.reference_env
        )
        host_out, host = _serve(
            [*_COMMON, *_SMALL_POOL, *_HICACHE, *reference],
            workload=self._workload,
            env=self.reference_env,
        )
        self.assertEqual({r["path"] for r in device + host}, {"step"})
        loaded = [_host_tokens(o) for o in host_out["resume"]]
        print(f"[reference] host-loaded prefix tokens per follow-up: {loaded}")
        self._assert_resumed_from_host(host_out, loaded)
        self.assertEqual(
            [o["meta_info"]["cached_tokens"] for o in host_out["resume"]],
            [o["meta_info"]["cached_tokens"] for o in device_out["resume"]],
        )
        # Resumed decode: the same continuations, token for token.
        self.assertEqual(_outputs(host_out["resume"]), _outputs(device_out["resume"]))

        # Steps of the follow-ups, matched by what they compute (their slots differ).
        # Steps of different requests can share inputs (same token at the same
        # position and length); only steps whose inputs occur once are matched.
        by_key = _unique_by_key(device)
        matched, crcs, differ = 0, 0, []
        for key, got in _unique_by_key(host).items():
            want = by_key.get(key)
            if want is None:
                continue
            matched += 1
            if not torch.equal(got["tensors"]["logits"], want["tensors"]["logits"]):
                diff = (
                    got["tensors"]["logits"].float() - want["tensors"]["logits"].float()
                )
                differ.append(
                    (got["index"], want["index"], got["mode"], diff.abs().max().item())
                )
            if got["prefix_kv_crc"]:
                # Byte-equal: the prefix the step read, loaded back from host,
                # holds the bytes the device-resident prefix held.
                self.assertEqual(got["prefix_kv_crc"], want["prefix_kv_crc"])
                crcs += len(got["prefix_kv_crc"])
        print(f"[reference] matched steps {matched}, prefix KV CRCs equal {crcs}")
        print(f"[reference] steps whose logits differ: {differ}")
        for name, trace in (("device", device), ("hicache", host)):
            for mode in ("extend", "decode"):
                outside = [
                    r["host_us"] - r["execute_us"]
                    for r in trace[2:]
                    if r["mode"] == mode
                ]
                print(
                    f"[reference, {name}] {mode}: median host time outside "
                    f"execute {statistics.median(outside):.0f} us (n={len(outside)})"
                )
        self.assertEqual(differ, [])
        self.assertGreater(crcs, 0)

    def _assert_resumed_from_host(self, out, host_tokens):
        # The follow-ups share the preamble: the first loads it back from host,
        # the others then find it on device.
        self.assertGreater(host_tokens[0], 0)
        cached = [o["meta_info"]["cached_tokens"] for o in out["resume"]]
        self.assertTrue(all(c >= host_tokens[0] for c in cached), cached)


if __name__ == "__main__":
    unittest.main(verbosity=3)
