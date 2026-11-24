"""LoRA through the model runner API.

Qwen3-0.6B serves three LoRA adapters (written by the test: random PEFT
adapters on every attention and MLP projection) with two adapter slots, so
SGLang evicts and reloads slots while other requests keep decoding. A batch of
requests over the three adapters and the base model, then an adapter loaded at
runtime, used and unloaded, and the others again, run direct, direct again, then with
``--model-runner sglang``, eager and with overlap scheduling and decode
graphs: identical outputs, raw logits bitwise for every step whose inputs
(adapters included) the other run also has, every step of the API run on the
step path with its Step.adapters slots, at least
one slot serving different adapters over the run while requests of the
other slot were in flight, and the load and unload through the runner's
``control``."""

import glob
import json
import os
import tempfile
import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.model_runner_reference import read_step_trace, with_model_runner
from sglang.test.test_utils import (
    ADMIN_HEADERS_FOR_TEST,
    ADMIN_LAUNCH_ARGS_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=900, stage="extra-a", runner_config="1-gpu-large")

_MODEL = "Qwen/Qwen3-0.6B"
_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_RANK = 8
_PROMPTS = [
    "The capital of France is",
    "Write a haiku about the sea:",
    "def fib(n):\n    if n < 2:\n        return n\n",
    "List three planets:",
    "Once upon a time,",
    "The best way to learn a language is",
    "import numpy as np\n",
    "Norway is known for",
]
# Three adapters and the base model over two slots.
_ADAPTERS = ["a", "b", None, "c", "a", "c", "b", None]
_COMMON = [
    *ADMIN_LAUNCH_ARGS_FOR_TEST,
    "--disable-flashinfer-autotune",
    "--mem-fraction-static",
    "0.4",
    "--enable-lora",
    "--max-loras-per-batch",
    "2",
    "--max-lora-rank",
    str(_RANK),
    "--lora-target-modules",
    *_TARGETS,
    "--cuda-graph-max-bs-decode",
    "8",
]
_EAGER = [
    "--disable-overlap-schedule",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
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
            tuple(r["lora_ids"] or ()),
        )
        seen[key] = seen.get(key, 0) + 1
        by_key[key] = r
    return {key: r for key, r in by_key.items() if seen[key] == 1}, set(seen)


def _write_adapter(directory: str, *, seed: int, config: dict) -> str:
    """A random PEFT LoRA adapter of rank ``_RANK`` on every target projection."""
    from safetensors.torch import save_file

    hidden = config["hidden_size"]
    head_dim = config.get("head_dim") or hidden // config["num_attention_heads"]
    q = config["num_attention_heads"] * head_dim
    kv = config["num_key_value_heads"] * head_dim
    inter = config["intermediate_size"]
    shapes = {
        "self_attn.q_proj": (hidden, q),
        "self_attn.k_proj": (hidden, kv),
        "self_attn.v_proj": (hidden, kv),
        "self_attn.o_proj": (q, hidden),
        "mlp.gate_proj": (hidden, inter),
        "mlp.up_proj": (hidden, inter),
        "mlp.down_proj": (inter, hidden),
    }
    generator = torch.Generator().manual_seed(seed)
    tensors = {}
    for layer in range(config["num_hidden_layers"]):
        for module, (fan_in, fan_out) in shapes.items():
            prefix = f"base_model.model.model.layers.{layer}.{module}"
            tensors[f"{prefix}.lora_A.weight"] = (
                torch.randn(_RANK, fan_in, generator=generator) / fan_in**0.5
            ).to(torch.bfloat16)
            tensors[f"{prefix}.lora_B.weight"] = (
                torch.randn(fan_out, _RANK, generator=generator) * 0.01
            ).to(torch.bfloat16)
    os.makedirs(directory, exist_ok=True)
    save_file(tensors, os.path.join(directory, "adapter_model.safetensors"))
    with open(os.path.join(directory, "adapter_config.json"), "w") as f:
        json.dump(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "base_model_name_or_path": _MODEL,
                "r": _RANK,
                "lora_alpha": 2 * _RANK,
                "lora_dropout": 0.0,
                "bias": "none",
                "target_modules": list(_TARGETS),
            },
            f,
        )
    return directory


def _post(path, payload):
    return requests.post(
        DEFAULT_URL_FOR_TEST + path,
        json=payload,
        headers=ADMIN_HEADERS_FOR_TEST,
        timeout=300,
    )


def _generate(prompts, adapters):
    response = _post(
        "/generate",
        {
            "text": prompts,
            "lora_path": adapters,
            "sampling_params": {"temperature": 0, "max_new_tokens": 24},
        },
    )
    return response.status_code, response.json()


class TestModelRunnerLoRA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from huggingface_hub import snapshot_download

        with open(os.path.join(snapshot_download(_MODEL), "config.json")) as f:
            config = json.load(f)
        root = tempfile.mkdtemp()
        cls.paths = {
            name: _write_adapter(os.path.join(root, name), seed=seed, config=config)
            for seed, name in enumerate(("a", "b", "c", "d"))
        }

    def _workload(self):
        out = {}
        status, out["mixed"] = _generate(_PROMPTS, _ADAPTERS)
        self.assertEqual(status, 200, out["mixed"])
        response = _post(
            "/load_lora_adapter", {"lora_name": "d", "lora_path": self.paths["d"]}
        )
        self.assertEqual(response.status_code, 200, response.text)
        status, out["loaded"] = _generate(_PROMPTS[:4], ["d", "a", None, "d"])
        self.assertEqual(status, 200, out["loaded"])
        response = _post("/unload_lora_adapter", {"lora_name": "d"})
        self.assertEqual(response.status_code, 200, response.text)
        # The slot d held serves the others again.
        status, out["after_unload"] = _generate(_PROMPTS[4:], ["c", "b", None, "a"])
        self.assertEqual(status, 200, out["after_unload"])
        return {
            key: (
                [r["output_ids"] for r in value] if isinstance(value, list) else value
            )
            for key, value in out.items()
        }

    def _serve(self, args):
        trace_dir = tempfile.mkdtemp()
        lora_paths = [f"{name}={self.paths[name]}" for name in ("a", "b", "c")]
        process = popen_launch_server(
            _MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=with_model_runner(
                [*_COMMON, "--lora-paths", *lora_paths, *args]
            ),
            env={
                **os.environ,
                "SGLANG_MODEL_RUNNER_STEP_TRACE_DIR": trace_dir,
                "SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS": "1",
            },
        )
        try:
            out = self._workload()
        finally:
            terminate_and_kill_process_tree(process)
        controls = []
        for path in glob.glob(os.path.join(trace_dir, "controls-*.jsonl")):
            with open(path) as f:
                controls += [json.loads(line) for line in f]
        return out, read_step_trace(trace_dir), controls

    def _assert_steps_bitwise(self, want, got, *, output="logits"):
        """Steps matched by inputs bitwise; every step of one run whose inputs
        the other run has is compared."""
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
        # The adapter-slot admission makes batch composition depend on timing,
        # so even the direct path repeats only part of its steps.
        self.assertTrue(matched)
        return f"{len(matched)}/{len(got_seen)}"

    def _case(self, name, args):
        direct, direct_steps, _ = self._serve(args)
        again, again_steps, _ = self._serve(args)
        api, api_steps, controls = self._serve([*args, "--model-runner", "sglang"])
        self.assertEqual(again, direct)
        self.assertEqual(api, direct)
        # Prompt 0 under adapter a, then d, answers differently; prompt 2 under
        # the base model answers alike in both batches.
        mixed, loaded = direct["mixed"], direct["loaded"]
        self.assertNotEqual(loaded[0], mixed[0])
        self.assertEqual(loaded[2], mixed[2])
        repeat = self._assert_steps_bitwise(direct_steps, again_steps)
        through_api = self._assert_steps_bitwise(direct_steps, api_steps)
        self.assertEqual({r["path"] for r in api_steps}, {"step"})
        lora_steps = [r for r in api_steps if r["lora_ids"] is not None]
        self.assertTrue(lora_steps)
        self.assertTrue(all(r["adapters"] is not None for r in lora_steps))
        # Which adapters each slot served, and whether a slot changed adapter
        # while the other slot's requests were still decoding.
        served, reused_in_flight = {}, 0
        for r in lora_steps:
            slots = dict(zip(r["adapters"], r["lora_ids"]))
            for slot, uid in slots.items():
                before = served.setdefault(slot, [])
                if before and before[-1] != uid and len(slots) > 1:
                    reused_in_flight += 1
                if not before or before[-1] != uid:
                    before.append(uid)
        self.assertTrue(set(served) <= {0, 1}, served)
        self.assertTrue(any(len(set(uids)) > 1 for uids in served.values()), served)
        self.assertEqual(
            [(c["op"], c["success"]) for c in controls],
            [("lora_load", True), ("lora_unload", True)],
        )
        outside = sorted(r["host_us"] - r["execute_us"] for r in api_steps[2:])
        boundary = {
            label: sorted(r["host_us"] for r in run[2:])
            for label, run in (("direct", direct_steps), ("API", api_steps))
        }
        print(
            f"[lora {name}] steps bitwise: direct again {repeat}, API {through_api}; "
            f"LoRA steps {len(lora_steps)}; slot history {served}; slot changes "
            f"beside a running slot {reused_in_flight}; graph replays "
            f"{sum(r['replayed_graph'] for r in api_steps)}; median host time "
            f"outside execute {outside[len(outside) // 2]:.0f} us; at the worker "
            f"boundary {', '.join(f'{k} {v[len(v) // 2]:.0f} us' for k, v in boundary.items())}"
        )

    def test_lora_eager(self):
        self._case("eager", _EAGER)

    def test_lora_overlap_and_graphs(self):
        self._case("overlap+graphs", [])


if __name__ == "__main__":
    unittest.main(verbosity=3)
