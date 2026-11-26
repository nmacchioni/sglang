"""Control operations through the model runner API.

The built-in runner, with the memory saver, overlap scheduling and decode
graphs on: one session generates, releases all device memory, resumes it,
reloads the weights (released weights come back uninitialized) and generates
again, loads a modified checkpoint from disk, generates, loads the
original back and generates once more. The session runs native, native again,
then with ``--model-runner sglang``: same outputs per phase, raw logits of
every step bitwise in order, every control request through the runner's
``control`` (the control trace), the KV epoch advanced by the resume, and the
steps after it on the step path, replaying their graphs.

The reference (external) runner declares two control operations, a weight
update from disk and a weight read, and SGLang answers every other with the
reply's own failed status, naming the operation."""

import glob
import json
import os
import shutil
import tempfile
import unittest

import requests
import torch
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.model_runner_reference import (
    REFERENCE_RUNNER,
    install_reference_runner,
    native_lane,
    read_step_trace,
)
from sglang.test.test_utils import (
    ADMIN_HEADERS_FOR_TEST,
    ADMIN_LAUNCH_ARGS_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=600, stage="extra-a", runner_config="1-gpu-large")

_MODEL = "Qwen/Qwen3-0.6B"
_COMMON = [
    *ADMIN_LAUNCH_ARGS_FOR_TEST,
    "--disable-flashinfer-autotune",
    "--mem-fraction-static",
    "0.5",
    "--cuda-graph-max-bs-decode",
    "4",
]
_BUILTIN = [*_COMMON, "--enable-memory-saver"]
_REFERENCE = [
    *_COMMON,
    "--disable-overlap-schedule",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
    "--model-runner",
    REFERENCE_RUNNER,
]
_PROMPTS = [
    "The capital of France is",
    "Write a haiku about the sea:",
    "def fib(n):\n    if n < 2:\n        return n\n",
]
_SIGNATURE = ("input_ids", "positions", "seq_lens")
# A norm inside the first layer, doubled: a checkpoint that changes the outputs
# (the final norm's scale would leave every argmax in place).
_SCALED = "model.layers.0.post_attention_layernorm.weight"


def _url(path):
    return DEFAULT_URL_FOR_TEST + path


def _post(path, payload=None):
    return requests.post(
        _url(path), json=payload or {}, headers=ADMIN_HEADERS_FOR_TEST, timeout=600
    )


def _workload(tokenizer):
    out = []
    for prompt in _PROMPTS:
        response = _post(
            "/generate",
            {
                "input_ids": tokenizer.encode(prompt),
                "sampling_params": {"max_new_tokens": 16, "temperature": 0},
            },
        )
        response.raise_for_status()
        out.append(response.json()["output_ids"])
    return out


def _modified_checkpoint(path: str) -> str:
    """A copy of the checkpoint at ``path`` with ``_SCALED`` doubled."""
    from safetensors.torch import load_file, save_file

    out = tempfile.mkdtemp()
    for name in os.listdir(path):
        if not name.endswith(".safetensors"):
            shutil.copy(os.path.join(path, name), out)
    for file in glob.glob(os.path.join(path, "*.safetensors")):
        tensors = load_file(file)
        if _SCALED in tensors:
            tensors[_SCALED] = tensors[_SCALED] * 2
        save_file(tensors, os.path.join(out, os.path.basename(file)))
    return out


def _read_controls(trace_dir):
    records = []
    for path in sorted(glob.glob(os.path.join(trace_dir, "controls-*.jsonl"))):
        with open(path) as f:
            records += [json.loads(line) for line in f]
    return records


class _Session:
    """One launch: generate, release, resume, update, restore."""

    def __init__(self, test, *, args, env=None, original, modified):
        self.test = test
        self.trace_dir = tempfile.mkdtemp()
        self.process = popen_launch_server(
            _MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env={
                **os.environ,
                **(env or {}),
                "SGLANG_MODEL_RUNNER_STEP_TRACE_DIR": self.trace_dir,
                "SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS": "1",
                **native_lane(args),
            },
        )
        self.original, self.modified = original, modified

    def close(self):
        terminate_and_kill_process_tree(self.process)
        return read_step_trace(self.trace_dir), _read_controls(self.trace_dir)

    def update_weights(self, path):
        response = _post("/update_weights_from_disk", {"model_path": path})
        self.test.assertEqual(response.status_code, 200, response.text)


class TestBuiltinRunnerControl(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from huggingface_hub import snapshot_download

        cls.tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        cls.original = snapshot_download(_MODEL)
        cls.modified = _modified_checkpoint(cls.original)

    def _session(self, args):
        session = _Session(
            self, args=args, original=self.original, modified=self.modified
        )
        phases = {}
        try:
            phases["before"] = _workload(self.tokenizer)
            for op in ("release", "resume"):
                response = _post(f"/{op}_memory_occupation")
                self.assertEqual(response.status_code, 200, response.text)
            # Released weights come back uninitialized, as in an RL loop.
            session.update_weights(self.original)
            phases["resumed"] = _workload(self.tokenizer)
            session.update_weights(self.modified)
            phases["modified"] = _workload(self.tokenizer)
            session.update_weights(self.original)
            phases["restored"] = _workload(self.tokenizer)
        finally:
            steps, controls = session.close()
        return phases, steps, controls

    def _assert_steps_bitwise(self, want, got):
        want = [r for r in want if "tensors" in r]
        got = [r for r in got if "tensors" in r]
        self.assertEqual(len(got), len(want))
        for a, b in zip(want, got):
            for name in _SIGNATURE:
                self.assertTrue(
                    torch.equal(a["tensors"][name], b["tensors"][name]),
                    f"step {a['index']}: {name} differ",
                )
            self.assertTrue(
                torch.equal(a["tensors"]["logits"], b["tensors"]["logits"]),
                f"step {a['index']} ({a['mode']}): logits differ",
            )
        return len(got)

    def test_release_resume_and_weight_updates_match_the_native_lane(self):
        native, native_steps, native_controls = self._session(_BUILTIN)
        again, again_steps, _ = self._session(_BUILTIN)
        api, api_steps, controls = self._session(
            [*_BUILTIN, "--model-runner", "sglang"]
        )
        # Each phase: same outputs on every path; resumed and restored weights
        # serve what they served before, the modified checkpoint differs.
        self.assertEqual(again, native)
        self.assertEqual(api, native)
        self.assertEqual(native["resumed"], native["before"])
        self.assertEqual(native["restored"], native["before"])
        self.assertNotEqual(native["modified"], native["before"])
        repeat = self._assert_steps_bitwise(native_steps, again_steps)
        through_api = self._assert_steps_bitwise(native_steps, api_steps)
        self.assertEqual({r["path"] for r in api_steps}, {"step"})
        # The native lane runs its control through the same runner.
        self.assertEqual(
            [(c["api"], c["success"], c["epoch"]) for c in native_controls],
            [(c["api"], c["success"], c["epoch"]) for c in controls],
        )
        self.assertEqual(
            [(c["api"], c["success"], c["epoch"]) for c in controls],
            [
                ("release_memory_occupation", True, 0),
                ("resume_memory_occupation", True, 1),
                ("update_weights_from_disk", True, 1),
                ("update_weights_from_disk", True, 1),
                ("update_weights_from_disk", True, 1),
            ],
        )
        decodes = [r for r in api_steps if r["mode"] == "decode"]
        replayed = sum(r["replayed_graph"] for r in decodes)
        print(
            f"[builtin control] steps bitwise: native again {repeat}, API "
            f"{through_api}; decode graph replays {replayed}/{len(decodes)}; "
            f"control host us {[round(c['host_us']) for c in controls]}"
        )
        # Graphs replay after the release and resume (and the weight loads).
        self.assertGreater(sum(r["replayed_graph"] for r in decodes[-10:]), 0)


class TestReferenceRunnerControl(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from huggingface_hub import snapshot_download

        cls.tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        cls.original = snapshot_download(_MODEL)
        cls.modified = _modified_checkpoint(cls.original)
        cls.env = install_reference_runner(tempfile.mkdtemp())

    def _norm(self):
        response = _post("/get_weights_by_name", {"name": _SCALED, "truncate_size": 4})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_declared_ops_run_and_others_are_refused(self):
        session = _Session(
            self,
            args=_REFERENCE,
            env=self.env,
            original=self.original,
            modified=self.modified,
        )
        try:
            before, norm = _workload(self.tokenizer), self._norm()
            session.update_weights(self.modified)
            modified, scaled = _workload(self.tokenizer), self._norm()
            session.update_weights(self.original)
            restored, norm_again = _workload(self.tokenizer), self._norm()
            refused = {
                path: _post(path, payload)
                for path, payload in (
                    ("/release_memory_occupation", {}),
                    (
                        "/init_weights_update_group",
                        {
                            "master_address": "127.0.0.1",
                            "master_port": 29999,
                            "rank_offset": 1,
                            "world_size": 2,
                        },
                    ),
                    ("/weights_checker", {"action": "snapshot"}),
                )
            }
            flushed = _post("/flush_cache")
            after_flush = _workload(self.tokenizer)
        finally:
            steps, controls = session.close()
        self.assertEqual({r["runner"] for r in steps}, {REFERENCE_RUNNER})
        self.assertEqual(restored, before)
        self.assertEqual(after_flush, before)
        self.assertNotEqual(modified, before)
        self.assertEqual(norm_again, norm)
        self.assertEqual(scaled, [2 * x for x in norm], "the weight after the update")
        for path, response in refused.items():
            self.assertNotEqual(response.status_code, 200, path)
        api_op = {
            "/release_memory_occupation": ("release_memory_occupation", "release"),
            "/init_weights_update_group": (
                "init_weights_update_group",
                "weights_update_group",
            ),
            "/weights_checker": ("weights_checker", "weights_checksum"),
        }
        for path, (api, op) in api_op.items():
            self.assertIn(
                f"--model-runner {REFERENCE_RUNNER} does not support {api} "
                f"(control op {op}; serve with --model-runner sglang)",
                refused[path].text,
            )
        self.assertEqual(flushed.status_code, 200, flushed.text)
        self.assertEqual(
            [(c["op"], c["success"]) for c in controls],
            [
                ("weights_get", True),
                ("weights_update_from_disk", True),
                ("weights_get", True),
                ("weights_update_from_disk", True),
                ("weights_get", True),
                ("release", False),
                ("weights_update_group", False),
                ("weights_checksum", False),
            ],
        )
        print(f"[reference control] {[(c['api'], c['success']) for c in controls]}")


if __name__ == "__main__":
    unittest.main(verbosity=3)
