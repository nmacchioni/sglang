"""Multimodal requests through the model runner API.

Qwen3.5-0.8B (a vision-language model with M-RoPE) serves image requests --
one image, two images, a repeated image, and text alone -- native, native
again, then with ``--model-runner sglang``, eager and with overlap scheduling
and graphs: identical outputs, raw logits of every step bitwise in order, and
every step of the API run on the step path, its extends carrying the images'
items. Then both frontends resolve SGLang's processor for the built-in
runner: the Rust frontend (``SGLANG_RUST_SERVER=1``) serves the same image
requests through the API as it does on the native lane."""

import base64
import importlib.util
import io
import os
import tempfile
import unittest

import numpy as np
import requests
import torch
from PIL import Image

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.model_runner_reference import native_lane, read_step_trace
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=900, stage="extra-a", runner_config="1-gpu-large")

_MODEL = "Qwen/Qwen3.5-0.8B"
_COMMON = [
    "--disable-flashinfer-autotune",
    "--mem-fraction-static",
    "0.5",
    "--cuda-graph-max-bs-decode",
    "4",
]
_EAGER = [
    "--disable-overlap-schedule",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
]
_VISION = "<|vision_start|><|image_pad|><|vision_end|>"
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
        )
        seen[key] = seen.get(key, 0) + 1
        by_key[key] = r
    return {key: r for key, r in by_key.items() if seen[key] == 1}, set(seen)


def _prompt(question, images=1):
    return (
        f"<|im_start|>user\n{_VISION * images}{question}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _image(color, *, stripes=False):
    pixels = np.full((96, 128, 3), color, dtype=np.uint8)
    if stripes:
        pixels[::16] = 255 - np.array(color, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


_RED, _BLUE = _image((220, 20, 20)), _image((20, 40, 220), stripes=True)
_REQUESTS = [
    (_prompt("What color is this image? One word."), [_RED]),
    (_prompt("Describe these two images briefly.", images=2), [_RED, _BLUE]),
    # The same image again: the prefix and encoder caches may serve it.
    (_prompt("What color is this image? One word."), [_RED]),
    (_prompt("Name a primary color.", images=0), None),
]


def _workload():
    out = []
    for text, images in _REQUESTS:
        payload = {
            "text": text,
            "sampling_params": {"temperature": 0, "max_new_tokens": 24},
        }
        if images is not None:
            payload["image_data"] = images
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate", json=payload, timeout=300
        )
        response.raise_for_status()
        out.append(response.json()["output_ids"])
    return out


def _serve(args, env=None):
    trace_dir = tempfile.mkdtemp()
    process = popen_launch_server(
        _MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=[*_COMMON, *args],
        env={
            **os.environ,
            **(env or {}),
            "SGLANG_MODEL_RUNNER_STEP_TRACE_DIR": trace_dir,
            "SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS": "1",
            **native_lane([*_COMMON, *args]),
        },
    )
    try:
        out = _workload()
    finally:
        terminate_and_kill_process_tree(process)
    return out, read_step_trace(trace_dir)


class TestModelRunnerMultimodal(CustomTestCase):
    def _assert_steps_bitwise(self, want, got, *, output="logits"):
        """Steps matched by inputs bitwise; every step of one run whose inputs
        the other run has is compared, and few have none."""
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
        self.assertLessEqual(
            len(got_seen - want_seen), 0.1 * len(got_seen), "steps without a match"
        )
        self.assertTrue(matched)
        return len(matched)

    def _case(self, name, args, env=None):
        native, native_steps = _serve(args, env)
        again, again_steps = _serve(args, env)
        api, api_steps = _serve([*args, "--model-runner", "sglang"], env)
        self.assertEqual(again, native)
        self.assertEqual(api, native)
        repeat = self._assert_steps_bitwise(native_steps, again_steps)
        through_api = self._assert_steps_bitwise(native_steps, api_steps)
        self.assertEqual({r["path"] for r in api_steps}, {"step"})
        mm_extends = [r for r in api_steps if r["mode"] == "extend" and r["mm_items"]]
        self.assertTrue(mm_extends)
        outside = sorted(r["host_us"] - r["execute_us"] for r in api_steps[2:])
        boundary = {
            label: sorted(r["host_us"] for r in run[2:])
            for label, run in (("native", native_steps), ("API", api_steps))
        }
        print(
            f"[multimodal {name}] steps bitwise: native again {repeat}, API "
            f"{through_api}; extends with images {len(mm_extends)} "
            f"(items {[r['mm_items'] for r in mm_extends]}); graph replays "
            f"{sum(r['replayed_graph'] for r in api_steps)}; median host time "
            f"outside execute {outside[len(outside) // 2]:.0f} us; at the worker "
            f"boundary {', '.join(f'{k} {v[len(v) // 2]:.0f} us' for k, v in boundary.items())}"
        )
        return native

    def test_images_eager(self):
        self._case("eager", _EAGER)

    def test_images_overlap_and_graphs(self):
        self._case("overlap+graphs", [])

    @unittest.skipIf(
        importlib.util.find_spec("sglang.srt.rust_extensions._server") is None,
        "sglang-server rust extension not installed",
    )
    def test_rust_frontend_resolves_the_processor(self):
        self._case("rust frontend", [], env={"SGLANG_RUST_SERVER": "1"})


if __name__ == "__main__":
    unittest.main(verbosity=3)
