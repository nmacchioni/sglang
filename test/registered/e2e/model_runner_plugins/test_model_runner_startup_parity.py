"""--model-runner sglang reaches the same final capacities and admission limits as
the direct path: default sizing, an explicit --max-total-tokens, and post-capture
KV resizing."""

import os
import tempfile
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=420, stage="base-b", runner_config="1-gpu-small")

# Small and ungated: dense full attention, so the plain MHA KV pool.
_MODEL = "Qwen/Qwen3-0.6B"
_STARTED = "Model runner 'sglang' started"
_RESIZED = "Post-capture KV sizing:"


def _launch(other_args, env=None):
    """Serve once, return (capacity facts, server log)."""
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        process = popen_launch_server(
            _MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env={**os.environ, **(env or {})},
            return_stdout_stderr=(out, err),
        )
        try:
            info = requests.get(DEFAULT_URL_FOR_TEST + "/server_info").json()
            generated = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"max_new_tokens": 8, "temperature": 0},
                },
            )
            generated.raise_for_status()
        finally:
            terminate_and_kill_process_tree(process)
        out.seek(0)
        err.seek(0)
        log = out.read() + err.read()
    state = info["internal_states"][0]
    facts = {
        "max_total_num_tokens": info["max_total_num_tokens"],
        "max_req_input_len": info["max_req_input_len"],
        "max_running_requests": state["effective_max_running_requests_per_dp"],
        "token_capacity": state["memory_usage"]["token_capacity"],
        "kvcache_gb": state["memory_usage"]["kvcache"],
    }
    return facts, log


class TestModelRunnerStartupParity(CustomTestCase):
    def _assert_parity(self, other_args, env=None):
        direct, direct_log = _launch(other_args, env)
        plugin, plugin_log = _launch([*other_args, "--model-runner", "sglang"], env)

        self.assertNotIn(_STARTED, direct_log)
        self.assertIn(_STARTED, plugin_log)
        self.assertEqual(plugin, direct)
        return direct, direct_log, plugin_log

    def test_default_sizing(self):
        self._assert_parity(["--disable-overlap-schedule"])

    def test_explicit_max_total_tokens(self):
        direct, _, _ = self._assert_parity(
            ["--max-total-tokens", "20000", "--max-running-requests", "24"]
        )
        self.assertLessEqual(direct["max_total_num_tokens"], 20000)
        self.assertLessEqual(direct["max_running_requests"], 24)

    def test_post_capture_resize(self):
        # Post-capture sizing needs every prefill batch to fit a captured
        # prefill graph bucket, so the chunk size is the largest bucket.
        _, direct_log, plugin_log = self._assert_parity(
            [
                "--chunked-prefill-size",
                "4096",
                "--cuda-graph-config",
                '{"prefill": {"backend": "breakable", "bs": [1024, 2048, 4096]}}',
            ],
            env={"SGLANG_ENABLE_POST_CAPTURE_KV_SIZING": "1"},
        )
        self.assertIn(_RESIZED, direct_log)
        self.assertIn(_RESIZED, plugin_log)


if __name__ == "__main__":
    unittest.main(verbosity=3)
