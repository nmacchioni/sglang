"""Rank failure under --model-runner sglang (TP=2).

A rank that dies or stalls mid-step tears its peers down within a bound,
instead of leaving them in a collective forever: the bound is the scheduler
watchdog's, which the step path runs under like the direct path. A rank whose
step does not match what it published at the per-step rendezvous refuses the
step before entering a collective, so the server fails fast, well inside that
bound, with the mismatch named.
"""

import os
import re
import signal
import threading
import time
import unittest

import psutil
import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=600, stage="extra-a", runner_config="2-gpu-large")

_MODEL = "Qwen/Qwen3-0.6B"
_WATCHDOG_S = 20
_ARGS = [
    "--tp-size",
    "2",
    "--model-runner",
    "sglang",
    "--disable-flashinfer-autotune",
    "--watchdog-timeout",
    str(_WATCHDOG_S),
]
# The watchdog checks every half timeout, then the launcher kills the tree.
_TEARDOWN_BOUND_S = 3 * _WATCHDOG_S + 30
# The launcher's crash diagnostics (py-spy, then a 60 s wait for CUDA
# coredumps) delay teardown by a configured amount that no runner affects.
_NO_CRASH_DIAGNOSTICS = {
    "SGLANG_PYSPY_DUMP_BEFORE_CRASH": "0",
    "SGLANG_CUDA_COREDUMP_BEFORE_CRASH": "0",
}
# A refused step raises at once, then the launcher kills the tree.
_FAIL_FAST_BOUND_S = 30


def _long_generation():
    try:
        requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "text": "Count from one to ten thousand:",
                "sampling_params": {
                    "max_new_tokens": 8000,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
            timeout=_TEARDOWN_BOUND_S * 2,
        )
    except requests.RequestException:
        pass


def _scheduler(process, tp_rank: int) -> psutil.Process:
    for child in psutil.Process(process.pid).children(recursive=True):
        try:
            title = " ".join(child.cmdline())
        except psutil.Error:
            continue
        if title.startswith("sglang::scheduler") and re.search(
            rf"TP{tp_rank}\b", title
        ):
            return child
    raise AssertionError(f"no scheduler process for TP rank {tp_rank}")


def _seconds_until_gone(process, pids, bound) -> float:
    tic = time.perf_counter()
    while time.perf_counter() - tic < bound:
        alive = [p for p in pids if psutil.pid_exists(p) and _running(p)]
        if process.poll() is not None and not alive:
            return time.perf_counter() - tic
        time.sleep(1)
    raise AssertionError(f"server still up {bound} s after the rank failed")


def _running(pid) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _generate_on(*, dp_rank: int, timeout: float) -> requests.Response:
    return requests.post(
        DEFAULT_URL_FOR_TEST + "/generate",
        json={
            "text": "The capital of France is",
            "sampling_params": {"max_new_tokens": 8},
            "routed_dp_rank": dp_rank,
        },
        timeout=timeout,
    )


class TestRankFailure(CustomTestCase):
    def _fail_rank_mid_decode(self, sig):
        process = popen_launch_server(
            _MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_ARGS,
            env={**os.environ, **_NO_CRASH_DIAGNOSTICS},
        )
        try:
            pids = [p.pid for p in psutil.Process(process.pid).children(True)]
            victim = _scheduler(process, tp_rank=1)
            peer = _scheduler(process, tp_rank=0)
            worker = threading.Thread(target=_long_generation, daemon=True)
            worker.start()
            # Decoding by now: both ranks sit in the step's collectives.
            time.sleep(5)
            os.kill(victim.pid, sig)
            seconds = _seconds_until_gone(process, pids, _TEARDOWN_BOUND_S)
            print(f"[{signal.Signals(sig).name}] every rank gone after {seconds:.1f} s")
            self.assertFalse(psutil.pid_exists(peer.pid) and _running(peer.pid))
        finally:
            if process.poll() is None:
                os.kill(victim.pid, signal.SIGCONT)
            terminate_and_kill_process_tree(process)

    def test_killed_rank_tears_down_its_peers_within_the_bound(self):
        self._fail_rank_mid_decode(signal.SIGKILL)

    def test_stalled_rank_tears_down_its_peers_within_the_bound(self):
        self._fail_rank_mid_decode(signal.SIGSTOP)


class TestRendezvousMismatch(CustomTestCase):
    def test_a_rank_that_runs_other_than_it_published_fails_fast(self):
        """With DP attention, TP rank 1 publishes one row more than its decode
        step runs at the MLP-sync gather (an injected fault): its peer sizes
        the step's collectives by the published count, so rank 1 must refuse
        the step before any collective, and the server shuts down at once."""
        stdout, stderr = (
            os.path.join(os.environ.get("TMPDIR", "/tmp"), f"rdv-{os.getpid()}.{s}")
            for s in ("out", "err")
        )
        with open(stdout, "w") as out, open(stderr, "w") as err:
            process = popen_launch_server(
                _MODEL,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    *_ARGS,
                    "--attn-dp-size",
                    "2",
                    # Prefill graph steps under DP attention keep the direct
                    # path, which has no check; this fault must reach a Step.
                    "--disable-prefill-cuda-graph",
                    # The warmup request decodes; the fault must first hit a
                    # request of this test, after launch.
                    "--skip-server-warmup",
                    "--watchdog-timeout",
                    "600",
                ],
                env={
                    **os.environ,
                    **_NO_CRASH_DIAGNOSTICS,
                    "SGLANG_TEST_RENDEZVOUS_FAULT_TP_RANK": "1",
                },
                return_stdout_stderr=(out, err),
            )
            try:
                pids = [p.pid for p in psutil.Process(process.pid).children(True)]
                # Rank 0 decodes while rank 1 idles: nothing is faulted yet.
                _generate_on(dp_rank=0, timeout=_TEARDOWN_BOUND_S).raise_for_status()
                tic = time.perf_counter()
                try:
                    _generate_on(dp_rank=1, timeout=_FAIL_FAST_BOUND_S)
                except requests.RequestException:
                    pass
                remaining = _FAIL_FAST_BOUND_S - (time.perf_counter() - tic)
                _seconds_until_gone(process, pids, max(remaining, 1))
                seconds = time.perf_counter() - tic
                print(
                    f"[rendezvous mismatch] server gone {seconds:.1f} s after the request"
                )
            finally:
                terminate_and_kill_process_tree(process)
        with open(stderr) as f:
            text = f.read()
        with open(stdout) as f:
            text += f.read()
        self.assertRegex(
            text,
            r"RendezvousMismatch: DP rank 1 published (\d+) input rows at the "
            r"rendezvous but its step has (\d+)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
