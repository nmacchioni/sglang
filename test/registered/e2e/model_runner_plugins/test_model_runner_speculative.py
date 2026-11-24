"""Speculative decoding through the model runner API (TP1).

Each configuration runs the same pre-tokenized workload direct, direct again
(to show the configuration repeats) and through --model-runner sglang, and
requires: identical outputs; identical accepts per verify (accept lengths,
accept indices and the sampled tokens they select); raw logits bitwise equal
at every step matched by its inputs (a verify's inputs are the drafts, so
equal verify inputs are equal drafts); and the committed KV of a request's
accepted tokens equal, read through a follow-up that reuses it as a cached
prefix. The trace proves the path: every target step ran as a Step, verify
steps ran, drafts were accepted beyond the bonus token and rejected, and a
verify followed each rejection. Eager, CUDA graphs and overlap scheduling are
separate configurations. N-gram trees sample through a tree-sampling kernel
that does not build for sm103a in the image this was proven on, so stochastic
rejection is covered by the chain draft's configurations.

EAGLE3 runs its draft model through the API as well: each round's expansion
is one DRAFT Step and each draft forward over the target's tokens a
DRAFT_EXTEND Step fed the target's hidden states.

The reference runner (an independent plugin) verifies n-gram drafts itself:
it repeats across launches, and its accepts reproduce its own non-speculative
greedy decode, token for token.
"""

import os
import tempfile
import unittest
from collections import Counter

import requests
import torch
from transformers import AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.model_runner_reference import (
    REFERENCE_RUNNER,
    install_reference_runner,
    read_accept_trace,
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

register_cuda_ci(est_time=2400, stage="extra-b", runner_config="1-gpu-small")

_MODEL = "Qwen/Qwen3-0.6B"
_COMMON = ["--disable-flashinfer-autotune", "--mem-fraction-static", "0.5"]
_EAGER = [
    "--disable-overlap-schedule",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
]
_GRAPHS = ["--disable-overlap-schedule", "--cuda-graph-max-bs-decode", "4"]
_OVERLAP_GRAPHS = ["--cuda-graph-max-bs-decode", "4"]
_NGRAM = ["--speculative-algorithm", "NGRAM", "--speculative-num-draft-tokens", "8"]
# A chain of 3 drafts (4 verified tokens), and a tree of 6 nodes from top-2
# expansion; the tree runs on the Triton backend, as the image's FlashInfer
# cannot plan this tree verify on the direct path.
_EAGLE3_HEAD = "GavinLucky/SGLang-EAGLE3-Qwen3-0.6B-SpecForge"
_EAGLE3 = [
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    _EAGLE3_HEAD,
    "--speculative-num-steps",
    "3",
]
_CHAIN = [
    *_EAGLE3,
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]
_TREE = [
    *_EAGLE3,
    "--speculative-eagle-topk",
    "2",
    "--speculative-num-draft-tokens",
    "6",
    "--attention-backend",
    "triton",
]
_SAMPLED = {"temperature": 0.8, "top_p": 0.95, "sampling_seed": 7}
_CHAIN_SHAPE = [
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]
# A separate small draft model of the target's family.
_STANDALONE_TARGET = "Qwen/Qwen3-1.7B"
_STANDALONE = [
    "--speculative-algorithm",
    "STANDALONE",
    "--speculative-draft-model-path",
    _MODEL,
    *_CHAIN_SHAPE,
]
# The model's own MTP layer (EAGLE over NEXTN); Qwen3.5 interleaves linear
# attention with full attention, so its recurrent state is committed as well.
_MTP_MODEL = "Qwen/Qwen3.5-0.8B"
_MTP = ["--speculative-algorithm", "NEXTN", *_CHAIN_SHAPE, "--disable-multimodal"]
# DSpark, native-only: blocks of 7 drafts (8 verified tokens) from a DSpark
# head, with the accept fused into SGLang's verify; on Triton, as the image's
# FlashInfer cannot plan this verify on the direct path.
_DSPARK_TARGET = "Qwen/Qwen3-4B-FP8"
_DSPARK = [
    "--trust-remote-code",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    "deepseek-ai/dspark_qwen3_4b_block7",
    "--speculative-dspark-block-size",
    "7",
    "--attention-backend",
    "triton",
]

# name -> (model, server args, sampling)
_CONFIGS = {
    "ngram_eager": (_MODEL, [*_NGRAM, *_EAGER], {}),
    "ngram_graphs": (_MODEL, [*_NGRAM, *_GRAPHS], {}),
    "ngram_overlap_graphs": (_MODEL, [*_NGRAM, *_OVERLAP_GRAPHS], {}),
    "eagle3_eager": (_MODEL, [*_CHAIN, *_EAGER], {}),
    "eagle3_graphs": (_MODEL, [*_CHAIN, *_GRAPHS], {}),
    "eagle3_overlap_graphs": (_MODEL, [*_CHAIN, *_OVERLAP_GRAPHS], {}),
    "eagle3_tree_graphs": (_MODEL, [*_TREE, *_GRAPHS], {}),
    # Stochastic rejection of a sampled chain against the target distribution.
    "eagle3_rejection_sampling": (
        _MODEL,
        [*_CHAIN, *_GRAPHS, "--speculative-use-rejection-sampling"],
        _SAMPLED,
    ),
    "standalone_graphs": (_STANDALONE_TARGET, [*_STANDALONE, *_GRAPHS], {}),
    "standalone_overlap_graphs": (
        _STANDALONE_TARGET,
        [*_STANDALONE, *_OVERLAP_GRAPHS],
        {},
    ),
    "mtp_eager": (_MTP_MODEL, [*_MTP, *_EAGER], {}),
    "mtp_graphs": (_MTP_MODEL, [*_MTP, *_GRAPHS], {}),
    "mtp_overlap_graphs": (_MTP_MODEL, [*_MTP, *_OVERLAP_GRAPHS], {}),
    "dspark_graphs": (_DSPARK_TARGET, [*_DSPARK, *_GRAPHS], {}),
    "dspark_overlap_graphs": (_DSPARK_TARGET, [*_DSPARK, *_OVERLAP_GRAPHS], {}),
}
# Accepted inside DSpark's own verify, past the accept trace: each request's
# accept lengths come from the committed lengths of its consecutive verifies.
_ACCEPT_FROM_STEPS = {"dspark_graphs", "dspark_overlap_graphs"}
# Models whose state SGLang's slot-row digest cannot read (recurrent state):
# their committed state shows in the bitwise logits of every later step.
_NO_KV_DIGEST = {"mtp_eager", "mtp_graphs", "mtp_overlap_graphs"}
# Configurations whose draft model runs as DRAFT and DRAFT_EXTEND Steps.
_DRAFT_MODELS = {
    name for name in _CONFIGS if name.startswith(("eagle3", "standalone", "mtp"))
}

# Repetitive text, so n-gram drafts match; plain text, so some are rejected.
_PROMPTS = [
    "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, "
    "20, 1, 2, 3, 4, 5,",
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + "
    "fib(n - 2)\n\ndef fib2(n):\n",
    "The capital of France is",
    "Write a haiku about the sea:",
]
_SIGNATURE = ("input_ids", "positions", "out_cache_loc", "seq_lens", "req_pool_indices")
_NEW_TOKENS = 48


def _generate(input_ids, sampling):
    response = requests.post(
        DEFAULT_URL_FOR_TEST + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": _NEW_TOKENS,
                "temperature": 0,
                "ignore_eos": True,
                **sampling,
            },
        },
    )
    response.raise_for_status()
    return response.json()


def _output_ids(result):
    return [r["output_ids"] for r in (result if isinstance(result, list) else [result])]


def _serve(model, other_args, workload, env=None):
    trace_dir = tempfile.mkdtemp()
    process = popen_launch_server(
        model,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=with_model_runner(other_args),
        env={
            **os.environ,
            **(env or {}),
            "SGLANG_MODEL_RUNNER_STEP_TRACE_DIR": trace_dir,
            "SGLANG_MODEL_RUNNER_STEP_TRACE_LOGITS": "1",
        },
    )
    try:
        outputs = workload()
    finally:
        terminate_and_kill_process_tree(process)
    return outputs, read_step_trace(trace_dir), read_accept_trace(trace_dir)


def _target_steps(trace):
    return [r for r in trace if r.get("role") != "draft"]


def _by_inputs(trace) -> dict:
    """Target steps with saved logits keyed by their inputs, without keys that
    repeat within the run."""
    by_inputs, repeated = {}, set()
    for record in _target_steps(trace):
        tensors = record.get("tensors", {})
        if "logits" not in tensors:
            continue
        key = (record["mode"],) + tuple(
            tensors[name].numpy().tobytes() for name in _SIGNATURE
        )
        if key in by_inputs:
            repeated.add(key)
        by_inputs[key] = record
    return {k: r for k, r in by_inputs.items() if k not in repeated}


def _accepts_from_steps(trace, num_draft_tokens):
    """Per verify, as accept records: each request's accept length is the
    growth of its committed length by its next verify (a larger growth is a
    new request in the same row)."""
    verifies = [
        r
        for r in _target_steps(trace)
        if r["mode"] == "target_verify" and "tensors" in r
    ]
    last = {}
    records = []
    for record in verifies:
        rows = record["tensors"]["req_pool_indices"].tolist()
        lens = record["tensors"]["seq_lens"].tolist()
        rids, accept_lens = [], []
        for row, n in zip(rows, lens):
            if row in last and 0 < n - last[row] <= num_draft_tokens:
                rids.append(row)
                accept_lens.append(n - last[row])
            last[row] = n
        records.append({"rids": rids, "accept_lens": accept_lens})
    return records


def _accepts(records):
    return [(r["accept_lens"], r["accept_index"], r["predict"]) for r in records]


class _SpeculativeCase(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizers = {}

    def _prompts(self, model):
        if model not in self.tokenizers:
            tokenizer = AutoTokenizer.from_pretrained(model)
            self.tokenizers[model] = [tokenizer.encode(p) for p in _PROMPTS]
        return self.tokenizers[model]

    def _workload(self, model, sampling):
        prompts = self._prompts(model)

        def run():
            first = _generate(prompts[:1], sampling)
            if sampling:
                # Seeded sampling repeats per request, not across batch rows:
                # one request at a time.
                outputs = [first] + [_generate([p], sampling) for p in prompts[1:]]
            else:
                outputs = [
                    first,
                    _generate(prompts[:2], sampling),
                    _generate(prompts, sampling),
                ]
            # Reuses the first request's KV, generated tokens included: what its
            # verifies wrote and SGLang moved to the accepted positions.
            committed = prompts[0] + first[0]["output_ids"][:-1]
            outputs.append(_generate([committed + [11]], sampling))
            return outputs

        return run

    def _check_repeats(self, name, want, got, *, exact=True):
        """``exact``: outputs and accepts repeat. Stochastic rejection draws its
        uniforms from the process RNG, which launch-time health requests
        advance by a varying count, so a sampled run only matches step by
        step until its first different draw."""
        (want_out, want_trace, want_acc), (got_out, got_trace, got_acc) = want, got
        if exact:
            self.assertEqual(
                [_output_ids(o) for o in got_out],
                [_output_ids(o) for o in want_out],
                name,
            )
            self.assertEqual(_accepts(got_acc), _accepts(want_acc), name)
        want_steps, got_steps = _by_inputs(want_trace), _by_inputs(got_trace)
        common = want_steps.keys() & got_steps.keys()
        if exact:
            self.assertGreaterEqual(len(common), 0.9 * len(want_steps), name)
        modes = Counter()
        for key in common:
            a, b = want_steps[key], got_steps[key]
            self.assertTrue(
                torch.equal(a["tensors"]["logits"], b["tensors"]["logits"]),
                f"{name}: step {a['index']} ({a['mode']}) logits differ",
            )
            modes[a["mode"]] += 1
        self.assertGreater(modes["target_verify"], 0, name)
        crcs = 0
        for key in common:
            a, b = want_steps[key], got_steps[key]
            self.assertEqual(a["prefix_kv_crc"], b["prefix_kv_crc"], name)
            crcs += len(a["prefix_kv_crc"] or ())
        if exact and name.split(" ")[0] not in _NO_KV_DIGEST:
            self.assertGreater(crcs, 0, f"{name}: no committed prefix was compared")
        return modes

    def _check_speculation(self, name, accepts, num_draft_tokens):
        """Drafts were accepted beyond the bonus token and rejected, and the
        request decoded on after each rejection."""
        lens = Counter(n for r in accepts for n in r["accept_lens"])
        self.assertGreater(sum(c for n, c in lens.items() if n > 1), 0, name)
        rejected = [
            (i, rid)
            for i, r in enumerate(accepts)
            for rid, n in zip(r["rids"], r["accept_lens"])
            if n < num_draft_tokens
        ]
        self.assertTrue(rejected, name)
        followed = sum(
            any(rid in r["rids"] for r in accepts[i + 1 :]) for i, rid in rejected
        )
        self.assertGreater(followed, 0, name)
        return lens

    def _check_draft_steps(self, name, trace):
        """The draft model ran as Steps: one DRAFT (the whole expansion) per
        verify, and DRAFT_EXTEND after each target extend and verify."""
        drafts = Counter(r["mode"] for r in trace if r.get("role") == "draft")
        target = Counter(r["mode"] for r in _target_steps(trace))
        self.assertEqual(drafts["draft"], target["target_verify"], name)
        self.assertEqual(
            drafts["draft_extend"], target["target_verify"] + target["extend"], name
        )
        return drafts

    def _check_config(self, name, *, num_draft_tokens=8):
        model, args, sampling = _CONFIGS[name]
        workload = self._workload(model, sampling)
        direct = _serve(model, args, workload)
        again = _serve(model, args, workload)
        steps = _serve(model, [*args, "--model-runner", "sglang"], workload)

        self.assertEqual({r["path"] for r in direct[1] + again[1]}, {"direct"})
        self.assertEqual({r["path"] for r in steps[1]}, {"step"})
        exact = not sampling
        self._check_repeats(f"{name} direct/direct", direct, again, exact=exact)
        modes = self._check_repeats(f"{name} direct/steps", direct, steps, exact=exact)
        accepts = (
            _accepts_from_steps(steps[1], num_draft_tokens)
            if name in _ACCEPT_FROM_STEPS
            else steps[2]
        )
        lens = self._check_speculation(name, accepts, num_draft_tokens)
        if name in _DRAFT_MODELS:
            drafts = self._check_draft_steps(name, steps[1])
            replayed = sum(
                r["replayed_graph"] for r in steps[1] if r.get("role") == "draft"
            )
            print(f"[{name}] draft steps {dict(drafts)}, {replayed} replayed graphs")
        print(
            f"[{name}] steps matched bitwise per mode {dict(modes)}; "
            f"{len(accepts)} verifies, accept lengths {dict(sorted(lens.items()))}"
        )
        return direct, steps


class TestNgramSteps(_SpeculativeCase):
    def test_eager(self):
        self._check_config("ngram_eager")

    def test_graphs(self):
        self._check_config("ngram_graphs")

    def test_overlap_graphs(self):
        self._check_config("ngram_overlap_graphs")


class TestEagle3Steps(_SpeculativeCase):
    def test_eager(self):
        self._check_config("eagle3_eager", num_draft_tokens=4)

    def test_graphs(self):
        self._check_config("eagle3_graphs", num_draft_tokens=4)

    def test_overlap_graphs(self):
        self._check_config("eagle3_overlap_graphs", num_draft_tokens=4)

    def test_tree(self):
        # A path of the tree verifies at most num_steps drafts plus the bonus.
        self._check_config("eagle3_tree_graphs", num_draft_tokens=4)

    def test_rejection_sampling(self):
        self._check_config("eagle3_rejection_sampling", num_draft_tokens=4)


class TestStandaloneAndMTPSteps(_SpeculativeCase):
    def test_standalone_graphs(self):
        self._check_config("standalone_graphs", num_draft_tokens=4)

    def test_standalone_overlap_graphs(self):
        self._check_config("standalone_overlap_graphs", num_draft_tokens=4)

    def test_mtp_eager(self):
        self._check_config("mtp_eager", num_draft_tokens=4)

    def test_mtp_graphs(self):
        self._check_config("mtp_graphs", num_draft_tokens=4)

    def test_mtp_overlap_graphs(self):
        self._check_config("mtp_overlap_graphs", num_draft_tokens=4)


class TestNativeOnlySteps(_SpeculativeCase):
    """Native-only algorithms: the target's steps run through the API, the
    accept fused into the native verify; the draft stays inside its worker."""

    def test_dspark_graphs(self):
        self._check_config("dspark_graphs")

    def test_dspark_overlap_graphs(self):
        self._check_config("dspark_overlap_graphs")


class TestReferenceRunnerVerify(_SpeculativeCase):
    def test_reference_runner_verifies_ngram_drafts(self):
        env = install_reference_runner(tempfile.mkdtemp())
        args = [*_EAGER, *_COMMON, "--model-runner", REFERENCE_RUNNER]
        workload = self._workload(_MODEL, {})
        spec = _serve(_MODEL, [*_NGRAM, *args], workload, env=env)
        again = _serve(_MODEL, [*_NGRAM, *args], workload, env=env)
        plain = _serve(_MODEL, args, workload, env=env)

        self.assertEqual(
            {(r["path"], r["runner"]) for r in spec[1]}, {("step", REFERENCE_RUNNER)}
        )
        self.assertIn("target_verify", {r["mode"] for r in spec[1]})
        self._check_repeats("reference ngram repeat", spec, again)
        lens = self._check_speculation("reference ngram", spec[2], 8)
        # Greedy verification reproduces the runner's own greedy decode where
        # no near-tie flips: a verify scores D rows where decode scores one, so
        # bf16 rounding differs, as on every runner.
        want = [_output_ids(o) for o in plain[0]]
        got = [_output_ids(o) for o in spec[0]]
        self.assertEqual(got[:2], want[:2])
        agree = sum(
            a == b
            for w, g in zip(want, got)
            for wr, gr in zip(w, g)
            for a, b in zip(wr, gr)
        )
        total = sum(len(wr) for w in want for wr in w)
        print(
            f"[reference ngram] {len(spec[2])} verifies, accept lengths "
            f"{dict(sorted(lens.items()))}; greedy tokens equal to its plain "
            f"decode {agree}/{total}"
        )


for _args in _CONFIGS.values():
    _args[1].extend(_COMMON)


if __name__ == "__main__":
    unittest.main()
