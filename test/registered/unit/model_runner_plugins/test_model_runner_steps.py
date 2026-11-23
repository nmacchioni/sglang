"""Model runner steps: Step as a view of the step's ForwardBatch, its checks,
which batches take the API, prompt logprobs through project(), release order,
and what this release lets an external runner run."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_runner_plugins.api import (
    AdmissionLimits,
    Capacity,
    GraphBuckets,
    KVLayout,
    KVTables,
    LoadContext,
    LoadResult,
    Logits,
    MemoryNeeds,
    ModelInfo,
    OutputSpec,
    ResolvedConfig,
    Rows,
    RunnerCapabilities,
    RuntimeConfig,
    Step,
    StepMode,
    StepOutput,
)
from sglang.srt.model_runner_plugins.external import ExternalRunner
from sglang.srt.model_runner_plugins.native_steps import NativeStepLowering
from sglang.srt.model_runner_plugins.refusal import launch_refusals
from sglang.srt.model_runner_plugins.selection import capabilities_in_this_release
from sglang.srt.model_runner_plugins.step_forward import StepForward
from sglang.srt.model_runner_plugins.steps import build_step, step_refusal

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_VOCAB = 11
_PAGE_TABLE = torch.zeros(4, 64, dtype=torch.int32)


def _extend_batch(*, prefix_lens, extend_lens, logprob_starts=None):
    """A two-request extend batch as ForwardBatch.init_new lays one out."""
    num_tokens = sum(extend_lens)
    seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
    positions = torch.cat(
        [torch.arange(p, p + e) for p, e in zip(prefix_lens, extend_lens)]
    )
    starts = torch.tensor([0, *torch.tensor(extend_lens).cumsum(0)[:-1].tolist()])
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(extend_lens),
        input_ids=torch.arange(num_tokens) % _VOCAB,
        req_pool_indices=torch.arange(1, len(extend_lens) + 1),
        seq_lens=torch.tensor(seq_lens),
        seq_lens_cpu=torch.tensor(seq_lens),
        out_cache_loc=torch.arange(100, 100 + num_tokens),
        seq_lens_sum=sum(seq_lens),
        positions=positions,
        extend_seq_lens=torch.tensor(extend_lens, dtype=torch.int32),
        extend_prefix_lens=torch.tensor(prefix_lens, dtype=torch.int32),
        extend_seq_lens_cpu=list(extend_lens),
        extend_prefix_lens_cpu=list(prefix_lens),
        extend_start_loc=starts,
        extend_num_tokens=num_tokens,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )
    if logprob_starts is not None:
        batch.return_logprob = True
        batch.extend_logprob_start_lens_cpu = list(logprob_starts)
        batch.top_logprobs_nums = [0] * len(extend_lens)
        batch.token_ids_logprobs = [None] * len(extend_lens)
        # The token each logprob row scores: here, row t scores (t + 1) % vocab.
        batch.extend_input_logprob_token_ids_gpu = torch.cat(
            [
                (torch.arange(s + lp, s + e) + 1) % _VOCAB
                for s, e, lp in zip(starts.tolist(), extend_lens, logprob_starts)
            ]
        )
    return batch


def _decode_batch():
    return ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=2,
        input_ids=torch.tensor([3, 4]),
        req_pool_indices=torch.tensor([1, 2]),
        seq_lens=torch.tensor([5, 9]),
        seq_lens_cpu=torch.tensor([5, 9]),
        out_cache_loc=torch.tensor([7, 8]),
        seq_lens_sum=14,
        positions=torch.tensor([4, 8]),
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


class TestStepView(CustomTestCase):
    def test_step_views_the_forward_batch_tensors(self):
        """A copy anywhere would break the zero-copy step path, and with it
        native plan()'s check that the Step is a view of its batch."""
        extend = _extend_batch(prefix_lens=[0, 3], extend_lens=[4, 2])
        step = build_step(forward_batch=extend, page_table=_PAGE_TABLE)
        self.assertIs(step.mode, StepMode.EXTEND)
        for got, want in (
            (step.input_ids, extend.input_ids),
            (step.positions, extend.positions),
            (step.kv.write_slots, extend.out_cache_loc),
            (step.kv.page_table, _PAGE_TABLE),
            (step.rows.req_pool_indices, extend.req_pool_indices),
            (step.rows.seq_lens, extend.seq_lens),
            (step.rows.query_start, extend.extend_start_loc),
            (step.rows.query_lens, extend.extend_seq_lens),
            (step.rows.prefix_lens, extend.extend_prefix_lens),
        ):
            self.assertIs(got, want)

        decode = build_step(forward_batch=_decode_batch(), page_table=_PAGE_TABLE)
        self.assertIs(decode.mode, StepMode.DECODE)
        self.assertIsNone(decode.rows.query_start)
        self.assertIsNone(decode.rows.prefix_lens_cpu)

    def test_prompt_logprobs_are_requested_only_when_an_input_row_reads_them(self):
        for starts, expected in (([0, 2], True), ([4, 2], False), (None, False)):
            with self.subTest(starts=starts):
                batch = _extend_batch(
                    prefix_lens=[0, 3], extend_lens=[4, 2], logprob_starts=starts
                )
                step = build_step(forward_batch=batch, page_table=_PAGE_TABLE)
                self.assertEqual(step.outputs.prompt_logprobs, expected)

    def test_step_shape_checks(self):
        def step(mode, *, tokens=2, slots=2, positions=2, batch=2, queries=False):
            rows = Rows(
                req_pool_indices=torch.zeros(batch, dtype=torch.int64),
                seq_lens=torch.ones(batch, dtype=torch.int64),
                query_start=torch.zeros(batch) if queries else None,
            )
            return Step(
                mode=mode,
                input_ids=torch.zeros(tokens, dtype=torch.int64),
                positions=torch.zeros(positions, dtype=torch.int64),
                rows=rows,
                kv=KVTables(write_slots=torch.zeros(slots), page_table=_PAGE_TABLE),
                outputs=OutputSpec(),
            )

        step(StepMode.DECODE)
        step(StepMode.EXTEND, tokens=5, slots=5, positions=5, queries=True)
        for build, error in (
            (lambda: step(StepMode.DECODE, positions=3), "positions"),
            (lambda: step(StepMode.DECODE, slots=1), "write slots"),
            (lambda: step(StepMode.EXTEND), "query offsets"),
            (lambda: step(StepMode.MIXED), "query offsets"),
            (lambda: step(StepMode.DECODE, queries=True), "one input row"),
            (lambda: step(StepMode.DECODE, tokens=3, slots=3, positions=3), "rows"),
        ):
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    build()

    def test_native_plan_takes_back_only_the_bound_batch(self):
        lowering = NativeStepLowering()
        batch = _extend_batch(prefix_lens=[0, 3], extend_lens=[4, 2])
        step = build_step(forward_batch=batch, page_table=_PAGE_TABLE)
        with lowering.bind(step=step, forward_batch=batch):
            self.assertIs(lowering.forward_batch_of(step), batch)
            other = build_step(forward_batch=batch, page_table=_PAGE_TABLE)
            with self.assertRaisesRegex(RuntimeError, "plans only the step"):
                lowering.forward_batch_of(other)
        with self.assertRaisesRegex(RuntimeError, "plans only the step"):
            lowering.forward_batch_of(step)
        copied = _extend_batch(prefix_lens=[0, 3], extend_lens=[4, 2])
        with lowering.bind(step=step, forward_batch=copied):
            with self.assertRaisesRegex(RuntimeError, "does not view"):
                lowering.forward_batch_of(step)


class TestStepSubset(CustomTestCase):
    def test_every_batch_feature_outside_block_3_is_refused(self):
        """A batch the API cannot express must take the direct path (native)
        or fail (external), never run with the feature silently dropped."""
        self.assertIsNone(step_refusal(_decode_batch()))
        cases = {
            "forward mode": dict(forward_mode=ForwardMode.TARGET_VERIFY),
            "hidden-state capture": dict(capture_hidden_mode=CaptureHiddenMode.LAST),
            "input embeddings": dict(input_embeds=torch.zeros(2, 4)),
            "LoRA": dict(lora_ids=[None, "adapter"]),
            "beam search": dict(beam=object()),
            "position scoring": dict(token_indices_to_pool=[torch.zeros(1)]),
            "input candidate selectors": dict(input_candidate_selectors=[None, 1]),
            "MLP-sync": dict(global_num_tokens_cpu=[2]),
        }
        for reason, fields in cases.items():
            with self.subTest(reason=reason):
                batch = _decode_batch()
                for name, value in fields.items():
                    setattr(batch, name, value)
                self.assertIn(reason, step_refusal(batch))


class _FakeExecutor:
    """Hidden state of input row t is one-hot t; the LM head is a fixed matrix."""

    def __init__(self):
        self.lm_head = torch.randn(
            64, _VOCAB, generator=torch.Generator().manual_seed(0)
        )
        self.calls = []
        self.project_rows = []

    def plan(self, step):
        self.calls.append(("plan", step.mode))
        return SimpleNamespace(step=step)

    def execute(self, plan):
        self.calls.append(("execute", plan.step.mode))
        rows = plan.step.rows
        last = (
            torch.arange(rows.batch_size)
            if rows.query_start is None
            else rows.query_start + rows.query_lens - 1
        )
        return StepOutput(logits=Logits(tensor=self.lm_head[last]))

    def project(self, req):
        self.project_rows.append(req.rows.tolist())
        return self.lm_head[req.rows]

    def release(self, out, after):
        self.calls.append(("release", out))


def _external_forward(executor):
    return StepForward(
        runner_name="fake",
        executor=executor,
        page_table=_PAGE_TABLE,
        vocab_size=_VOCAB,
        native_prompt_logprobs=False,
    )


class TestStepForward(CustomTestCase):
    def test_prompt_logprobs_through_project_match_the_logits(self):
        executor = _FakeExecutor()
        batch = _extend_batch(
            prefix_lens=[0, 3], extend_lens=[4, 3], logprob_starts=[1, 0]
        )

        out = _external_forward(executor).forward(batch).logits_output

        # Native row selection: request 0 reads rows 1..3, request 1 rows 4..6.
        self.assertEqual(sorted(sum(executor.project_rows, [])), [1, 2, 3, 4, 5, 6])
        rows = torch.tensor([1, 2, 3, 4, 5, 6])
        logprobs = torch.log_softmax(executor.lm_head[rows].float(), dim=-1)
        expected = logprobs[torch.arange(len(rows)), (rows + 1) % _VOCAB]
        torch.testing.assert_close(out.input_token_logprobs.float(), expected)
        torch.testing.assert_close(out.next_token_logits, executor.lm_head[[3, 6]])

    def test_each_output_is_released_when_the_next_step_starts_and_at_close(self):
        executor = _FakeExecutor()
        steps = _external_forward(executor)
        steps.forward(_decode_batch())
        steps.forward(_decode_batch())
        steps.close()
        steps.close()
        names = [name for name, _ in executor.calls]
        self.assertEqual(
            names,
            ["plan", "execute", "release", "plan", "execute", "release"],
        )

    def test_a_batch_outside_the_subset_fails_for_external_runners_only(self):
        batch = _decode_batch()
        batch.capture_hidden_mode = CaptureHiddenMode.LAST
        with self.assertRaisesRegex(ValueError, "fake cannot run a step with hidden"):
            _external_forward(_FakeExecutor()).takes(batch)
        native = StepForward(
            runner_name="sglang",
            executor=_FakeExecutor(),
            page_table=_PAGE_TABLE,
            vocab_size=_VOCAB,
            native_prompt_logprobs=True,
            direct_fallback=True,
        )
        self.assertFalse(native.takes(batch))
        self.assertTrue(native.takes(_decode_batch()))

    def test_execute_must_return_a_step_output(self):
        executor = _FakeExecutor()
        executor.execute = lambda plan: {"logits": None}
        with self.assertRaisesRegex(TypeError, "expected StepOutput"):
            _external_forward(executor).forward(_decode_batch())


class TestExternalRelease(CustomTestCase):
    def test_external_runners_keep_only_block_3_features(self):
        """Whatever an external runner declares, launches outside TP1 eager
        text with overlap off are refused until their block lands."""
        declared = RunnerCapabilities.all_supported()
        external = capabilities_in_this_release(declared, native=False)
        self.assertIs(capabilities_in_this_release(declared, native=True), declared)
        reasons = launch_refusals(
            external,
            ResolvedConfig(
                model_path="m",
                tp_size=2,
                cuda_graph_phases=("decode",),
                overlap_schedule=True,
                advanced_kv=("sliding_window",),
                unified_memory=True,
            ),
        )
        self.assertEqual(len(reasons), 5, reasons)
        self.assertTrue(external.prompt_logprobs and external.top_logprobs)
        self.assertEqual(launch_refusals(external, ResolvedConfig(model_path="m")), [])

    def test_architecture_list_refuses_other_models(self):
        caps = RunnerCapabilities(architectures=("Qwen3ForCausalLM",))
        cfg = ResolvedConfig(model_path="m", architectures=("GptOssForCausalLM",))
        self.assertEqual(
            launch_refusals(caps, cfg),
            ["model architecture GptOssForCausalLM (serves: Qwen3ForCausalLM)"],
        )
        cfg = ResolvedConfig(model_path="m", architectures=("Qwen3ForCausalLM",))
        self.assertEqual(launch_refusals(caps, cfg), [])


_LIMITS = AdmissionLimits(
    max_total_num_tokens=4096,
    max_running_requests=8,
    max_req_len=2047,
    max_req_input_len=2042,
)


class _Plugin:
    def __init__(self, *, final_slots=4096, page_size=1):
        self.calls = []
        self.final_slots = final_slots
        self.page_size = page_size

    def load(self, ctx):
        self.calls.append("load")
        return LoadResult(
            model_info=ModelInfo(context_len=2048, vocab_size=_VOCAB),
            memory=MemoryNeeds(weight_bytes=1),
        )

    def allocate(self, cfg):
        self.calls.append("allocate")
        return KVLayout(page_size=self.page_size, num_slots=4096)

    def capture(self, buckets):
        self.calls.append("capture")

    def finalize(self, capacity):
        self.calls.append("finalize")
        return KVLayout(page_size=self.page_size, num_slots=self.final_slots)

    def close(self):
        self.calls.append("close")


class _Host:
    def __init__(self, calls):
        self.calls = calls
        info = (4096, 0, 8, None, 2047, 2042, 0, "cuda", None, 9, 2048, 4096)
        self.tp_worker = SimpleNamespace(get_worker_info=lambda: info)
        self.draft_worker = None

    def load_model_workers(self):
        self.calls.append("host.load")

    def alloc_model_memory_pools(self):
        self.calls.append("host.allocate")

    def capture_model_graphs(self):
        self.calls.append("host.capture")

    def finalize_model_capacity(self):
        self.calls.append("host.finalize")


class TestExternalStartup(CustomTestCase):
    def _run(self, plugin):
        runner = ExternalRunner(name="fake", plugin=plugin, host=_Host(plugin.calls))
        runner.load(LoadContext(model_path="m", device="cuda", gpu_id=0))
        runner.allocate(RuntimeConfig(page_size=1))
        runner.capture(GraphBuckets())
        return runner, runner.finalize(Capacity())

    def test_plugin_phase_precedes_the_scheduler_phase_built_on_it(self):
        plugin = _Plugin()
        runner, final = self._run(plugin)
        self.assertEqual(
            plugin.calls,
            [
                "load",
                "host.load",
                "allocate",
                "host.allocate",
                "capture",
                "host.capture",
                "finalize",
                "host.finalize",
            ],
        )
        # The scheduler's allocators saw the plugin's slots before they existed.
        self.assertEqual(runner.layout.num_slots, 4096)
        # Admission limits are SGLang's, from the pools it sized.
        self.assertEqual(final.limits, _LIMITS)

    def test_a_resize_or_a_foreign_page_size_is_refused(self):
        with self.assertRaisesRegex(ValueError, "keep their allocated size"):
            self._run(_Plugin(final_slots=2048))
        with self.assertRaisesRegex(ValueError, "page size 16"):
            self._run(_Plugin(page_size=16))


if __name__ == "__main__":
    unittest.main()
