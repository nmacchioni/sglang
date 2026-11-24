"""Model runner steps: Step as a view of the step's ForwardBatch, its checks,
which batches take the API, prompt logprobs through project(), output
lifetimes (release order and shared buffers), graph bucket choice, the
rendezvous check under DP attention, pipeline stage outputs, vocabulary-sharded
logits, and what this release lets an external runner run."""

import contextlib
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.model_runner_plugins.api import (
    AdmissionLimits,
    Capacity,
    FullAttentionLayer,
    GraphBuckets,
    GraphChoice,
    GraphPhase,
    KVBuffer,
    KVKind,
    KVLayout,
    KVSubPool,
    KVTables,
    LoadContext,
    LoadResult,
    LogitRows,
    Logits,
    LogitsLayout,
    MemoryNeeds,
    ModelInfo,
    OutputBuffer,
    OutputSpec,
    PromptLogprobs,
    RendezvousMismatch,
    ResolvedConfig,
    Rows,
    RunnerCapabilities,
    RuntimeConfig,
    StateTables,
    Step,
    StepMode,
    StepOutput,
    WindowTables,
)
from sglang.srt.model_runner_plugins.external import ExternalRunner
from sglang.srt.model_runner_plugins.graphs import bucket_graph_choice
from sglang.srt.model_runner_plugins.lifetimes import OutputLifetimes
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
        def step(
            mode,
            *,
            tokens=2,
            slots=2,
            positions=2,
            batch=2,
            queries=False,
            graph=GraphChoice(),
            window_slots=None,
        ):
            window = (
                None
                if window_slots is None
                else WindowTables(
                    write_slots=torch.zeros(window_slots), slot_map=_PAGE_TABLE[0]
                )
            )
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
                kv=KVTables(
                    write_slots=torch.zeros(slots),
                    page_table=_PAGE_TABLE,
                    window=window,
                ),
                outputs=OutputSpec(),
                graph=graph,
            )

        step(StepMode.DECODE)
        step(StepMode.DECODE, window_slots=2)
        step(StepMode.DECODE, graph=GraphChoice(phase=GraphPhase.DECODE, bucket=2))
        step(StepMode.EXTEND, tokens=5, slots=5, positions=5, queries=True)
        for build, error in (
            (lambda: step(StepMode.DECODE, positions=3), "positions"),
            (lambda: step(StepMode.DECODE, slots=1), "write slots"),
            (lambda: step(StepMode.DECODE, window_slots=1), "window write slots"),
            (
                lambda: StateTables(
                    req_to_state=torch.zeros(2), copy_src=torch.zeros(1)
                ),
                "sources and destinations",
            ),
            (lambda: step(StepMode.EXTEND), "query offsets"),
            (lambda: step(StepMode.MIXED), "query offsets"),
            (lambda: step(StepMode.DECODE, queries=True), "one input row"),
            (lambda: step(StepMode.DECODE, tokens=3, slots=3, positions=3), "rows"),
            (
                lambda: step(
                    StepMode.DECODE,
                    graph=GraphChoice(phase=GraphPhase.DECODE, bucket=1),
                ),
                "cannot replay the decode graph of 1",
            ),
            (lambda: GraphChoice(bucket=4), "a phase and a bucket"),
            (lambda: GraphBuckets(decode=(4, 2)), "strictly ascending"),
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


def _dp_decode_batch(*, published, votes=(True, False)):
    """The decode batch on DP rank 1 of two, after the MLP-sync gather."""
    batch = _decode_batch()
    batch.global_num_tokens_cpu = list(published)
    batch.original_global_num_tokens_cpu = list(published)
    batch.global_num_tokens_for_logprob_cpu = list(published)
    batch.can_run_decode_cuda_graph, batch.can_run_dp_prefill_cuda_graph = votes
    return batch


@contextlib.contextmanager
def _fake_runtime(prefill_buckets, *, attn_tp_size=1):
    """What dp_prefill_graph_bucket reads besides the batch."""
    from unittest.mock import patch

    graph = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(bs=prefill_buckets))
    )
    with contextlib.ExitStack() as stack:
        for target, value in (
            ("sglang.srt.runtime_context.get_exec", SimpleNamespace(graph=graph)),
            (
                "sglang.srt.runtime_context.get_parallel",
                SimpleNamespace(attn_tp_size=attn_tp_size),
            ),
            ("sglang.srt.layers.cp.base.get_cp_strategy", None),
            (
                "sglang.srt.model_executor.forward_batch_info."
                "prefill_graph_tolerates_sum_len",
                False,
            ),
            (
                "sglang.srt.model_executor.forward_batch_info."
                "_mega_moe_materializes_idle_rank",
                False,
            ),
        ):
            stack.enter_context(patch(target, lambda *a, v=value: v))
        yield


class TestRendezvous(CustomTestCase):
    """Under DP attention a rank's peers size the step's collectives from what
    the rank published at the MLP-sync gather; a step that disagrees with it
    must be refused before any collective, on the rank that diverged."""

    def _step(self, batch, graph=GraphChoice()):
        from unittest.mock import patch

        with patch("sglang.srt.layers.dp_attention.dp_gather_width", lambda: 2):
            with patch("sglang.srt.layers.dp_attention.dp_gather_slot", lambda: 1):
                return build_step(
                    forward_batch=batch, page_table=_PAGE_TABLE, graph=graph
                )

    def test_step_carries_the_gathered_decisions(self):
        decode_graph = GraphChoice(phase=GraphPhase.DECODE, bucket=4)
        step = self._step(_dp_decode_batch(published=[3, 2]), decode_graph)
        self.assertEqual(step.dp.rank, 1)
        self.assertEqual(step.dp.global_num_tokens, (3, 2))
        self.assertTrue(step.dp.decode_graph)
        self.assertFalse(step.dp.prefill_graph)
        self.assertIsNone(step.dp.tbo_split_seq_index)
        # An idle peer, and no rendezvous at all, are fine too.
        self._step(_dp_decode_batch(published=[0, 2]))
        self.assertIsNone(self._step(_decode_batch()).dp)

    def test_step_carries_the_two_batch_overlap_split(self):
        batch = _dp_decode_batch(published=[3, 2])
        batch.tbo_split_seq_index = 1
        self.assertEqual(self._step(batch).dp.tbo_split_seq_index, 1)
        batch.tbo_split_seq_index = 3
        with self.assertRaisesRegex(ValueError, "split at request 3 of a step with 2"):
            self._step(batch)

    def test_a_decode_step_replays_a_prefill_graph_only_presented_as_extend(self):
        """A decode rank pads into one-token extends when a peer extends and the
        rendezvous voted prefill graphs; otherwise a prefill graph on a decode
        step is a lowering bug."""
        prefill = GraphChoice(phase=GraphPhase.PREFILL, bucket=8)
        batch = _dp_decode_batch(published=[5, 2], votes=(False, True))
        batch.is_extend_in_batch = True
        self.assertEqual(self._step(batch, prefill).graph, prefill)
        batch.is_extend_in_batch = False
        with self.assertRaisesRegex(ValueError, "presented as an extend"):
            self._step(batch, prefill)

    def test_a_step_unlike_its_published_rows_or_votes_is_refused(self):
        for batch, graph, error in (
            (_dp_decode_batch(published=[2, 3]), GraphChoice(), "published 3"),
            (
                _dp_decode_batch(published=[3, 2], votes=(False, False)),
                GraphChoice(phase=GraphPhase.DECODE, bucket=4),
                "voted the step eager",
            ),
            (
                _dp_decode_batch(published=[3, 2]),
                GraphChoice(phase=GraphPhase.DECODE, bucket=2),
                "below the largest rank's 3",
            ),
        ):
            with self.subTest(error=error):
                with self.assertRaisesRegex(RendezvousMismatch, error):
                    self._step(batch, graph)


class TestPipelineStages(CustomTestCase):
    def _forward(self, out, *, pp_in=None):
        executor = _FakeExecutor()
        steps = []
        executor.plan = lambda step: steps.append(step) or SimpleNamespace(step=step)
        executor.execute = lambda plan: out
        forward = StepForward(
            runner_name="fake",
            executor=executor,
            page_table=_PAGE_TABLE,
            vocab_size=_VOCAB,
            native_prompt_logprobs=False,
            last_stage=False,
        )
        proxy = None if pp_in is None else PPProxyTensors(pp_in)
        return forward.forward(_decode_batch(), proxy), steps[0]

    def test_an_earlier_stage_returns_pp_out_and_no_logits(self):
        hidden = {"hidden_states": torch.ones(2, 4), "residual": torch.zeros(2, 4)}
        received = {"hidden_states": torch.zeros(2, 4)}
        result, step = self._forward(StepOutput(pp_out=hidden), pp_in=received)
        self.assertIs(step.pp_in, received)
        self.assertTrue(step.outputs.pp_out)
        self.assertIs(step.outputs.logit_rows, LogitRows.NONE)
        self.assertIsInstance(result.logits_output, PPProxyTensors)
        self.assertIs(result.logits_output.tensors, hidden)
        with self.assertRaisesRegex(ValueError, "no pp_out"):
            self._forward(StepOutput(logits=Logits(tensor=torch.zeros(2, _VOCAB))))


class TestVocabShardedLogits(CustomTestCase):
    def test_sglang_gathers_shards_and_drops_the_vocabulary_padding(self):
        """Rank r holds columns [r * shard, (r + 1) * shard) of a vocabulary
        padded to tp_size * shard; SGLang's gather must return exactly the
        vocabulary, in order, for every rank."""
        full = torch.arange(2 * 12, dtype=torch.float32).view(2, 12)
        shards = [full[:, :6], full[:, 6:]]

        class _Group:
            world_size = 2

            def all_gather(self, shard, dim):
                return torch.cat(shards, dim=dim)

        executor = _FakeExecutor()
        executor.execute = lambda plan: StepOutput(
            logits=Logits(tensor=shards[1], layout=LogitsLayout.VOCAB_SHARDED)
        )
        forward = StepForward(
            runner_name="fake",
            executor=executor,
            page_table=_PAGE_TABLE,
            vocab_size=_VOCAB,
            native_prompt_logprobs=False,
            tp_group=_Group(),
        )
        logits = forward.forward(_decode_batch()).logits_output.next_token_logits
        torch.testing.assert_close(logits, full[:, :_VOCAB])
        self.assertTrue(logits.is_contiguous())


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
            "decode graph": dict(input_ids_from_graph=True),
        }
        for reason, fields in cases.items():
            with self.subTest(reason=reason):
                batch = _decode_batch()
                for name, value in fields.items():
                    setattr(batch, name, value)
                self.assertIn(reason, step_refusal(batch))

    def test_dp_prefill_graph_steps_take_the_api_when_every_rank_shares_a_bucket(self):
        """The forward pads every DP rank of such a step to one captured prefill
        bucket; the Step names that bucket on every rank, idle and decode ranks
        included. Without a shared bucket (the ranks' tokens exceed the largest)
        each rank would pick its own graph after padding: direct path."""
        from sglang.srt.model_runner_plugins import steps

        def batch(published, buckets=(4, 8, 16)):
            b = _decode_batch()
            b.global_num_tokens_cpu = published
            b.global_num_tokens_for_logprob_cpu = published
            b.is_extend_in_batch = True
            b.can_run_dp_prefill_cuda_graph = True
            b.tbo_split_seq_index = 1
            return b, _fake_runtime(buckets)

        for published, buckets, bucket in (
            ([2, 5], (4, 8, 16), 8),
            ([2, 0], (4, 8, 16), 4),
            ([2, 17], (4, 8, 16), None),
            ([2, 5], (), None),
        ):
            with self.subTest(published=published, buckets=buckets):
                b, runtime = batch(published, buckets)
                with runtime:
                    self.assertEqual(steps.dp_prefill_graph_bucket(b), bucket)
                    refusal = step_refusal(b)
                if bucket is None:
                    self.assertIn("without a shared bucket", refusal)
                else:
                    # Two-batch overlap steps take the API as well.
                    self.assertIsNone(refusal)


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


def _logits_out(tag):
    return StepOutput(logits=Logits(tensor=torch.full((1, _VOCAB), float(tag))))


class _Releases:
    def __init__(self):
        self.calls = []

    def __call__(self, out, after):
        self.calls.append((int(out.logits.tensor[0, 0]) if out.logits else None, after))


class TestOutputLifetimes(CustomTestCase):
    def _lifetimes(self, *, deferred=True, shared=(OutputBuffer.LOGITS,)):
        releases = _Releases()
        return releases, OutputLifetimes(
            runner_name="fake",
            release=releases,
            shared=frozenset(shared),
            deferred=deferred,
        )

    def test_overlap_releases_in_production_order_with_each_consumer_event(self):
        """A step consumed before an earlier one waits for it: a runner told
        step n+1 is free while step n is not could reuse step n's buffer."""
        releases, lifetimes = self._lifetimes(shared=())
        leases = [lifetimes.produced(_logits_out(i)) for i in range(3)]
        lifetimes.step_started()  # overlap: the next step starting frees nothing
        self.assertEqual(releases.calls, [])
        lifetimes.consumers_enqueued(leases[1], after="event-1")
        self.assertEqual(releases.calls, [])
        lifetimes.consumers_enqueued(leases[0], after="event-0")
        self.assertEqual(releases.calls, [(0, "event-0"), (1, "event-1")])
        with self.assertRaisesRegex(RuntimeError, "consumed twice"):
            lifetimes.consumers_enqueued(leases[0], after=None)
        lifetimes.close()
        self.assertEqual(releases.calls[-1], (2, None))
        self.assertEqual(lifetimes.pending, 0)

    def test_without_overlap_the_next_step_releases_with_no_event(self):
        releases, lifetimes = self._lifetimes(deferred=False)
        lifetimes.produced(_logits_out(0))
        lifetimes.step_started()
        lifetimes.before_execute()
        self.assertEqual(releases.calls, [(0, None)])

    def test_a_shared_buffer_still_held_blocks_the_next_execute(self):
        """The race of a graph replay rewriting logits its consumers have not
        been told about fails loudly instead of corrupting a sample."""
        _, lifetimes = self._lifetimes()
        lease = lifetimes.produced(_logits_out(0))
        with self.assertRaisesRegex(RuntimeError, "step 0's logits is not released"):
            lifetimes.before_execute()
        lifetimes.consumers_enqueued(lease, after="event")
        lifetimes.before_execute()

        # Outputs in buffers no later step writes may stay in flight.
        prompt_only = StepOutput(logprobs=PromptLogprobs(token_logprobs=torch.zeros(1)))
        lifetimes.produced(prompt_only)
        lifetimes.before_execute()
        _, unshared = self._lifetimes(shared=())
        unshared.produced(_logits_out(0))
        unshared.before_execute()


class TestGraphChoice(CustomTestCase):
    def test_smallest_bucket_that_holds_the_step(self):
        buckets = GraphBuckets(decode=(1, 2, 4, 8), prefill=(16, 64))
        for mode, bs, tokens, expected in (
            (StepMode.DECODE, 1, 1, (GraphPhase.DECODE, 1)),
            (StepMode.DECODE, 3, 3, (GraphPhase.DECODE, 4)),
            (StepMode.DECODE, 8, 8, (GraphPhase.DECODE, 8)),
            (StepMode.DECODE, 9, 9, None),
            # Prefill buckets count tokens, not requests.
            (StepMode.EXTEND, 2, 17, (GraphPhase.PREFILL, 64)),
            (StepMode.MIXED, 9, 16, (GraphPhase.PREFILL, 16)),
            (StepMode.EXTEND, 1, 65, None),
            (StepMode.IDLE, 0, 0, None),
        ):
            with self.subTest(mode=mode, bs=bs, tokens=tokens):
                choice = bucket_graph_choice(
                    mode=mode, batch_size=bs, num_tokens=tokens, buckets=buckets
                )
                got = None if choice.is_eager else (choice.phase, choice.bucket)
                self.assertEqual(got, expected)
        eager_decode = bucket_graph_choice(
            mode=StepMode.DECODE,
            batch_size=1,
            num_tokens=1,
            buckets=GraphBuckets(prefill=(16,)),
        )
        self.assertTrue(eager_decode.is_eager)


class TestExternalRelease(CustomTestCase):
    def test_external_runners_keep_only_features_of_landed_blocks(self):
        """Whatever an external runner declares, launches outside TP text are
        refused until their block lands; TP, overlap and graphs have landed."""
        declared = RunnerCapabilities.all_supported()
        external = capabilities_in_this_release(declared, native=False)
        self.assertIs(capabilities_in_this_release(declared, native=True), declared)
        reasons = launch_refusals(
            external,
            ResolvedConfig(
                model_path="m",
                tp_size=2,
                pp_size=2,
                attn_dp_size=2,
                moe_ep_size=2,
                attn_cp_size=2,
                cuda_graph_phases=("decode",),
                overlap_schedule=True,
                advanced_kv=("sliding_window",),
                unified_memory=True,
                hicache=True,
                disaggregation_mode="prefill",
            ),
        )
        self.assertEqual(
            reasons,
            [
                "pipeline parallelism (pp_size=2)",
                "data-parallel attention (attn_dp_size=2)",
                "expert parallelism (ep_size=2; pass --ep-size 1)",
                "context parallelism (attn_cp_size=2; pass --attn-cp-size 1)",
                "PD disaggregation (disaggregation_mode=prefill)",
                "unified memory (drop --enable-unified-memory)",
                "KV beyond plain full attention (sliding_window; serve this "
                "model with --model-runner sglang)",
            ],
        )
        self.assertTrue(external.tensor_parallel)
        self.assertTrue(external.prompt_logprobs and external.top_logprobs)
        self.assertTrue(external.cuda_graphs and external.overlap_schedule)
        self.assertTrue(external.hicache)
        self.assertEqual(launch_refusals(external, ResolvedConfig(model_path="m")), [])

    def test_builtin_runner_keeps_the_direct_path_for_dcp_and_elastic_ep(self):
        """Prefill CP, EP and EPLB take the API on the built-in runner; decode
        CP and elastic EP keep the direct path, and so does the expert
        balancedness report, which returns metrics from each forward."""
        cases = {
            "": {},
            "decode context parallelism": dict(dcp_size=2),
            "elastic EP": dict(elastic_ep_backend="mooncake"),
            "per-forward captures": dict(balancedness=True),
        }
        for case, options in cases.items():
            with self.subTest(case=case):
                self.assertEqual(_native_refusals(**options), [case] if case else [])

    def test_external_refusals_say_how_to_launch_without_the_feature(self):
        """Context parallelism, expert parallelism and two-batch overlap stay
        refused for external runners in this release, with the flag to drop."""
        external = capabilities_in_this_release(
            RunnerCapabilities.all_supported(), native=False
        )
        reasons = launch_refusals(
            external,
            ResolvedConfig(
                model_path="m",
                moe_ep_size=4,
                attn_cp_size=2,
                dcp_size=2,
                two_batch_overlap=True,
            ),
        )
        self.assertEqual(
            reasons,
            [
                "expert parallelism (ep_size=4; pass --ep-size 1)",
                "context parallelism (attn_cp_size=2 dcp_size=2; pass "
                "--attn-cp-size 1 --dcp-size 1)",
                "two-batch overlap (drop --enable-two-batch-overlap)",
            ],
        )

    def test_graphs_are_refused_per_undeclared_phase(self):
        both = ResolvedConfig(model_path="m", cuda_graph_phases=("decode", "prefill"))
        decode_only = RunnerCapabilities(
            cuda_graphs=True, cuda_graph_phases=(GraphPhase.DECODE,)
        )
        self.assertEqual(
            launch_refusals(decode_only, both),
            ["CUDA graphs for prefill (pass --disable-prefill-cuda-graph)"],
        )
        decode = ResolvedConfig(model_path="m", cuda_graph_phases=("decode",))
        self.assertEqual(launch_refusals(decode_only, decode), [])
        # Without the capability, one reason covers every phase.
        self.assertEqual(len(launch_refusals(RunnerCapabilities(), both)), 1)

    def test_architecture_list_refuses_other_models(self):
        caps = RunnerCapabilities(architectures=("Qwen3ForCausalLM",))
        cfg = ResolvedConfig(model_path="m", architectures=("GptOssForCausalLM",))
        self.assertEqual(
            launch_refusals(caps, cfg),
            ["model architecture GptOssForCausalLM (serves: Qwen3ForCausalLM)"],
        )
        cfg = ResolvedConfig(model_path="m", architectures=("Qwen3ForCausalLM",))
        self.assertEqual(launch_refusals(caps, cfg), [])


def _native_refusals(*, dcp_size=1, elastic_ep_backend=None, balancedness=False):
    from unittest.mock import patch

    from sglang.srt.model_runner_plugins import native_steps

    features = SimpleNamespace(
        aux_hidden_state_capture=None,
        enable_return_routed_experts=False,
        enable_prefill_weight_versions=False,
        enable_encoder_swa_bounded_replay=False,
    )
    exec_ = SimpleNamespace(
        features=features,
        moe=SimpleNamespace(
            # EPLB statistics alone keep the API.
            expert_distribution_recorder_mode="stat",
            elastic_ep_backend=elastic_ep_backend,
        ),
        dllm=SimpleNamespace(dllm_algorithm=None),
    )
    model_runner = SimpleNamespace(
        is_generation=True, is_multimodal=False, is_draft_worker=False
    )
    getters = dict(
        reports_expert_balancedness=balancedness,
        # Prefill CP and EP sizes are not consulted.
        get_parallel=SimpleNamespace(
            dcp_size=dcp_size, attn_cp_size=2, moe_ep_size=2, tp_size=2
        ),
        get_schedule=SimpleNamespace(disable_overlap_schedule=True),
        get_spec=SimpleNamespace(speculative_algorithm=None),
        get_lora=SimpleNamespace(enable_lora=False),
        get_exec=exec_,
        use_mlx=False,
    )
    with contextlib.ExitStack() as stack:
        for name, value in getters.items():
            stack.enter_context(patch.object(native_steps, name, lambda v=value: v))
        return native_steps.native_step_refusals(model_runner)


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

    def finalize_model_capacity(self, *, agree_stages=True):
        self.calls.append("host.finalize")


class TestExternalStartup(CustomTestCase):
    def test_sglang_views_only_uniform_full_attention_slot_rows(self):
        """Host copies and transfers read a runner's KV as one row of slot bytes
        per slot and layer; a layout they cannot read that way is refused for
        them, not misread."""
        from sglang.srt.model_runner_plugins.kv_view import kv_view_refusal

        def layer(i, k_bytes=256):
            return FullAttentionLayer(
                layer_id=i,
                k_ptr=1 << 20,
                v_ptr=1 << 21,
                k_slot_bytes=k_bytes,
                v_slot_bytes=256,
            )

        def layout(*layers, sub_pools=()):
            return KVLayout(
                page_size=1, num_slots=8, full_attention=layers, sub_pools=sub_pools
            )

        self.assertIsNone(kv_view_refusal(layout(layer(0), layer(1))))
        self.assertEqual(
            kv_view_refusal(layout(layer(0), layer(1, k_bytes=512))),
            "full-attention layers with unequal slot sizes",
        )
        self.assertEqual(kv_view_refusal(layout()), "KV beyond full-attention layers")
        window = KVSubPool(
            kind=KVKind.SLIDING_WINDOW,
            num_items=4,
            buffers=(KVBuffer(layer_id=1, name="k", ptr=1 << 22, item_bytes=256),),
            window=2,
        )
        self.assertEqual(
            kv_view_refusal(layout(layer(0), sub_pools=(window,))),
            "KV beyond full-attention layers",
        )

    def _run(self, plugin, agree_min=lambda slots: slots, page_size=1):
        runner = ExternalRunner(
            name="fake", plugin=plugin, host=_Host(plugin.calls), agree_min=agree_min
        )
        runner.load(LoadContext(model_path="m", device="cuda", gpu_id=0))
        runner.allocate(RuntimeConfig(page_size=page_size))
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
        # Admission limits are SGLang's, set once PP stages have agreed.
        self.assertIsNone(final.limits)

    def test_tp_ranks_size_allocators_to_the_smallest_backing(self):
        """Each TP rank backs as many slots as its memory allows; SGLang's
        replicated allocators use the minimum over the group, page aligned."""
        seen = []

        def agree_min(slots):
            seen.append(slots)
            return 3001

        runner, final = self._run(
            _Plugin(page_size=16), agree_min=agree_min, page_size=16
        )
        self.assertEqual(seen, [4096])
        self.assertEqual(runner.layout.num_slots, 2992)
        self.assertEqual(final.num_slots, 2992)

    def test_a_resize_or_a_foreign_page_size_is_refused(self):
        with self.assertRaisesRegex(ValueError, "keep their allocated size"):
            self._run(_Plugin(final_slots=2048))
        with self.assertRaisesRegex(ValueError, "page size 16"):
            self._run(_Plugin(page_size=16))


if __name__ == "__main__":
    unittest.main()
