"""Refusal of launches and requests a model runner did not declare."""

import unittest
from array import array
from http import HTTPStatus
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import TokenizedEmbeddingReqInput
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_runner_plugins.api import (
    UNGATED_FEATURES,
    Feature,
    RefusedCombination,
    ResolvedConfig,
    RunnerCapabilities,
)
from sglang.srt.model_runner_plugins.native import SGLangRunnerPlugin
from sglang.srt.model_runner_plugins.refusal import (
    RequestOptions,
    admission_refusals,
    launch_refusals,
    launch_uses,
    refusal_message,
    request_uses,
)
from sglang.srt.model_runner_plugins.selection import ModelRunnerAdmission
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_EVERY_LAUNCH_FEATURE = ResolvedConfig(
    model_path="m",
    tp_size=2,
    pp_size=2,
    attn_dp_size=2,
    moe_ep_size=2,
    attn_cp_size=2,
    dcp_size=2,
    cuda_graph_phases=("decode", "prefill"),
    overlap_schedule=True,
    speculative_algorithm="EAGLE",
    lora=True,
    multimodal=True,
    pooling=True,
    disaggregation_mode="prefill",
    hicache=True,
    memory_saver=True,
    radix_cache=True,
    unified_memory=True,
    advanced_kv=("mla",),
    diffusion_algorithm="LowConfidence",
)
_EVERY_REQUEST_OPTION = RequestOptions(
    prompt_logprobs=True,
    top_logprobs_num=5,
    token_ids_logprob=True,
    return_hidden_states=True,
    input_embeds=True,
    positional_embed_overrides=True,
    lora_id="adapter",
    multimodal=True,
)


def _only(feature):
    return RunnerCapabilities(**{feature.value: True})


class TestCapabilityGates(CustomTestCase):
    def test_every_feature_is_reachable_and_gated_by_its_own_capability(self):
        """A Feature added without a detector or a capability field fails here."""
        launch = launch_uses(_EVERY_LAUNCH_FEATURE)
        request = request_uses(_EVERY_REQUEST_OPTION)
        self.assertEqual(launch.keys() | request.keys(), set(Feature))
        for feature in set(Feature) - UNGATED_FEATURES:
            with self.subTest(feature=feature):
                uses = {feature: "x"}
                gated = admission_refusals(
                    RunnerCapabilities(), launch={}, request=uses
                )
                self.assertEqual(len(gated), 1)
                self.assertEqual(
                    admission_refusals(_only(feature), launch={}, request=uses), []
                )

    def test_launch_refusal_lists_every_reason_in_one_message(self):
        cfg = ResolvedConfig(
            model_path="m",
            tp_size=2,
            cuda_graph_phases=("decode",),
            overlap_schedule=True,
            radix_cache=True,
        )
        reasons = launch_refusals(RunnerCapabilities(), cfg)
        self.assertEqual(
            refusal_message("foo", reasons),
            "--model-runner foo does not support: tensor parallelism (tp_size=2), "
            "CUDA graphs (pass --disable-decode-cuda-graph), overlap scheduling "
            "(pass --disable-overlap-schedule)",
        )

    def test_default_launch_needs_no_capability(self):
        self.assertEqual(
            launch_refusals(RunnerCapabilities(), ResolvedConfig(model_path="m")), []
        )

    def test_admission_lists_every_request_reason(self):
        reasons = admission_refusals(
            RunnerCapabilities(), launch={}, request=request_uses(_EVERY_REQUEST_OPTION)
        )
        self.assertEqual(
            reasons,
            [
                "prompt logprobs (logprob_start_len)",
                "top logprobs (top_logprobs_num=5 token_ids_logprob)",
                "returning hidden states (return_hidden_states)",
                "input embeddings (input_embeds positional_embed_overrides)",
                "LoRA (lora_path)",
                "multimodal inputs (image, video or audio items)",
            ],
        )


class TestRefusedCombinations(CustomTestCase):
    _caps = RunnerCapabilities(
        input_embeds=True,
        cuda_graphs=True,
        overlap_schedule=True,
        refused_combinations=(
            RefusedCombination(
                features=(Feature.INPUT_EMBEDS, Feature.RADIX_CACHE),
                hint="pass --disable-radix-cache",
            ),
            RefusedCombination(
                features=(Feature.CUDA_GRAPHS, Feature.OVERLAP_SCHEDULE)
            ),
        ),
    )

    def test_launch_combination_is_refused_at_launch_only(self):
        cfg = ResolvedConfig(
            model_path="m",
            cuda_graph_phases=("decode",),
            overlap_schedule=True,
            radix_cache=True,
        )
        self.assertEqual(
            launch_refusals(self._caps, cfg), ["CUDA graphs with overlap scheduling"]
        )
        # Admission repeats no launch-only refusal for a plain request.
        self.assertEqual(
            admission_refusals(self._caps, launch=launch_uses(cfg), request={}), []
        )

    def test_request_combination_needs_its_launch_half(self):
        request = request_uses(RequestOptions(input_embeds=True))
        with_radix = launch_uses(ResolvedConfig(model_path="m", radix_cache=True))
        self.assertEqual(
            admission_refusals(self._caps, launch=with_radix, request=request),
            ["input embeddings with radix cache (pass --disable-radix-cache)"],
        )
        self.assertEqual(admission_refusals(self._caps, launch={}, request=request), [])

    def test_combination_needs_two_distinct_features(self):
        with self.assertRaises(ValueError):
            RefusedCombination(features=(Feature.LORA, Feature.LORA))


class TestNativePlugin(CustomTestCase):
    def test_native_accepts_every_launch_and_request(self):
        caps = SGLangRunnerPlugin().describe(_EVERY_LAUNCH_FEATURE)
        self.assertEqual(launch_refusals(caps, _EVERY_LAUNCH_FEATURE), [])
        no_radix = launch_uses(ResolvedConfig(model_path="m"))
        request = request_uses(_EVERY_REQUEST_OPTION)
        self.assertEqual(admission_refusals(caps, launch=no_radix, request=request), [])

    def test_native_keeps_the_input_embeds_radix_cache_refusal(self):
        """The Python frontend refuses this pair today; admission must too."""
        caps = SGLangRunnerPlugin().describe(_EVERY_LAUNCH_FEATURE)
        reasons = admission_refusals(
            caps,
            launch=launch_uses(_EVERY_LAUNCH_FEATURE),
            request=request_uses(RequestOptions(input_embeds=True)),
        )
        self.assertEqual(
            reasons, ["input embeddings with radix cache (pass --disable-radix-cache)"]
        )


def _req(**kwargs):
    return Req("rid", "", array("q", [1, 2, 3, 4]), SamplingParams(), **kwargs)


class TestRequestOptions(CustomTestCase):
    def test_prompt_logprobs_follow_the_normalized_start(self):
        req = _req(return_logprob=True)
        req.logprob_start_len = 4  # Output logprobs only.
        self.assertFalse(RequestOptions.of_req(req).prompt_logprobs)
        req.logprob_start_len = 1
        self.assertTrue(RequestOptions.of_req(req).prompt_logprobs)
        req.return_logprob = False
        self.assertEqual(RequestOptions.of_req(req), RequestOptions())

    def test_request_fields_map_to_options(self):
        req = _req(
            return_logprob=True,
            top_logprobs_num=3,
            token_ids_logprob=[7],
            return_hidden_states=True,
            input_embeds=[[0.0]],
            lora_id="adapter",
        )
        req.logprob_start_len = -1
        self.assertEqual(
            RequestOptions.of_req(req),
            RequestOptions(
                top_logprobs_num=3,
                token_ids_logprob=True,
                return_hidden_states=True,
                input_embeds=True,
                lora_id="adapter",
            ),
        )


def _embedding_scheduler(admission):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model_runner_admission = admission
    scheduler.tokenizer = None
    scheduler.max_req_input_len = 1024
    scheduler._embedding_preserve_last_token_id = None
    scheduler._maybe_namespace_elastic_radix_cache = lambda req: None
    scheduler._record_kv_event_lora_name = lambda req, name: None
    scheduler.request_retirer = MagicMock()
    scheduler.queued = []
    scheduler._add_request_to_queue = scheduler.queued.append
    return scheduler


def _embedding_req(**kwargs):
    return TokenizedEmbeddingReqInput(
        rid="r",
        input_text="",
        input_ids=array("q", [1, 2, 3]),
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(),
        **kwargs,
    )


class TestSchedulerAdmission(CustomTestCase):
    def test_refused_request_returns_400_with_every_reason(self):
        admission = ModelRunnerAdmission(
            runner_name="foo", capabilities=RunnerCapabilities(), launch={}
        )
        with get_context().override_server_args():
            scheduler = _embedding_scheduler(admission)
            scheduler.handle_embedding_request(
                _embedding_req(lora_id="adapter", positional_embed_overrides=object())
            )
            scheduler.handle_embedding_request(_embedding_req())

        (req,) = [
            c.args[0] for c in scheduler.request_retirer.retire_unadmitted.mock_calls
        ]
        reason = req.finished_reason or req.to_finish
        self.assertEqual(reason.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(
            reason.message,
            "--model-runner foo does not support: input embeddings "
            "(positional_embed_overrides), LoRA (lora_path)",
        )
        # The plain request is admitted.
        self.assertEqual(len(scheduler.queued), 1)
        self.assertIsNone(scheduler.queued[0].to_finish)


if __name__ == "__main__":
    unittest.main()
