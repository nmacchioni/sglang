"""Launch and admission checks for ``--model-runner``, read from the published config."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec

from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.srt.model_runner_plugins.api import (
    UNGATED_FEATURES,
    ControlOp,
    Feature,
    ResolvedConfig,
    RunnerCapabilities,
)
from sglang.srt.model_runner_plugins.refusal import (
    RequestOptions,
    Uses,
    admission_refusals,
    launch_refusals,
    launch_uses,
    refusal_message,
    request_uses,
)
from sglang.srt.model_runner_plugins.registry import (
    LoadedModelRunner,
    load_model_runner,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_lora,
    get_memory,
    get_model,
    get_parallel,
    get_schedule,
    get_spec,
    process_model_config,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req


def check_model_runner_at_launch() -> None:
    """Refuse an unservable launch before any subprocess starts.

    A no-op without ``--model-runner``: nothing is enumerated or imported.
    """
    name = get_exec().kernel.model_runner
    if name is None:
        return
    runner = load_model_runner(name)
    cfg = resolved_config(process_model_config())
    capabilities = describe_runner(runner, cfg)
    reasons = launch_refusals(capabilities, cfg)
    if not runner.is_native:
        reasons += external_launch_refusals(capabilities, cfg)
    if reasons:
        raise ValueError(refusal_message(name, reasons))


# What an external runner can execute in this release, whatever it declares:
# text with SGLang sampling, with tensor parallelism, overlap scheduling and
# CUDA graphs when declared. Features of later blocks stay refused until SGLang
# can drive them through the API. Among them, for this release's parallel
# features: context parallelism shards KV across ranks in layouts KVLayout
# cannot describe; expert parallelism needs SGLang's expert placement (EPLB,
# redundant experts) and rebalance control, which the API does not carry yet;
# two-batch overlap and DP attention need Step.dp, and no external runner is
# driven under the MLP-sync rendezvous yet.
_EXTERNAL_FEATURES = frozenset(
    {
        Feature.TENSOR_PARALLEL,
        Feature.PROMPT_LOGPROBS,
        Feature.TOP_LOGPROBS,
        Feature.OVERLAP_SCHEDULE,
        Feature.CUDA_GRAPHS,
        # SGLang copies the runner's full-attention KV bytes to and from host
        # memory through the KVLayout (kv_view.py).
        Feature.HICACHE,
        # PD transfer backends register and move the same bytes, located
        # through the KVLayout, with SGLang's page leases and checksums.
        Feature.PD_DISAGGREGATION,
        # Target verify steps; only for drafts no model makes (n-gram), see
        # external_launch_refusals.
        Feature.SPECULATIVE_DECODING,
    }
)

# Speculative algorithms an external runner can serve: SGLang drafts without a
# model, the runner verifies. Draft models are loaded and run by SGLang's own
# runner, which reads its target's internals, so they need the built-in runner.
EXTERNAL_SPECULATIVE_ALGORITHMS = frozenset({"NGRAM"})


# Control operations an external runner can serve: those with arguments of
# their own (ControlRequest.args), which the reference runner proves. The
# others read SGLang's requests and native internals.
EXTERNAL_CONTROL_OPS = frozenset(
    {ControlOp.WEIGHTS_UPDATE_FROM_DISK, ControlOp.WEIGHTS_GET}
)


def external_launch_refusals(
    capabilities: RunnerCapabilities, cfg: ResolvedConfig
) -> list[str]:
    """Refusals beyond the declared capabilities, for an external runner."""
    algorithm = cfg.speculative_algorithm
    if (
        capabilities.speculative_decoding
        and algorithm is not None
        and algorithm.upper() not in EXTERNAL_SPECULATIVE_ALGORITHMS
    ):
        return [
            f"speculative decoding with a draft model (speculative_algorithm="
            f"{algorithm}; an installed runner verifies n-gram drafts only: pass "
            "--speculative-algorithm NGRAM, or serve with --model-runner sglang)"
        ]
    return []


def capabilities_in_this_release(
    capabilities: RunnerCapabilities, *, native: bool
) -> RunnerCapabilities:
    if native:
        return capabilities
    return msgspec.structs.replace(
        capabilities,
        **{
            f.value: False
            for f in Feature
            if f not in UNGATED_FEATURES and f not in _EXTERNAL_FEATURES
        },
        control_ops=tuple(
            op for op in capabilities.control_ops if op in EXTERNAL_CONTROL_OPS
        ),
    )


def resolved_config(model_config: ModelConfig) -> ResolvedConfig:
    parallel = get_parallel()
    graphs = get_exec().graph.cuda_graph_config
    return ResolvedConfig(
        model_path=get_model().model_path,
        architectures=tuple(model_config.hf_config.architectures or ()),
        tp_size=parallel.tp_size,
        pp_size=parallel.pp_size,
        attn_dp_size=parallel.attn_dp_size,
        moe_ep_size=parallel.moe_ep_size,
        attn_cp_size=parallel.attn_cp_size,
        dcp_size=parallel.dcp_size,
        two_batch_overlap=get_exec().overlap.enable_two_batch_overlap,
        cuda_graph_phases=tuple(
            phase
            for phase in Phase.ALL
            if graphs is not None and graphs[phase].backend != Backend.DISABLED
        ),
        overlap_schedule=not get_schedule().disable_overlap_schedule,
        speculative_algorithm=get_spec().speculative_algorithm,
        lora=bool(get_lora().enable_lora),
        multimodal=model_config.is_multimodal,
        pooling=not model_config.is_generation,
        disaggregation_mode=get_disagg().disaggregation_mode,
        hicache=get_memory().enable_hierarchical_cache,
        memory_saver=get_exec().features.enable_memory_saver,
        radix_cache=not get_memory().disable_radix_cache,
        unified_memory=get_memory().enable_unified_memory,
        advanced_kv=advanced_kv_kinds(model_config),
        diffusion_algorithm=get_exec().dllm.dllm_algorithm,
    )


def advanced_kv_kinds(model_config: ModelConfig) -> tuple[str, ...]:
    """The KV kinds beyond plain full attention that the model needs."""
    from sglang.srt.configs.hybrid_arch import mambaish_config
    from sglang.srt.configs.model_config import AttentionArch

    kinds = []
    if (
        model_config.is_hybrid_swa
        or model_config.sliding_window_size is not None
        or model_config.attention_chunk_size is not None
    ):
        kinds.append("sliding_window")
    if model_config.attention_arch == AttentionArch.MLA:
        kinds.append("mla")
    if mambaish_config(model_config) is not None:
        kinds.append("recurrent_state")
    if model_config.is_encoder_decoder:
        kinds.append("encoder_decoder")
    return tuple(kinds)


class ModelRunnerAdmission:
    """Per-request refusal at the scheduler's admission boundary."""

    def __init__(
        self, *, runner_name: str, capabilities: RunnerCapabilities, launch: Uses
    ):
        self._runner_name = runner_name
        self._capabilities = capabilities
        self._launch = launch

    @classmethod
    def maybe_create(
        cls, *, model_config: ModelConfig
    ) -> Optional[ModelRunnerAdmission]:
        """None without ``--model-runner``, so the default path checks nothing."""
        name = get_exec().kernel.model_runner
        if name is None:
            return None
        cfg = resolved_config(model_config)
        return cls(
            runner_name=name,
            capabilities=describe_runner(load_model_runner(name), cfg),
            launch=launch_uses(cfg),
        )

    def check(self, req: Req) -> Optional[str]:
        """The refusal message for ``req``, or None when the runner serves it."""
        return self.check_options(RequestOptions.of_req(req))

    def check_options(self, opts: RequestOptions) -> Optional[str]:
        reasons = admission_refusals(
            self._capabilities, launch=self._launch, request=request_uses(opts)
        )
        return refusal_message(self._runner_name, reasons) if reasons else None


def describe_runner(
    runner: LoadedModelRunner, cfg: ResolvedConfig
) -> RunnerCapabilities:
    """The runner's declared capabilities, narrowed to what this release can drive."""
    capabilities = runner.plugin.describe(cfg)
    if not isinstance(capabilities, RunnerCapabilities):
        raise TypeError(
            f"Model runner {runner.name!r} describe() returned "
            f"{type(capabilities).__name__}; expected RunnerCapabilities."
        )
    return capabilities_in_this_release(capabilities, native=runner.is_native)
