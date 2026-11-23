"""Refusal of launch configurations and requests a model runner cannot serve.

Pure functions over :class:`ResolvedConfig`, :class:`RequestOptions` and
:class:`RunnerCapabilities`: each lists every reason, never only the first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec

from sglang.srt.model_runner_plugins.api import (
    Feature,
    RefusedCombination,
    ResolvedConfig,
    RunnerCapabilities,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

_LABELS = {
    Feature.TENSOR_PARALLEL: "tensor parallelism",
    Feature.PIPELINE_PARALLEL: "pipeline parallelism",
    Feature.DP_ATTENTION: "data-parallel attention",
    Feature.EXPERT_PARALLEL: "expert parallelism",
    Feature.CONTEXT_PARALLEL: "context parallelism",
    Feature.TWO_BATCH_OVERLAP: "two-batch overlap",
    Feature.CUDA_GRAPHS: "CUDA graphs",
    Feature.OVERLAP_SCHEDULE: "overlap scheduling",
    Feature.SPECULATIVE_DECODING: "speculative decoding",
    Feature.LORA: "LoRA",
    Feature.MULTIMODAL: "multimodal inputs",
    Feature.POOLING: "embedding and pooling models",
    Feature.PD_DISAGGREGATION: "PD disaggregation",
    Feature.HICACHE: "hierarchical cache",
    Feature.MEMORY_SAVER: "memory saver",
    Feature.PROMPT_LOGPROBS: "prompt logprobs",
    Feature.TOP_LOGPROBS: "top logprobs",
    Feature.RETURN_HIDDEN_STATES: "returning hidden states",
    Feature.INPUT_EMBEDS: "input embeddings",
    Feature.UNIFIED_MEMORY: "unified memory",
    Feature.ADVANCED_KV: "KV beyond plain full attention",
    Feature.DIFFUSION_LLM: "diffusion LLMs",
    Feature.RADIX_CACHE: "radix cache",
}

# Feature -> detail shown in parentheses: the value in use or how to turn it off.
Uses = dict[Feature, str]


class RequestOptions(msgspec.Struct, frozen=True, kw_only=True):
    """The per-request options a runner may not support, after normalization."""

    prompt_logprobs: bool = False
    top_logprobs_num: int = 0
    token_ids_logprob: bool = False
    return_hidden_states: bool = False
    input_embeds: bool = False
    positional_embed_overrides: bool = False
    lora_id: Optional[str] = None
    multimodal: bool = False

    @classmethod
    def of_req(cls, req: Req) -> RequestOptions:
        """Read a ``Req`` whose ``logprob_start_len`` is already normalized."""
        return cls(
            prompt_logprobs=req.return_logprob
            and 0 <= req.logprob_start_len < len(req.origin_input_ids),
            top_logprobs_num=(
                req.logprob.top_logprobs_num if req.return_logprob else 0
            ),
            token_ids_logprob=req.return_logprob
            and req.logprob.token_ids_logprob is not None,
            return_hidden_states=bool(req.return_hidden_states),
            input_embeds=req.input_embeds is not None,
            positional_embed_overrides=req.positional_embed_overrides is not None,
            lora_id=req.lora_id,
            multimodal=req.multimodal_inputs is not None,
        )


def launch_uses(cfg: ResolvedConfig) -> Uses:
    uses: Uses = {}
    if cfg.tp_size > 1:
        uses[Feature.TENSOR_PARALLEL] = f"tp_size={cfg.tp_size}"
    if cfg.pp_size > 1:
        uses[Feature.PIPELINE_PARALLEL] = f"pp_size={cfg.pp_size}"
    if cfg.attn_dp_size > 1:
        uses[Feature.DP_ATTENTION] = f"attn_dp_size={cfg.attn_dp_size}"
    if cfg.moe_ep_size > 1:
        uses[Feature.EXPERT_PARALLEL] = f"ep_size={cfg.moe_ep_size}; pass --ep-size 1"
    cp = [
        (f"{name}={size}", f"--{name.replace('_', '-')} 1")
        for name, size in (
            ("attn_cp_size", cfg.attn_cp_size),
            ("dcp_size", cfg.dcp_size),
        )
        if size > 1
    ]
    if cp:
        uses[Feature.CONTEXT_PARALLEL] = (
            f"{' '.join(v for v, _ in cp)}; pass {' '.join(f for _, f in cp)}"
        )
    if cfg.two_batch_overlap:
        uses[Feature.TWO_BATCH_OVERLAP] = "drop --enable-two-batch-overlap"
    if cfg.cuda_graph_phases:
        flags = " ".join(f"--disable-{p}-cuda-graph" for p in cfg.cuda_graph_phases)
        uses[Feature.CUDA_GRAPHS] = f"pass {flags}"
    if cfg.overlap_schedule:
        uses[Feature.OVERLAP_SCHEDULE] = "pass --disable-overlap-schedule"
    if cfg.speculative_algorithm is not None:
        uses[Feature.SPECULATIVE_DECODING] = (
            f"speculative_algorithm={cfg.speculative_algorithm}"
        )
    if cfg.lora:
        uses[Feature.LORA] = "drop --enable-lora and --lora-paths"
    if cfg.multimodal:
        uses[Feature.MULTIMODAL] = "pass --disable-multimodal"
    if cfg.pooling:
        uses[Feature.POOLING] = "the model is served for embeddings or scores"
    if cfg.disaggregation_mode != "null":
        uses[Feature.PD_DISAGGREGATION] = (
            f"disaggregation_mode={cfg.disaggregation_mode}"
        )
    if cfg.hicache:
        uses[Feature.HICACHE] = "drop --enable-hierarchical-cache"
    if cfg.memory_saver:
        uses[Feature.MEMORY_SAVER] = "drop --enable-memory-saver"
    if cfg.unified_memory:
        uses[Feature.UNIFIED_MEMORY] = "drop --enable-unified-memory"
    if cfg.advanced_kv:
        uses[Feature.ADVANCED_KV] = " ".join(cfg.advanced_kv)
    if cfg.diffusion_algorithm is not None:
        uses[Feature.DIFFUSION_LLM] = f"dllm_algorithm={cfg.diffusion_algorithm}"
    if cfg.radix_cache:
        uses[Feature.RADIX_CACHE] = "pass --disable-radix-cache"
    return uses


def request_uses(opts: RequestOptions) -> Uses:
    uses: Uses = {}
    if opts.prompt_logprobs:
        uses[Feature.PROMPT_LOGPROBS] = "logprob_start_len"
    top = []
    if opts.top_logprobs_num > 0:
        top.append(f"top_logprobs_num={opts.top_logprobs_num}")
    if opts.token_ids_logprob:
        top.append("token_ids_logprob")
    if top:
        uses[Feature.TOP_LOGPROBS] = " ".join(top)
    if opts.return_hidden_states:
        uses[Feature.RETURN_HIDDEN_STATES] = "return_hidden_states"
    embeds = []
    if opts.input_embeds:
        embeds.append("input_embeds")
    if opts.positional_embed_overrides:
        embeds.append("positional_embed_overrides")
    if embeds:
        uses[Feature.INPUT_EMBEDS] = " ".join(embeds)
    if opts.lora_id is not None:
        uses[Feature.LORA] = "lora_path"
    if opts.multimodal:
        uses[Feature.MULTIMODAL] = "image, video or audio items"
    return uses


def launch_refusals(caps: RunnerCapabilities, cfg: ResolvedConfig) -> list[str]:
    uses = launch_uses(cfg)
    reasons = _unsupported(caps, uses)
    if caps.cuda_graphs:
        declared = {phase.value for phase in caps.cuda_graph_phases}
        reasons += [
            f"CUDA graphs for {phase} (pass --disable-{phase}-cuda-graph)"
            for phase in cfg.cuda_graph_phases
            if phase not in declared
        ]
    if caps.architectures is not None:
        reasons += [
            f"model architecture {arch} (serves: {', '.join(caps.architectures)})"
            for arch in cfg.architectures
            if arch not in caps.architectures
        ]
    reasons += [
        _combination_reason(combo)
        for combo in caps.refused_combinations
        if set(combo.features) <= uses.keys()
    ]
    return reasons


def admission_refusals(
    caps: RunnerCapabilities, *, launch: Uses, request: Uses
) -> list[str]:
    """Reasons a request is refused; launch-only refusals were raised at launch."""
    reasons = _unsupported(caps, request)
    in_use = launch.keys() | request.keys()
    reasons += [
        _combination_reason(combo)
        for combo in caps.refused_combinations
        if set(combo.features) <= in_use and not set(combo.features) <= launch.keys()
    ]
    return reasons


def refusal_message(runner_name: str, reasons: list[str]) -> str:
    return f"--model-runner {runner_name} does not support: {', '.join(reasons)}"


def _unsupported(caps: RunnerCapabilities, uses: Uses) -> list[str]:
    return [
        f"{_LABELS[feature]} ({detail})"
        for feature, detail in uses.items()
        if not caps.supports(feature)
    ]


def _combination_reason(combo: RefusedCombination) -> str:
    label = " with ".join(_LABELS[f] for f in combo.features)
    return f"{label} ({combo.hint})" if combo.hint else label
