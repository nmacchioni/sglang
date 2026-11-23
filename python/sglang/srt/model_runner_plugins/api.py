"""Public contract for model runner plugins (API version 1, selection subset).

A package registers a zero-argument factory in the ``sglang.srt.model_runners``
entry point group; ``--model-runner NAME`` selects it. The factory returns an
object satisfying :class:`ModelRunnerPlugin`.

Importing this module must stay cheap: it is read by the launcher before any
device or model import.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Protocol, runtime_checkable

import msgspec

MODEL_RUNNER_API_VERSION = 1
MODEL_RUNNERS_GROUP = "sglang.srt.model_runners"
# Built in, so no entry point may claim it.
NATIVE_MODEL_RUNNER = "sglang"


class Feature(str, Enum):
    """A feature a launch configuration or a request can use.

    Every member except :attr:`RADIX_CACHE` is gated by the
    :class:`RunnerCapabilities` field of the same name. ``RADIX_CACHE`` is
    owned by SGLang and never refused alone; it exists so a runner can refuse
    a combination with it.
    """

    TENSOR_PARALLEL = "tensor_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    DP_ATTENTION = "dp_attention"
    EXPERT_PARALLEL = "expert_parallel"
    CONTEXT_PARALLEL = "context_parallel"
    CUDA_GRAPHS = "cuda_graphs"
    OVERLAP_SCHEDULE = "overlap_schedule"
    SPECULATIVE_DECODING = "speculative_decoding"
    LORA = "lora"
    MULTIMODAL = "multimodal"
    POOLING = "pooling"
    PD_DISAGGREGATION = "pd_disaggregation"
    HICACHE = "hicache"
    MEMORY_SAVER = "memory_saver"
    PROMPT_LOGPROBS = "prompt_logprobs"
    TOP_LOGPROBS = "top_logprobs"
    RETURN_HIDDEN_STATES = "return_hidden_states"
    INPUT_EMBEDS = "input_embeds"
    RADIX_CACHE = "radix_cache"


UNGATED_FEATURES = frozenset({Feature.RADIX_CACHE})


class RefusedCombination(msgspec.Struct, frozen=True, kw_only=True):
    """Features that are each supported but not together.

    ``features`` may mix launch and request features; a combination naming a
    request feature is checked per request, otherwise at launch. ``hint`` says
    how to avoid it, e.g. ``"pass --disable-radix-cache"``.
    """

    features: tuple[Feature, ...]
    hint: str = ""

    def __post_init__(self) -> None:
        if len(set(self.features)) < 2:
            raise ValueError(
                "a refused combination names at least two distinct features"
            )


class RunnerCapabilities(msgspec.Struct, frozen=True, kw_only=True):
    """What a model runner can execute.

    Every feature defaults to unsupported: a runner declares only what it
    implements, and SGLang refuses the rest at launch or at admission.
    """

    tensor_parallel: bool = False
    pipeline_parallel: bool = False
    dp_attention: bool = False
    expert_parallel: bool = False
    context_parallel: bool = False
    cuda_graphs: bool = False
    overlap_schedule: bool = False
    speculative_decoding: bool = False
    lora: bool = False
    multimodal: bool = False
    pooling: bool = False
    pd_disaggregation: bool = False
    hicache: bool = False
    memory_saver: bool = False
    prompt_logprobs: bool = False
    top_logprobs: bool = False
    return_hidden_states: bool = False
    input_embeds: bool = False
    refused_combinations: tuple[RefusedCombination, ...] = ()

    @classmethod
    def all_supported(
        cls, *, refused_combinations: tuple[RefusedCombination, ...] = ()
    ) -> RunnerCapabilities:
        return cls(
            **{f.value: True for f in Feature if f not in UNGATED_FEATURES},
            refused_combinations=refused_combinations,
        )

    def supports(self, feature: Feature) -> bool:
        if feature in UNGATED_FEATURES:
            return True
        # Feature values name the capability fields one to one.
        return getattr(self, feature.value)


class ResolvedConfig(msgspec.Struct, frozen=True, kw_only=True):
    """The resolved launch configuration a runner is described against.

    Built from the published config bags and SGLang's ``ModelConfig``, after
    every resolution rule has run. Defaults describe a single-GPU eager text
    launch.
    """

    model_path: str
    architectures: tuple[str, ...] = ()
    tp_size: int = 1
    pp_size: int = 1
    attn_dp_size: int = 1
    moe_ep_size: int = 1
    attn_cp_size: int = 1
    dcp_size: int = 1
    # Phases ("decode", "prefill") whose CUDA graph backend is not disabled.
    cuda_graph_phases: tuple[str, ...] = ()
    overlap_schedule: bool = False
    speculative_algorithm: Optional[str] = None
    lora: bool = False
    multimodal: bool = False
    pooling: bool = False
    disaggregation_mode: str = "null"
    hicache: bool = False
    memory_saver: bool = False
    radix_cache: bool = False


@runtime_checkable
class ModelRunnerPlugin(Protocol):
    """The selection subset of the model runner plugin protocol.

    ``describe`` runs in the launcher and in each scheduler process before
    anything touches the device, so it must not allocate device memory, load
    weights or start processes.
    """

    api_version: int

    def describe(self, cfg: ResolvedConfig) -> RunnerCapabilities: ...
