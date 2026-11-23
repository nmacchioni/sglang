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


class LoadContext(msgspec.Struct, frozen=True, kw_only=True):
    """What SGLang hands a runner when it loads, on one rank.

    The model config, tokenizer and process groups are SGLang's and are
    resolved before ``load``; the checkpoint must be readable by SGLang's
    ``ModelConfig``.
    """

    model_path: str
    device: str
    gpu_id: int
    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1


class ModelInfo(msgspec.Struct, frozen=True, kw_only=True):
    """Facts about the loaded model that SGLang sizes and admits against."""

    context_len: int
    vocab_size: int
    is_generation: bool = True

    def __post_init__(self) -> None:
        if self.context_len <= 0 or self.vocab_size <= 0:
            raise ValueError(
                f"ModelInfo needs a positive context_len and vocab_size, got "
                f"context_len={self.context_len}, vocab_size={self.vocab_size}"
            )


class MemoryNeeds(msgspec.Struct, frozen=True, kw_only=True):
    """Device memory the loaded runner holds before any KV is allocated."""

    # Bytes the weights took on the device while loading.
    weight_bytes: int
    # Bytes already resident before SGLang measured free memory (a weight cache).
    preloaded_weight_bytes: int = 0

    def __post_init__(self) -> None:
        if self.weight_bytes < 0 or self.preloaded_weight_bytes < 0:
            raise ValueError("MemoryNeeds byte counts must not be negative")


class LoadResult(msgspec.Struct, frozen=True, kw_only=True):
    model_info: ModelInfo
    memory: MemoryNeeds


class RuntimeConfig(msgspec.Struct, frozen=True, kw_only=True):
    """The KV budget SGLang asks a runner to back in ``allocate``."""

    page_size: int
    mem_fraction_static: Optional[float] = None
    # Explicit limits (--max-total-tokens, --max-running-requests); None sizes
    # from the memory budget.
    max_total_tokens: Optional[int] = None
    max_running_requests: Optional[int] = None


class GraphBuckets(msgspec.Struct, frozen=True, kw_only=True):
    """Configured CUDA graph buckets per phase; empty when the phase is eager."""

    decode: tuple[int, ...] = ()
    prefill: tuple[int, ...] = ()


class Capacity(msgspec.Struct, frozen=True, kw_only=True):
    """What SGLang allows ``finalize`` to settle on.

    ``resize`` permits resizing the backing after capture, which happens only
    if the backing supports it; the explicit limits stay caps.
    """

    resize: bool = True
    max_total_tokens: Optional[int] = None
    max_running_requests: Optional[int] = None


class FullAttentionLayer(msgspec.Struct, frozen=True, kw_only=True):
    """One full-attention layer's K and V bytes: slot ``i`` starts at
    ``ptr + i * slot_bytes``."""

    layer_id: int
    k_ptr: int
    v_ptr: int
    k_slot_bytes: int
    v_slot_bytes: int


class AdmissionLimits(msgspec.Struct, frozen=True, kw_only=True):
    """The final token and request limits SGLang admits against."""

    max_total_num_tokens: int
    max_running_requests: int
    max_req_len: int
    max_req_input_len: int

    def __post_init__(self) -> None:
        if self.max_total_num_tokens <= 0 or self.max_running_requests <= 0:
            raise ValueError(
                "AdmissionLimits needs positive max_total_num_tokens and "
                f"max_running_requests, got {self.max_total_num_tokens} and "
                f"{self.max_running_requests}"
            )
        if not 0 < self.max_req_len or self.max_req_input_len > self.max_req_len:
            raise ValueError(
                f"AdmissionLimits needs 0 < max_req_len and max_req_input_len <= "
                f"max_req_len, got {self.max_req_len} and {self.max_req_input_len}"
            )


class KVLayout(msgspec.Struct, frozen=True, kw_only=True):
    """The bytes backing SGLang's KV slots on one rank.

    SGLang allocates every page and slot ID and translates logical IDs to
    physical slots; the runner owns the bytes. This version describes
    full-attention layers only; a runner whose KV is not plain full attention
    reports no layers. Addresses stay fixed for an ``epoch``: ``finalize`` may
    change ``num_slots`` but never moves a layer.
    """

    epoch: int = 0
    page_size: int
    # Physical slots per layer that the backing serves.
    num_slots: int
    full_attention: tuple[FullAttentionLayer, ...] = ()
    # Set by ``finalize``; None from ``allocate``.
    limits: Optional[AdmissionLimits] = None

    def __post_init__(self) -> None:
        if self.page_size < 1 or self.num_slots < 0:
            raise ValueError(
                f"KVLayout needs page_size >= 1 and num_slots >= 0, got "
                f"page_size={self.page_size}, num_slots={self.num_slots}"
            )
        layer_ids = [layer.layer_id for layer in self.full_attention]
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError(f"KVLayout lists a layer twice: {layer_ids}")


@runtime_checkable
class ModelRunnerPlugin(Protocol):
    """The selection subset of the model runner plugin protocol.

    ``describe`` runs in the launcher and in each scheduler process before
    anything touches the device, so it must not allocate device memory, load
    weights or start processes.
    """

    api_version: int

    def describe(self, cfg: ResolvedConfig) -> RunnerCapabilities: ...


@runtime_checkable
class ModelRunnerLifecycle(ModelRunnerPlugin, Protocol):
    """Startup and shutdown, called on each rank in this order: ``load``,
    ``allocate``, ``capture``, ``finalize``; ``close`` last, also after a
    failure in any phase, so it must release whatever was created so far.
    """

    def load(self, ctx: LoadContext) -> LoadResult: ...

    def allocate(self, cfg: RuntimeConfig) -> KVLayout: ...

    def capture(self, buckets: GraphBuckets) -> None: ...

    def finalize(self, capacity: Capacity) -> KVLayout: ...

    def close(self) -> None: ...
