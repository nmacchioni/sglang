"""Public contract for model runner plugins (API version 1).

A package registers a zero-argument factory in the ``sglang.srt.model_runners``
entry point group; ``--model-runner NAME`` selects it. The factory returns an
object satisfying :class:`ModelRunnerPlugin`.

Importing this module must stay cheap: it is read by the launcher before any
device or model import.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence, runtime_checkable

import msgspec

if TYPE_CHECKING:
    import torch

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
    # Two micro-batches per step whose communication overlaps the other's
    # compute; the split arrives in ``Step.dp``.
    TWO_BATCH_OVERLAP = "two_batch_overlap"
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
    UNIFIED_MEMORY = "unified_memory"
    # KV beyond plain full attention: sliding window, MLA, recurrent state,
    # encoder-decoder caches.
    ADVANCED_KV = "advanced_kv"
    DIFFUSION_LLM = "diffusion_llm"
    RADIX_CACHE = "radix_cache"


UNGATED_FEATURES = frozenset({Feature.RADIX_CACHE})


class GraphPhase(str, Enum):
    """A phase a runner can replay graphs for; values match ``--cuda-graph-config``."""

    DECODE = "decode"
    PREFILL = "prefill"


class OutputBuffer(str, Enum):
    """A ``StepOutput`` field whose storage a runner may reuse across steps."""

    LOGITS = "logits"
    PROMPT_LOGPROBS = "prompt_logprobs"


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
    two_batch_overlap: bool = False
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
    unified_memory: bool = False
    advanced_kv: bool = False
    diffusion_llm: bool = False
    refused_combinations: tuple[RefusedCombination, ...] = ()
    # Model architectures served (as in the HF config); None places no limit
    # beyond the features above, which already gate model-dependent KV.
    architectures: Optional[tuple[str, ...]] = None
    # StepOutput.logprobs carries prompt logprobs computed in execute();
    # otherwise SGLang computes them through project().
    native_prompt_logprobs: bool = False
    # Phases the runner captures graphs for, when ``cuda_graphs`` is set; a
    # launch with graphs on for another phase is refused.
    cuda_graph_phases: tuple[GraphPhase, ...] = (GraphPhase.DECODE, GraphPhase.PREFILL)
    # Outputs that live in buffers a later step overwrites (graph output
    # buffers). SGLang keeps no reference to them past ``release``, and releases
    # each before the next ``execute``; any other output is fresh per step.
    shared_outputs: tuple[OutputBuffer, ...] = ()

    @classmethod
    def all_supported(
        cls,
        *,
        refused_combinations: tuple[RefusedCombination, ...] = (),
        native_prompt_logprobs: bool = False,
        shared_outputs: tuple[OutputBuffer, ...] = (),
    ) -> RunnerCapabilities:
        return cls(
            **{f.value: True for f in Feature if f not in UNGATED_FEATURES},
            refused_combinations=refused_combinations,
            native_prompt_logprobs=native_prompt_logprobs,
            shared_outputs=shared_outputs,
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
    two_batch_overlap: bool = False
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
    unified_memory: bool = False
    # KV kinds beyond full attention the model needs ("sliding_window", "mla",
    # "recurrent_state", "encoder_decoder"); empty for a plain dense model.
    advanced_kv: tuple[str, ...] = ()
    diffusion_algorithm: Optional[str] = None


class RankGroup(msgspec.Struct, frozen=True, kw_only=True):
    """One process group SGLang built, as this rank sees it.

    ``device`` is the ``torch.distributed`` group for device tensors (NCCL on
    CUDA), ``cpu`` the one for host tensors (Gloo); both are None when the
    group has one rank. A runner may run collectives on them only inside the
    calls SGLang makes on every member of the group in lockstep.
    """

    rank: int = 0
    size: int = 1
    # Global ranks of the members, in group-rank order.
    ranks: tuple[int, ...] = (0,)
    device: Optional[Any] = None
    cpu: Optional[Any] = None

    def __post_init__(self) -> None:
        if not 0 <= self.rank < self.size or len(self.ranks) != self.size:
            raise ValueError(
                f"RankGroup needs 0 <= rank < size == len(ranks), got rank="
                f"{self.rank}, size={self.size}, ranks={self.ranks}"
            )


SINGLE_RANK = RankGroup()


class ParallelLayout(msgspec.Struct, frozen=True, kw_only=True):
    """The process groups SGLang built for this rank, before ``load``.

    A runner joins no group of its own and adds no collective of its own to a
    step: every choice that changes a collective (graph bucket, eager
    fallback, padded width, local mode) arrives decided in the ``Step``.
    """

    # Every rank of one model replica: SGLang sends each member the same Step.
    tp: RankGroup = SINGLE_RANK
    pp: RankGroup = SINGLE_RANK
    # The attention TP group, a slice of ``tp`` when attention is data parallel.
    attn_tp: RankGroup = SINGLE_RANK
    attn_dp_rank: int = 0
    attn_dp_size: int = 1


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
    parallel: ParallelLayout = ParallelLayout()

    def __post_init__(self) -> None:
        tp, pp = self.parallel.tp, self.parallel.pp
        if (tp.rank, tp.size, pp.rank, pp.size) != (
            self.tp_rank,
            self.tp_size,
            self.pp_rank,
            self.pp_size,
        ):
            raise ValueError(
                "LoadContext ranks disagree with its parallel layout: "
                f"tp {self.tp_rank}/{self.tp_size} vs {tp.rank}/{tp.size}, "
                f"pp {self.pp_rank}/{self.pp_size} vs {pp.rank}/{pp.size}"
            )


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
    """CUDA graph buckets per phase, ascending; empty when the phase is eager.

    ``decode`` buckets count requests, ``prefill`` buckets count input tokens.
    SGLang replays a step in the smallest bucket that holds it and runs any
    larger step eagerly.
    """

    decode: tuple[int, ...] = ()
    prefill: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name, buckets in (("decode", self.decode), ("prefill", self.prefill)):
            if list(buckets) != sorted(set(buckets)) or any(b <= 0 for b in buckets):
                raise ValueError(
                    f"GraphBuckets.{name} must be positive and strictly ascending, "
                    f"got {buckets}"
                )

    def of(self, phase: GraphPhase) -> tuple[int, ...]:
        return self.decode if phase is GraphPhase.DECODE else self.prefill


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


class KVKind(str, Enum):
    """A sub-pool beside the full-attention layers."""

    # Sliding-window attention layers: token slots of their own, which SGLang
    # maps from the full-attention slots it allocates.
    SLIDING_WINDOW = "sliding_window"
    # Recurrent (linear-attention, SSM) state: one slot per request row,
    # copied to checkpoint slots that the prefix cache keeps.
    RECURRENT_STATE = "recurrent_state"


class KVBuffer(msgspec.Struct, frozen=True, kw_only=True):
    """One buffer of one layer in a sub-pool: item ``i`` (a token slot or a
    state slot) is the ``item_bytes`` bytes at ``ptr + i * item_bytes``."""

    layer_id: int
    # Which part of the layer's item this is ("k", "v", "conv0", "temporal").
    name: str
    ptr: int
    item_bytes: int

    def __post_init__(self) -> None:
        if self.item_bytes <= 0:
            raise ValueError(
                f"KVBuffer {self.name} of layer {self.layer_id} needs positive "
                f"item_bytes, got {self.item_bytes}"
            )


class KVSubPool(msgspec.Struct, frozen=True, kw_only=True):
    """Items of one kind the backing serves, every buffer ``num_items`` long."""

    kind: KVKind
    num_items: int
    buffers: tuple[KVBuffer, ...]
    # SLIDING_WINDOW: the tokens each query attends back over.
    window: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_items <= 0 or not self.buffers:
            raise ValueError(
                f"a {self.kind.value} sub-pool needs items and buffers, got "
                f"{self.num_items} items and {len(self.buffers)} buffers"
            )
        keys = [(b.layer_id, b.name) for b in self.buffers]
        if len(set(keys)) != len(keys):
            raise ValueError(f"a {self.kind.value} sub-pool lists a buffer twice")
        if (self.kind is KVKind.SLIDING_WINDOW) != (self.window is not None):
            raise ValueError("a window size goes with a sliding-window sub-pool")

    def placement(self) -> tuple[tuple[int, str, int, int], ...]:
        """What may not change within an epoch: every buffer's address and item size."""
        return tuple((b.layer_id, b.name, b.ptr, b.item_bytes) for b in self.buffers)


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
    physical slots; the runner owns the bytes. The backing serves physical
    slots ``[0, num_slots + page_size)``: SGLang hands out ``num_slots`` of
    them and keeps the first page for padded rows. Sliding-window layers and
    recurrent state live in ``sub_pools``, each with its own item count.
    Addresses stay fixed for an ``epoch``: ``finalize`` may change ``num_slots``
    and sub-pool sizes but never moves a buffer.
    """

    epoch: int = 0
    page_size: int
    # Physical slots per layer that the backing serves.
    num_slots: int
    full_attention: tuple[FullAttentionLayer, ...] = ()
    sub_pools: tuple[KVSubPool, ...] = ()
    # Set by ``finalize``; None from ``allocate``.
    limits: Optional[AdmissionLimits] = None

    def __post_init__(self) -> None:
        if self.page_size < 1 or self.num_slots < 0:
            raise ValueError(
                f"KVLayout needs page_size >= 1 and num_slots >= 0, got "
                f"page_size={self.page_size}, num_slots={self.num_slots}"
            )
        layer_ids = [layer.layer_id for layer in self.full_attention]
        layer_ids += [
            layer_id
            for pool in self.sub_pools
            for layer_id in sorted({b.layer_id for b in pool.buffers})
        ]
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError(f"KVLayout lists a layer twice: {sorted(layer_ids)}")
        kinds = [pool.kind for pool in self.sub_pools]
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"KVLayout lists a sub-pool kind twice: {kinds}")

    def sub_pool(self, kind: KVKind) -> Optional[KVSubPool]:
        return next((p for p in self.sub_pools if p.kind is kind), None)

    def placement(self) -> tuple:
        """Every buffer's address and item size: fixed within an epoch."""
        return (
            self.full_attention,
            tuple((p.kind, p.placement()) for p in self.sub_pools),
        )


class StepMode(str, Enum):
    """What one step runs; EXTEND, DECODE, MIXED and IDLE execute in this version."""

    EXTEND = "extend"
    DECODE = "decode"
    # Extend rows and decode rows in one batch, laid out as an extend.
    MIXED = "mixed"
    IDLE = "idle"
    VERIFY = "verify"
    DRAFT = "draft"
    DRAFT_EXTEND = "draft_extend"


EXECUTABLE_STEP_MODES = frozenset(
    {StepMode.EXTEND, StepMode.DECODE, StepMode.MIXED, StepMode.IDLE}
)


class SamplingOwner(str, Enum):
    SGLANG = "sglang"
    PLUGIN = "plugin"


class GraphChoice(msgspec.Struct, frozen=True, kw_only=True):
    """The graph SGLang chose for a step: replay ``phase``'s graph of ``bucket``
    rows (requests for decode, tokens for prefill), or run eager when both are
    None. Rows past the step's own are padding, whose KV writes go to the
    reserved slot 0."""

    phase: Optional[GraphPhase] = None
    bucket: Optional[int] = None

    def __post_init__(self) -> None:
        if (self.phase is None) != (self.bucket is None):
            raise ValueError("GraphChoice names a phase and a bucket, or neither")

    @property
    def is_eager(self) -> bool:
        return self.bucket is None


EAGER = GraphChoice()


class Rows(msgspec.Struct, frozen=True, kw_only=True):
    """Which requests a step serves and where their input rows are.

    Request ``b`` owns input rows ``[query_start[b], query_start[b] +
    query_lens[b])`` and continues a sequence whose first ``prefix_lens[b]``
    positions are already in KV. Decode and idle steps leave the ``query_*``
    and ``prefix_*`` fields None: request ``b`` then owns input row ``b``.
    """

    # [B] device: request b's row of ``KVTables.page_table``.
    req_pool_indices: torch.Tensor
    # [B] device: request b's KV length once this step has written it.
    seq_lens: torch.Tensor
    # [B] host copy of seq_lens, when SGLang has one.
    seq_lens_cpu: Optional[torch.Tensor] = None
    query_start: Optional[torch.Tensor] = None
    query_lens: Optional[torch.Tensor] = None
    prefix_lens: Optional[torch.Tensor] = None
    query_lens_cpu: Optional[Sequence[int]] = None
    prefix_lens_cpu: Optional[Sequence[int]] = None

    @property
    def batch_size(self) -> int:
        return self.req_pool_indices.shape[0]


class WindowTables(msgspec.Struct, frozen=True, kw_only=True):
    """The sliding-window sub-pool's slots, translated by SGLang."""

    # [T] device: the window slot input row t writes its K and V to.
    write_slots: torch.Tensor
    # [num_slots + page_size] device, persistent across steps: slot_map[s] is
    # the window slot of full-attention slot s (``page_table`` holds full
    # slots). Positions that fell out of the window map to a freed slot, which
    # a runner must not read.
    slot_map: torch.Tensor


class StateTables(msgspec.Struct, frozen=True, kw_only=True):
    """Recurrent-state slots, and the state operations SGLang scheduled for this
    step. The operations run on the runner stream before any layer reads the
    state, in this order: ``clear``, then the copies."""

    # [R] device, persistent across steps: the state slot of request row r.
    req_to_state: torch.Tensor
    # [N] device: zero these state slots (a request that starts without state).
    clear: Optional[torch.Tensor] = None
    # [M] device each: copy state slot copy_src[i] to copy_dst[i] (a request
    # resuming from a prefix-cache checkpoint).
    copy_src: Optional[torch.Tensor] = None
    copy_dst: Optional[torch.Tensor] = None
    # Checkpoints the prefix cache keeps, [B] device each: where checkpoint_mask
    # is set, request b's state after checkpoint_lens[b] tokens (extend steps;
    # after the step on decode steps) is copied to slot checkpoint_slots[b].
    checkpoint_slots: Optional[torch.Tensor] = None
    checkpoint_mask: Optional[torch.Tensor] = None
    checkpoint_lens: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if (self.copy_src is None) != (self.copy_dst is None):
            raise ValueError("state copies name sources and destinations")


class KVTables(msgspec.Struct, frozen=True, kw_only=True):
    """Physical KV slots, as SGLang allocated and translated them."""

    # [T] device: the slot input row t writes its K and V to.
    write_slots: torch.Tensor
    # [R, max_context] device, persistent across steps: page_table[r, p] is the
    # slot holding position p of the request in row r. SGLang owns and writes it.
    page_table: torch.Tensor
    # With a sliding-window sub-pool (KVLayout.sub_pools); None otherwise.
    window: Optional[WindowTables] = None
    # With a recurrent-state sub-pool; None otherwise.
    state: Optional[StateTables] = None


class LogitRows(str, Enum):
    # One row per request, for its last input row; request b is row b.
    LAST = "last"
    # No logits (idle steps, and every pipeline stage but the last).
    NONE = "none"


class OutputSpec(msgspec.Struct, frozen=True, kw_only=True):
    logit_rows: LogitRows = LogitRows.LAST
    # Some request reads prompt logprobs this step: from StepOutput.logprobs
    # when the runner declares native_prompt_logprobs, else through project().
    prompt_logprobs: bool = False
    # A pipeline stage before the last: the step returns StepOutput.pp_out, the
    # tensors SGLang sends to the next stage as its Step.pp_in.
    pp_out: bool = False


class DPInfo(msgspec.Struct, frozen=True, kw_only=True):
    """What SGLang's per-step MLP-sync rendezvous decided for every DP rank.

    Under data-parallel attention each DP rank runs its own batch, or an idle
    step, and every rank joins the MLP collectives of every step. The values
    here were gathered from every rank at the rendezvous and are identical on
    all of them; a runner sizes the step's collectives from them and never
    from its local batch alone.
    """

    # This rank's index into the per-rank tuples.
    rank: int
    # Input rows each DP rank runs this step, as published at the rendezvous;
    # global_num_tokens[rank] is this step's own T.
    global_num_tokens: tuple[int, ...]
    global_num_tokens_for_logprob: tuple[int, ...]
    # Some DP rank extends this step. A decode batch then runs presented as an
    # extend, as native does, so every rank's collectives take one layout.
    extend_in_batch: bool = False
    # The coordinated graph votes: every rank could replay its decode
    # (resp. prefill) graph. Step.graph is eager on every rank otherwise.
    decode_graph: bool = False
    # With this vote and an extend on some rank, every rank replays the
    # prefill graph of one bucket (``Step.graph``), idle and decode ranks
    # included: each pads its step to the bucket's rows, a decode step
    # presented as one-token extends and an idle step as one padding request.
    prefill_graph: bool = False
    # Two-batch overlap: the rendezvous voted every rank to run this step as
    # two micro-batches, this rank's requests [0, split) and [split, B), each
    # joining its own round of MLP collectives. None: no rank splits.
    tbo_split_seq_index: Optional[int] = None

    def __post_init__(self) -> None:
        n = len(self.global_num_tokens)
        if not 0 <= self.rank < n or len(self.global_num_tokens_for_logprob) != n:
            raise ValueError(
                f"DPInfo needs 0 <= rank < {n} entries per rank, got rank "
                f"{self.rank} and {len(self.global_num_tokens_for_logprob)} "
                "logprob counts"
            )


class Step(msgspec.Struct, frozen=True, kw_only=True):
    """One forward step, as SGLang scheduled it.

    Tensors are views of what SGLang built for the step; a runner must not
    modify them. Fields of later blocks (input embeddings, masks, adapters,
    multimodal items, speculative and parallel inputs) are not part of this
    version.

    With overlap scheduling, ``plan`` runs while the device still executes
    the previous step: host fields (shapes, ``*_cpu`` copies) are final then,
    device tensors only once ``inputs_ready`` fires. A decode step's
    ``input_ids`` are the previous step's samples.
    """

    mode: StepMode
    # [T] device token ids; T = 0 for an idle step.
    input_ids: torch.Tensor
    # [T] device positions.
    positions: torch.Tensor
    rows: Rows
    kv: KVTables
    outputs: OutputSpec
    sampling_owner: SamplingOwner = SamplingOwner.SGLANG
    graph: GraphChoice = EAGER
    # A device event recorded on the runner stream once the device inputs are
    # written; work on another stream waits on it. None: they were written
    # before plan() was called (overlap scheduling off).
    inputs_ready: Optional[Any] = None
    # Under data-parallel attention; None otherwise.
    dp: Optional[DPInfo] = None
    # Every pipeline stage but the first: what the previous stage's pp_out
    # held, received by SGLang. None on the first stage.
    pp_in: Optional[dict[str, torch.Tensor]] = None

    def __post_init__(self) -> None:
        num_tokens = self.input_ids.shape[0]
        if self.positions.shape[-1] != num_tokens:
            raise ValueError(
                f"Step has {num_tokens} input ids but "
                f"{self.positions.shape[-1]} positions"
            )
        if self.kv.write_slots.shape[0] != num_tokens:
            raise ValueError(
                f"Step has {num_tokens} input ids but "
                f"{self.kv.write_slots.shape[0]} KV write slots"
            )
        window = self.kv.window
        if window is not None and window.write_slots.shape[0] != num_tokens:
            raise ValueError(
                f"Step has {num_tokens} input ids but "
                f"{window.write_slots.shape[0]} window write slots"
            )
        has_queries = self.rows.query_start is not None
        if self.mode in (StepMode.EXTEND, StepMode.MIXED) and not has_queries:
            raise ValueError(f"a {self.mode.value} step needs query offsets")
        if self.mode in (StepMode.DECODE, StepMode.IDLE) and (
            has_queries or num_tokens != self.rows.batch_size
        ):
            raise ValueError(
                f"a {self.mode.value} step has one input row per request, got "
                f"{num_tokens} rows for {self.rows.batch_size} requests"
            )
        if self.graph.bucket is not None:
            rows = (
                self.rows.batch_size
                if self.graph.phase is GraphPhase.DECODE
                else num_tokens
            )
            if rows > self.graph.bucket:
                raise ValueError(
                    f"a step of {rows} rows cannot replay the "
                    f"{self.graph.phase.value} graph of {self.graph.bucket}"
                )
        if self.dp is not None:
            _check_rendezvous(self.dp, num_tokens=num_tokens, graph=self.graph)
            split = self.dp.tbo_split_seq_index
            if split is not None and not 0 <= split <= self.rows.batch_size:
                raise ValueError(
                    f"a two-batch-overlap split at request {split} of a step "
                    f"with {self.rows.batch_size} requests"
                )
        if (
            self.graph.phase is GraphPhase.PREFILL
            and self.mode in (StepMode.DECODE, StepMode.IDLE)
            and (self.dp is None or not self.dp.extend_in_batch)
        ):
            raise ValueError(
                f"a {self.mode.value} step replays a prefill graph only under DP "
                "attention, presented as an extend while another rank extends"
            )


class RendezvousMismatch(ValueError):
    """A step disagrees with what its rank published at SGLang's per-step
    rendezvous: its peers already sized the step's collectives from the
    published values, so running it would hang or corrupt them."""


def _check_rendezvous(dp: DPInfo, *, num_tokens: int, graph: GraphChoice) -> None:
    published = dp.global_num_tokens[dp.rank]
    if published != num_tokens:
        raise RendezvousMismatch(
            f"DP rank {dp.rank} published {published} input rows at the "
            f"rendezvous but its step has {num_tokens}"
        )
    if graph.bucket is None:
        return
    voted = dp.decode_graph if graph.phase is GraphPhase.DECODE else dp.prefill_graph
    if not voted:
        raise RendezvousMismatch(
            f"DP rank {dp.rank} replays its {graph.phase.value} graph, but the "
            "rendezvous voted the step eager"
        )
    if graph.bucket < max(dp.global_num_tokens):
        raise RendezvousMismatch(
            f"DP rank {dp.rank} replays the graph of {graph.bucket}, below the "
            f"largest rank's {max(dp.global_num_tokens)} rows at the rendezvous"
        )


class LogitsLayout(str, Enum):
    # [rows, vocab_size] on every TP rank.
    FULL = "full"
    # [rows, shard] on each TP rank: rank r holds vocabulary columns
    # [r * shard, (r + 1) * shard) of a vocabulary padded to tp_size * shard.
    # SGLang gathers the shards across the TP group, the collective it schedules
    # on every member in lockstep, and drops the padding.
    VOCAB_SHARDED = "vocab_sharded"


class Logits(msgspec.Struct, frozen=True, kw_only=True):
    """Next-token logits; SGLang may modify ``tensor`` in place."""

    # [rows, vocab_size], or [rows, shard] when vocabulary-sharded.
    tensor: torch.Tensor
    layout: LogitsLayout = LogitsLayout.FULL
    # [rows] device: the request each row belongs to; None when row b is request b.
    row_requests: Optional[torch.Tensor] = None


class PromptLogprobs(msgspec.Struct, frozen=True, kw_only=True):
    """Prompt logprobs computed by the runner, in SGLang's input-logprob layout
    (``LogitsProcessorOutput.input_*``)."""

    token_logprobs: torch.Tensor
    top_logprobs_val: Optional[list] = None
    top_logprobs_idx: Optional[list] = None
    top_logprobs_weights: Optional[list] = None
    token_ids_logprobs_val: Optional[list] = None
    token_ids_logprobs_idx: Optional[list] = None
    # Fires when host copies of the lists above are complete.
    copy_done: Optional[Any] = None


class StepOutput(msgspec.Struct, frozen=True, kw_only=True):
    """What one step produced.

    Lifetimes: SGLang reads an output only after ``ready``; it may modify
    ``logits.tensor`` in place and nothing else; an output listed in
    ``RunnerCapabilities.shared_outputs`` may be overwritten by a later step
    only once ``release``'s ``after`` event for this output has fired.
    """

    logits: Optional[Logits] = None
    logprobs: Optional[PromptLogprobs] = None
    # Stages before the last (OutputSpec.pp_out): what SGLang sends to the
    # next stage, which receives it as Step.pp_in.
    pp_out: Optional[dict[str, torch.Tensor]] = None
    # A device event that fires when the outputs are written; SGLang's runner
    # stream waits on it before any consumer. None: the outputs are ordered on
    # the runner stream before execute() returns.
    ready: Optional[Any] = None

    def buffers(self) -> frozenset[OutputBuffer]:
        held = set()
        if self.logits is not None:
            held.add(OutputBuffer.LOGITS)
        if self.logprobs is not None:
            held.add(OutputBuffer.PROMPT_LOGPROBS)
        return frozenset(held)


class ProjectRequest(msgspec.Struct, frozen=True, kw_only=True):
    """Apply the LM head to input rows of an executed step.

    SGLang selects and chunks the rows as native prompt logprobs do and calls
    ``project`` once per chunk, between ``execute`` and ``release``, on every
    TP rank in lockstep with the same rows, so a runner may gather inside it.
    ``layout`` is the layout SGLang reads the result in.
    """

    # The plan whose step's rows are projected.
    plan: Any
    # [N] int64 device: input rows of that step.
    rows: torch.Tensor
    layout: LogitsLayout = LogitsLayout.FULL


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


@runtime_checkable
class ModelRunnerExecutor(ModelRunnerLifecycle, Protocol):
    """Execution, after startup. SGLang calls ``plan`` then ``execute`` for every
    step, ``project`` for prompt-logprob rows when the runner computes no native
    prompt logprobs, and ``release`` once the last consumer of the output
    (sampling, host copies, grammar) is enqueued. ``execute`` launches device
    work on the current (runner) stream, or rejoins it before returning, and
    must not synchronize with the host.

    Outputs are released in the order they were produced, each before the
    next ``execute`` when it holds a shared output buffer.
    """

    def plan(self, step: Step) -> Any: ...

    def execute(self, plan: Any) -> StepOutput: ...

    # [N, vocab_size] logits for the requested rows.
    def project(self, req: ProjectRequest) -> torch.Tensor: ...

    # ``after`` is a device event that fires when SGLang's last consumer of
    # ``out`` is done; a runner orders any overwrite of ``out``'s buffers after
    # it (a stream wait, not a host sync). None: every consumer is complete or
    # ordered on the runner stream before this call.
    def release(self, out: StepOutput, after: Optional[Any]) -> None: ...
