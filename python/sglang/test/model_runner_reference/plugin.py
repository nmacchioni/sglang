"""The reference model runner: an independent, pure-PyTorch plugin that owns its
weights and KV bytes and serves text through the Step API, tensor parallel over
SGLang's TP group, with overlap scheduling and decode CUDA graphs, and verifies
speculative drafts (eagerly, under the draft tree's full mask).

Its decode graphs write logits into one buffer shared across steps; it declares
logits a shared output and honors the lifetime contract: a replay waits on the
release event of the previous output in that buffer before rewriting it, and
refuses to run if that output was never released."""

from __future__ import annotations

import logging
from typing import Optional

import msgspec
import torch

from sglang.srt.model_runner_plugins.api import (
    MODEL_RUNNER_API_VERSION,
    Capacity,
    ControlOp,
    ControlRequest,
    ControlResult,
    FullAttentionLayer,
    GraphBuckets,
    GraphPhase,
    KVLayout,
    LoadContext,
    LoadResult,
    LogitRows,
    Logits,
    LogitsLayout,
    MemoryNeeds,
    ModelInfo,
    OutputBuffer,
    ProjectRequest,
    ResolvedConfig,
    RunnerCapabilities,
    RuntimeConfig,
    SamplingOwner,
    Step,
    StepOutput,
)
from sglang.test.model_runner_reference.graphs import DecodeGraphs
from sglang.test.model_runner_reference.model import (
    DenseDecoder,
    last_rows,
    load_config,
    load_weights,
    resolve_checkpoint,
)

logger = logging.getLogger(__name__)

ARCHITECTURES = ("LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM")
# Arbitrary: memory left to SGLang's sampler and allocators when no
# --mem-fraction-static is resolved.
_DEFAULT_MEM_FRACTION = 0.8


class ReferencePlan(msgspec.Struct, kw_only=True):
    step: Step
    # Final-norm hidden states of every input row, kept for project().
    hidden: Optional[torch.Tensor] = None


class ReferenceRunner:
    api_version = MODEL_RUNNER_API_VERSION

    def __init__(self, *, logits_layout: LogitsLayout = LogitsLayout.FULL):
        # FULL gathers the vocabulary shards inside the runner; VOCAB_SHARDED
        # returns this rank's shard for SGLang to gather.
        self._logits_layout = logits_layout
        self._model: Optional[DenseDecoder] = None
        self._layout: Optional[KVLayout] = None
        self._graphs: Optional[DecodeGraphs] = None
        # The shared logits buffer: whether an output in it awaits release, and
        # the event the last release handed over.
        self._logits_held = False
        self._logits_free: Optional[torch.cuda.Event] = None
        self.steps_executed = 0

    def describe(self, cfg: ResolvedConfig) -> RunnerCapabilities:
        return RunnerCapabilities(
            tensor_parallel=True,
            prompt_logprobs=True,
            top_logprobs=True,
            overlap_schedule=True,
            cuda_graphs=True,
            hicache=True,
            pd_disaggregation=True,
            speculative_decoding=True,
            cuda_graph_phases=(GraphPhase.DECODE,),
            shared_outputs=(OutputBuffer.LOGITS,),
            architectures=ARCHITECTURES,
            control_ops=(ControlOp.WEIGHTS_UPDATE_FROM_DISK, ControlOp.WEIGHTS_GET),
            # Greedy argmax, inside the decode graphs; its argmax over shards
            # would be wrong, so the vocabulary-sharded variant leaves it to SGLang.
            fused_sampling=self._logits_layout is LogitsLayout.FULL,
        )

    @torch.inference_mode()
    def control(self, req: ControlRequest) -> ControlResult:
        if req.op is ControlOp.WEIGHTS_UPDATE_FROM_DISK:
            path = resolve_checkpoint(req.args["model_path"])
            if load_config(path) != self._model.full_config:
                return ControlResult(
                    success=False,
                    message=f"{path} is not a checkpoint of the served model's shape",
                )
            self._model.load_in_place(load_weights(path, device=self._model.device))
            logger.info("Reference runner loaded weights from %s", path)
            return ControlResult(message="Succeeded to update model weights.")
        if req.op is ControlOp.WEIGHTS_GET:
            name = req.args["name"]
            if name not in self._model.w:
                return ControlResult(success=False, message=f"no weight named {name}")
            weight = self._model.full_weight(name)[: req.args["truncate_size"]]
            return ControlResult(value=weight.float().cpu().tolist())
        raise ValueError(f"the reference runner does not declare {req.op.value}")

    def load(self, ctx: LoadContext) -> LoadResult:
        device = torch.device(ctx.device, ctx.gpu_id)
        path = resolve_checkpoint(ctx.model_path)
        config = load_config(path)
        before = torch.cuda.memory_allocated(device)
        self._model = DenseDecoder(
            config=config,
            weights=load_weights(path, device=device),
            device=device,
            tp=ctx.parallel.tp,
        )
        weight_bytes = torch.cuda.memory_allocated(device) - before
        logger.info("Reference runner loaded %s (%d bytes)", path, weight_bytes)
        return LoadResult(
            model_info=ModelInfo(
                context_len=config.max_position_embeddings,
                vocab_size=config.vocab_size,
            ),
            memory=MemoryNeeds(weight_bytes=weight_bytes),
        )

    def allocate(self, cfg: RuntimeConfig) -> KVLayout:
        model = self._model
        num_slots = cfg.max_total_tokens or self._slots_from_budget(cfg)
        num_slots -= num_slots % cfg.page_size
        # The first page is SGLang's padding page.
        kv = model.allocate_kv(num_slots + cfg.page_size)
        self._layout = KVLayout(
            page_size=cfg.page_size,
            num_slots=num_slots,
            full_attention=tuple(
                FullAttentionLayer(
                    layer_id=layer,
                    k_ptr=kv.k[layer].data_ptr(),
                    v_ptr=kv.v[layer].data_ptr(),
                    k_slot_bytes=kv.k.stride(1) * kv.k.element_size(),
                    v_slot_bytes=kv.v.stride(1) * kv.v.element_size(),
                )
                for layer in range(model.config.num_layers)
            ),
        )
        return self._layout

    @torch.inference_mode()
    def capture(self, buckets: GraphBuckets) -> None:
        if buckets.prefill:
            # SGLang refuses the launch first; prefill runs eager.
            raise ValueError("the reference runner captures decode graphs only")
        if buckets.decode:
            self._graphs = DecodeGraphs(
                model=self._model,
                buckets=buckets.decode,
                sharded=self._logits_layout is LogitsLayout.VOCAB_SHARDED,
            )
            logger.info("Reference runner captured decode graphs %s", buckets.decode)

    def finalize(self, capacity: Capacity) -> KVLayout:
        return self._layout

    def plan(self, step: Step) -> ReferencePlan:
        return ReferencePlan(step=step)

    @torch.inference_mode()
    def execute(self, plan: ReferencePlan) -> StepOutput:
        step = plan.step
        self.steps_executed += 1
        stream = torch.cuda.current_stream()
        if step.inputs_ready is not None:
            # Everything here runs on the runner stream, so this orders nothing
            # more; a runner working on a private stream needs it.
            stream.wait_event(step.inputs_ready)
        if step.outputs.logit_rows is LogitRows.NONE:
            return StepOutput()
        next_ids = None
        if step.graph.is_eager:
            plan.hidden = self._model.hidden_states(step)
            # A verify step scores every draft token.
            last = (
                plan.hidden
                if step.outputs.logit_rows is LogitRows.ALL
                else plan.hidden[last_rows(step)]
            )
            if self._logits_layout is LogitsLayout.VOCAB_SHARDED:
                logits = self._model.logits_shard(last)
            else:
                logits = self._model.logits(last)
            if step.sampling_owner is SamplingOwner.PLUGIN:
                next_ids = logits.argmax(dim=-1)
        else:
            if self._logits_held:
                raise RuntimeError(
                    "SGLang executed a step before releasing the previous output "
                    "of the shared logits buffer"
                )
            logits, graph_ids = self._graphs.replay(
                step, bucket=step.graph.bucket, before_logits=self._wait_logits_free
            )
            self._logits_held = True
            if step.sampling_owner is SamplingOwner.PLUGIN:
                # Out of the shared buffer: SGLang feeds them to the next step.
                next_ids = graph_ids.clone()
        ready = torch.cuda.Event()
        ready.record(stream)
        return StepOutput(
            logits=Logits(tensor=logits, layout=self._logits_layout),
            next_ids=next_ids,
            ready=ready,
        )

    def _wait_logits_free(self) -> None:
        if self._logits_free is not None:
            torch.cuda.current_stream().wait_event(self._logits_free)

    @torch.inference_mode()
    def project(self, req: ProjectRequest) -> torch.Tensor:
        # Every TP rank projects the same rows here, so the gather lines up.
        return self._model.logits(req.plan.hidden[req.rows])

    def release(self, out: StepOutput, after: Optional[torch.cuda.Event]) -> None:
        # Eager outputs are fresh per step; the plan, and with it the hidden
        # states, goes when SGLang drops it.
        if out.logits is not None and self._graphs is not None:
            if self._graphs.holds(out.logits.tensor):
                self._logits_held = False
                self._logits_free = after

    @property
    def graph_replays(self) -> int:
        return 0 if self._graphs is None else self._graphs.replays

    def close(self) -> None:
        self._graphs = None
        self._model = None
        self._layout = None

    def _slots_from_budget(self, cfg: RuntimeConfig) -> int:
        free, total = torch.cuda.mem_get_info(self._model.device)
        fraction = cfg.mem_fraction_static or _DEFAULT_MEM_FRACTION
        budget = free - (1 - fraction) * total
        if budget <= 0:
            raise RuntimeError("no device memory left for the reference runner's KV")
        return int(budget // self._model.bytes_per_slot())
