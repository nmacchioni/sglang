"""The built-in ``sglang`` model runner plugin.

It declares everything SGLang's own runner supports. Its lifecycle delegates to
the scheduler's native startup phases, in the native order. Its steps run
today's ``ModelRunner.forward`` on the batch SGLang lowered for the ``Step``;
configurations outside the Step subset keep the direct path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol

import msgspec

from sglang.srt.model_runner_plugins.api import (
    MODEL_RUNNER_API_VERSION,
    Capacity,
    ControlOp,
    ControlRequest,
    ControlResult,
    Feature,
    GraphBuckets,
    KVLayout,
    LoadContext,
    LoadResult,
    OutputBuffer,
    ProjectRequest,
    RefusedCombination,
    ResolvedConfig,
    RunnerCapabilities,
    RuntimeConfig,
    Step,
    StepOutput,
)
from sglang.srt.model_runner_plugins.control import NativeControlBinding
from sglang.srt.model_runner_plugins.native_results import (
    native_kv_layout,
    native_load_result,
)
from sglang.srt.model_runner_plugins.native_steps import (
    NativePlan,
    NativeStepLowering,
    native_execute,
    native_plan,
)

if TYPE_CHECKING:
    import torch

    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.model_runner import ModelRunnerOutput

_NATIVE_CAPABILITIES = RunnerCapabilities.all_supported(
    refused_combinations=(
        # The tokenizer manager refuses the same pair for the Python frontend.
        RefusedCombination(
            features=(Feature.INPUT_EMBEDS, Feature.RADIX_CACHE),
            hint="pass --disable-radix-cache",
        ),
    ),
    native_prompt_logprobs=True,
    # Graph runners write logits into one persistent buffer (GraphSharedOutput).
    shared_outputs=(OutputBuffer.LOGITS,),
)


class NativeStartupHost(Protocol):
    """Today's startup, split at the lifecycle boundaries (the ``Scheduler``).

    Target and draft workers stay together here: native startup interleaves
    their loading, pool allocation, attention setup and graph capture.
    """

    tp_worker: TpModelWorker
    # A speculative draft worker, or None.
    draft_worker: object

    def load_model_workers(self) -> None: ...

    def alloc_model_memory_pools(self) -> None: ...

    def capture_model_graphs(self) -> None: ...

    def finalize_model_capacity(self, *, agree_stages: bool = True) -> None: ...

    def agree_stage_capacity(self) -> None: ...


class SGLangRunnerPlugin:
    api_version = MODEL_RUNNER_API_VERSION

    def __init__(self, host: Optional[NativeStartupHost] = None):
        # None when built for describe() alone (the launcher, admission).
        self._host = host
        self.lowering = NativeStepLowering()
        self.control_binding = NativeControlBinding()
        # Releasing the KV ends its epoch; the resume that follows starts the next.
        self._kv_released = False
        self._epoch = 0
        # The host's model runner, once a step has run (run_plan).
        self._model_runner = None

    def describe(self, cfg: ResolvedConfig) -> RunnerCapabilities:
        return _NATIVE_CAPABILITIES

    def load(self, ctx: LoadContext) -> LoadResult:
        host = self._require_host()
        host.load_model_workers()
        return native_load_result(
            tp_worker=host.tp_worker, draft_worker=host.draft_worker
        )

    def allocate(self, cfg: RuntimeConfig) -> KVLayout:
        # The native pool configurator sizes from the same published bags cfg
        # is built from, so delegating honors it.
        host = self._require_host()
        host.alloc_model_memory_pools()
        return native_kv_layout(tp_worker=host.tp_worker, with_limits=False)

    def capture(self, buckets: GraphBuckets) -> None:
        # Native derives its capture buckets from the same graph config.
        self._require_host().capture_model_graphs()

    def finalize(self, capacity: Capacity) -> KVLayout:
        # Resizes only when the post-capture backing is active. PP stages agree
        # on capacity after this phase, and SGLang then sets the limits.
        host = self._require_host()
        host.finalize_model_capacity(agree_stages=False)
        return native_kv_layout(tp_worker=host.tp_worker, with_limits=False)

    def plan(self, step: Step) -> NativePlan:
        return native_plan(
            step=step, forward_batch=self.lowering.forward_batch_of(step)
        )

    def plan_batch(
        self, *, forward_batch: object, pp_proxy_tensors: Optional[object]
    ) -> NativePlan:
        """The plan of a batch whose Step SGLang does not materialize, since
        nothing would read it (``plan`` reads only the batch back): the native
        forward decides its graph, as on the direct path."""
        return NativePlan(
            forward_batch=forward_batch, graph=None, pp_proxy_tensors=pp_proxy_tensors
        )

    def execute(self, plan: NativePlan) -> StepOutput:
        return native_execute(
            model_runner=self._require_host().tp_worker.model_runner,
            plan=plan,
            lowering=self.lowering,
        )

    def run_plan(self, plan: NativePlan) -> ModelRunnerOutput:
        """execute() for a step nothing reads the StepOutput of (see
        plan_batch): the native forward's own output whole, so nothing (no
        capture) is dropped and no StepOutput is built. Its release would be a
        no-op, so none follows."""
        model_runner = self._model_runner
        if model_runner is None:
            model_runner = self._model_runner = (
                self._require_host().tp_worker.model_runner
            )
        return model_runner.forward(
            plan.forward_batch, pp_proxy_tensors=plan.pp_proxy_tensors
        )

    def project(self, req: ProjectRequest) -> torch.Tensor:
        raise RuntimeError(
            "The built-in runner returns prompt logprobs natively; it has no project()"
        )

    def release(self, out: StepOutput, after: object) -> None:
        # Native outputs live in per-forward tensors, or in the shared logits
        # buffer, whose consumers (sampling) run on the runner stream before the
        # next replay; host copies read only sampler outputs. That stream order
        # is the native replay fence, so ``after`` adds nothing to wait on.
        pass

    def control(self, req: ControlRequest) -> ControlResult:
        # The native handler does the work, rank agreement and PD queue
        # deregistration included; its reply keeps today's format.
        reply = self.control_binding.run(req)
        success = _reply_success(reply)
        layout = None
        if success and req.op in (ControlOp.RELEASE, ControlOp.RESUME):
            layout = self._kv_epoch_after(req)
        return ControlResult(success=success, value=reply, layout=layout)

    def _kv_epoch_after(self, req: ControlRequest) -> Optional[KVLayout]:
        """The layout of the epoch a KV resume starts; None for anything else."""
        from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE

        tags = self.control_binding.request_of(req).tags
        if tags and GPU_MEMORY_TYPE_KV_CACHE not in tags:
            return None
        if req.op is ControlOp.RELEASE:
            self._kv_released = True
            return None
        if not self._kv_released:
            return None
        self._kv_released = False
        self._epoch += 1
        layout = native_kv_layout(
            tp_worker=self._require_host().tp_worker, with_limits=False
        )
        return msgspec.structs.replace(layout, epoch=self._epoch)

    def close(self) -> None:
        # Native workers own no teardown beyond process exit; drop the handle.
        self._host = self._model_runner = None

    def _require_host(self) -> NativeStartupHost:
        if self._host is None:
            raise RuntimeError(
                "The built-in model runner's lifecycle runs only inside the "
                "scheduler that created it (or after close())."
            )
        return self._host


def _reply_success(reply: Any) -> bool:
    """A native control reply's status; replies without one report success."""
    if reply is None or "success" not in type(reply).__struct_fields__:
        return True
    return bool(reply.success)
