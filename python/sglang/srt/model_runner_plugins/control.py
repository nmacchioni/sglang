"""SGLang's runner control requests through a model runner's ``control``.

Under ``--model-runner``, every scheduler request that acts on the runner's
weights or memory between steps (weight updates and reads, checksums and
export, memory release and resume, expert statistics, elastic EP, LoRA
adapter loads and unloads) maps to a
``ControlOp`` and goes through the runner. A runner that does not declare the
operation is answered with the reply's own failed status on every rank, so no
reply format changes. Requests that act only on SGLang's state (cache flush,
profiling, pause and continue, the weight version) stay SGLang's.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

import msgspec

from sglang.srt.model_runner_plugins.api import (
    ControlOp,
    ControlRequest,
    ControlResult,
    KVLayout,
)

if TYPE_CHECKING:
    from sglang.srt.utils.common import TypeBasedDispatcher

logger = logging.getLogger(__name__)


class _Route(msgspec.Struct, frozen=True, kw_only=True):
    """How one SGLang control request maps onto ``control``."""

    op: ControlOp
    # The HTTP API the request comes from, named in refusals.
    api: str
    # The request's reply with a failed status and ``message``.
    refuse: Callable[[str], Any]
    # The ``ControlRequest.args`` of an external runner's call; None for
    # operations only the built-in runner serves.
    args: Optional[Callable[[Any], dict]] = None
    # The reply to an external runner's result.
    reply: Optional[Callable[[Any, ControlResult], Any]] = None


def _routes() -> dict[type, _Route]:
    from sglang.srt.managers import io_struct as io

    def status(out_type):
        return lambda message: out_type(success=False, message=message)

    def lora_refusal(message):
        return io.LoRAUpdateOutput(success=False, error_message=message)

    return {
        io.UpdateWeightFromDiskReqInput: _Route(
            op=ControlOp.WEIGHTS_UPDATE_FROM_DISK,
            api="update_weights_from_disk",
            refuse=status(io.UpdateWeightFromDiskReqOutput),
            args=lambda req: {
                "model_path": req.model_path,
                "load_format": req.load_format,
            },
            reply=lambda req, result: io.UpdateWeightFromDiskReqOutput(
                success=result.success, message=result.message
            ),
        ),
        io.GetWeightsByNameReqInput: _Route(
            op=ControlOp.WEIGHTS_GET,
            api="get_weights_by_name",
            # This reply carries no status: a refused read is an empty one.
            refuse=lambda message: io.GetWeightsByNameReqOutput(parameter=None),
            args=lambda req: {"name": req.name, "truncate_size": req.truncate_size},
            reply=lambda req, result: io.GetWeightsByNameReqOutput(
                parameter=result.value if result.success else None
            ),
        ),
        io.UpdateWeightsFromTensorReqInput: _Route(
            op=ControlOp.WEIGHTS_UPDATE_FROM_TENSOR,
            api="update_weights_from_tensor",
            refuse=status(io.UpdateWeightsFromTensorReqOutput),
        ),
        io.UpdateWeightsFromDistributedReqInput: _Route(
            op=ControlOp.WEIGHTS_UPDATE_FROM_DISTRIBUTED,
            api="update_weights_from_distributed",
            refuse=status(io.UpdateWeightsFromDistributedReqOutput),
        ),
        io.UpdateWeightsFromIPCReqInput: _Route(
            op=ControlOp.WEIGHTS_UPDATE_FROM_IPC,
            api="update_weights_from_ipc",
            refuse=status(io.UpdateWeightsFromIPCReqOutput),
        ),
        io.BeginWeightUpdateReqInput: _Route(
            op=ControlOp.WEIGHTS_UPDATE_SESSION,
            api="begin_weight_update",
            refuse=status(io.BeginWeightUpdateReqOutput),
        ),
        io.EndWeightUpdateReqInput: _Route(
            op=ControlOp.WEIGHTS_UPDATE_SESSION,
            api="end_weight_update",
            refuse=status(io.EndWeightUpdateReqOutput),
        ),
        io.PluginWeightUpdateReqInput: _Route(
            op=ControlOp.WEIGHTS_UPDATE_SESSION,
            api="plugin_weight_update",
            refuse=status(io.PluginWeightUpdateReqOutput),
        ),
        io.InitWeightsUpdateGroupReqInput: _Route(
            op=ControlOp.WEIGHTS_UPDATE_GROUP,
            api="init_weights_update_group",
            refuse=status(io.InitWeightsUpdateGroupReqOutput),
        ),
        io.DestroyWeightsUpdateGroupReqInput: _Route(
            op=ControlOp.WEIGHTS_UPDATE_GROUP,
            api="destroy_weights_update_group",
            refuse=status(io.DestroyWeightsUpdateGroupReqOutput),
        ),
        io.CheckWeightsReqInput: _Route(
            op=ControlOp.WEIGHTS_CHECKSUM,
            api="weights_checker",
            refuse=status(io.CheckWeightsReqOutput),
        ),
        io.InitWeightsSendGroupForRemoteInstanceReqInput: _Route(
            op=ControlOp.WEIGHTS_EXPORT,
            api="init_weights_send_group_for_remote_instance",
            refuse=status(io.InitWeightsSendGroupForRemoteInstanceReqOutput),
        ),
        io.SendWeightsToRemoteInstanceReqInput: _Route(
            op=ControlOp.WEIGHTS_EXPORT,
            api="send_weights_to_remote_instance",
            refuse=status(io.SendWeightsToRemoteInstanceReqOutput),
        ),
        io.RpcReqInput: _Route(
            # save_remote_model and save_sharded_model, the only RPC methods.
            op=ControlOp.WEIGHTS_EXPORT,
            api="rpc",
            refuse=status(io.RpcReqOutput),
        ),
        io.ReleaseMemoryOccupationReqInput: _Route(
            op=ControlOp.RELEASE,
            api="release_memory_occupation",
            refuse=status(io.ReleaseMemoryOccupationReqOutput),
        ),
        io.ResumeMemoryOccupationReqInput: _Route(
            op=ControlOp.RESUME,
            api="resume_memory_occupation",
            # This reply carries no status; a refused release left nothing to resume.
            refuse=lambda message: io.ResumeMemoryOccupationReqOutput(),
        ),
        io.ExpertDistributionReq: _Route(
            op=ControlOp.EXPERT_STATS,
            api="expert_distribution_record",
            refuse=lambda message: io.ExpertDistributionReqOutput(),
        ),
        io.LoadLoRAAdapterReqInput: _Route(
            op=ControlOp.LORA_LOAD,
            api="load_lora_adapter",
            refuse=lora_refusal,
        ),
        io.LoadLoRAAdapterFromTensorsReqInput: _Route(
            op=ControlOp.LORA_LOAD,
            api="load_lora_adapter_from_tensors",
            refuse=lora_refusal,
        ),
        io.UnloadLoRAAdapterReqInput: _Route(
            op=ControlOp.LORA_UNLOAD,
            api="unload_lora_adapter",
            refuse=lora_refusal,
        ),
        io.ScaleElasticEPReqInput: _Route(
            op=ControlOp.RECONFIGURE_PARALLEL,
            api="scale_elastic_ep",
            refuse=status(io.ScaleElasticEPReqOutput),
        ),
    }


class NativeControlBinding:
    """Hands the built-in runner's ``control`` the SGLang request it serves and
    the native handler that serves it, as ``NativeStepLowering`` hands ``plan``
    its batch."""

    def __init__(self):
        self._op: Optional[ControlOp] = None
        self._request: Any = None
        self._run: Optional[Callable[[], Any]] = None

    @contextlib.contextmanager
    def bind(
        self, *, op: ControlOp, request: Any, run: Callable[[], Any]
    ) -> Iterator[None]:
        self._op, self._request, self._run = op, request, run
        try:
            yield
        finally:
            self._op = self._request = self._run = None

    def request_of(self, req: ControlRequest) -> Any:
        """The SGLang request ``req`` stands for."""
        if req.op is not self._op or self._run is None:
            raise RuntimeError(
                "the built-in runner controls only the request SGLang binds"
            )
        return self._request

    def run(self, req: ControlRequest) -> Any:
        """The native handler's reply to the bound request."""
        self.request_of(req)
        return self._run()


class ModelRunnerControl:
    """Routes the scheduler's runner control requests through ``control``."""

    def __init__(
        self,
        *,
        runner_name: str,
        controller: Any,
        declared: frozenset[ControlOp],
        native: Optional[NativeControlBinding],
        layout: Callable[[], KVLayout],
        set_layout: Callable[[KVLayout], None],
        after_weights_update: Callable[[Any, bool, str], tuple[bool, str]],
        wait_for_launched_forward: Callable[[], None],
    ):
        self.runner_name = runner_name
        self._controller = controller
        self._declared = declared
        # The built-in runner: control() runs the native handler bound to it.
        self._native = native
        self._layout = layout
        self._set_layout = set_layout
        # SGLang's part after an external runner's weight update: the weight
        # version and the cache flush, as the native handlers do them.
        self._after_weights_update = after_weights_update
        self._wait_for_launched_forward = wait_for_launched_forward
        self._trace = ControlTrace.maybe_create()

    def wrap(self, dispatcher: TypeBasedDispatcher) -> None:
        """Send every routed request type through ``handle`` from now on."""
        for req_type, route in _routes().items():
            native_handler = dispatcher._mapping.get(req_type)
            if native_handler is None:
                continue
            dispatcher._mapping[req_type] = functools.partial(
                self.handle, route=route, native_handler=native_handler
            )
        dispatcher._mro_cache.clear()

    def handle(self, recv_req: Any, *, route: _Route, native_handler: Callable) -> Any:
        tic = time.perf_counter()
        if route.op not in self._declared or (
            self._native is None and route.args is None
        ):
            message = (
                f"--model-runner {self.runner_name} does not support {route.api} "
                f"(control op {route.op.value}; serve with --model-runner sglang)"
            )
            logger.error(message)
            self._record(route, success=False, tic=tic)
            return route.refuse(message)
        if self._native is not None:
            with self._native.bind(
                op=route.op, request=recv_req, run=lambda: native_handler(recv_req)
            ):
                result = self._call(ControlRequest(op=route.op))
            self._record(route, success=result.success, tic=tic)
            return result.value
        self._wait_for_launched_forward()
        result = self._call(ControlRequest(op=route.op, args=route.args(recv_req)))
        if route.op is ControlOp.WEIGHTS_UPDATE_FROM_DISK:
            success, message = self._after_weights_update(
                recv_req, result.success, result.message
            )
            result = msgspec.structs.replace(result, success=success, message=message)
        self._record(route, success=result.success, tic=tic)
        return route.reply(recv_req, result)

    def _call(self, req: ControlRequest) -> ControlResult:
        try:
            result = self._controller.control(req)
        except Exception as e:
            logger.exception(
                "Model runner %r control(%s) failed", self.runner_name, req.op.value
            )
            return ControlResult(success=False, message=f"{req.op.value} failed: {e}")
        if not isinstance(result, ControlResult):
            raise TypeError(
                f"Model runner {self.runner_name!r} control() returned "
                f"{type(result).__name__}; expected ControlResult."
            )
        if result.layout is not None:
            self._start_epoch(result.layout)
        return result

    def _start_epoch(self, layout: KVLayout) -> None:
        previous = self._layout()
        if layout.epoch <= previous.epoch or (layout.num_slots, layout.page_size) != (
            previous.num_slots,
            previous.page_size,
        ):
            raise ValueError(
                f"Model runner {self.runner_name!r} resumed into KV epoch "
                f"{layout.epoch} ({layout.num_slots} slots, page {layout.page_size}) "
                f"after epoch {previous.epoch} ({previous.num_slots} slots, page "
                f"{previous.page_size}); a new epoch keeps the sizes"
            )
        self._set_layout(msgspec.structs.replace(layout, limits=previous.limits))
        logger.info(
            "Model runner %r starts KV epoch %d (buffers %s)",
            self.runner_name,
            layout.epoch,
            "unmoved" if layout.placement() == previous.placement() else "moved",
        )

    def _record(self, route: _Route, *, success: bool, tic: float) -> None:
        if self._trace is not None:
            self._trace.record(
                runner=self.runner_name,
                api=route.api,
                op=route.op.value,
                success=success,
                epoch=self._layout().epoch,
                host_us=(time.perf_counter() - tic) * 1e6,
            )


class ControlTrace:
    """One JSON line per routed control request (``controls-<pid>.jsonl``),
    beside the step trace; off unless the step trace is on."""

    @classmethod
    def maybe_create(cls) -> Optional[ControlTrace]:
        from sglang.srt.environ import envs

        directory = envs.SGLANG_MODEL_RUNNER_STEP_TRACE_DIR.get()
        return None if directory is None else cls(directory=directory)

    def __init__(self, *, directory: str):
        os.makedirs(directory, exist_ok=True)
        self._path = os.path.join(directory, f"controls-{os.getpid()}.jsonl")

    def record(self, **fields) -> None:
        with open(self._path, "a") as f:
            f.write(json.dumps(fields) + "\n")
