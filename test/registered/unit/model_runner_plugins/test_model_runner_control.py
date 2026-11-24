"""Control requests through a model runner's control()."""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import io_struct as io
from sglang.srt.model_runner_plugins.api import (
    ControlOp,
    ControlRequest,
    ControlResult,
    FullAttentionLayer,
    KVLayout,
    RunnerCapabilities,
)
from sglang.srt.model_runner_plugins.control import ModelRunnerControl
from sglang.srt.model_runner_plugins.native import SGLangRunnerPlugin
from sglang.srt.model_runner_plugins.selection import capabilities_in_this_release
from sglang.srt.utils.common import TypeBasedDispatcher

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_LAYOUT = KVLayout(
    page_size=1,
    num_slots=128,
    full_attention=(
        FullAttentionLayer(
            layer_id=0, k_ptr=4096, v_ptr=8192, k_slot_bytes=16, v_slot_bytes=16
        ),
    ),
)


class _Recorder:
    """An external runner's control(), recording its calls."""

    def __init__(self, result=ControlResult()):
        self.calls = []
        self.result = result

    def control(self, req: ControlRequest) -> ControlResult:
        self.calls.append(req)
        return self.result


def _native_handlers():
    """The scheduler's own handlers for the routed requests, as mocks."""
    return {
        io.ReleaseMemoryOccupationReqInput: MagicMock(
            return_value=io.ReleaseMemoryOccupationReqOutput(
                success=True, message="Success"
            )
        ),
        io.ResumeMemoryOccupationReqInput: MagicMock(
            return_value=io.ResumeMemoryOccupationReqOutput()
        ),
        io.UpdateWeightFromDiskReqInput: MagicMock(
            return_value=io.UpdateWeightFromDiskReqOutput(success=True, message="ok")
        ),
        io.GetWeightsByNameReqInput: MagicMock(
            return_value=io.GetWeightsByNameReqOutput(parameter=[1.0])
        ),
        io.FlushCacheReqInput: MagicMock(return_value="flushed"),
    }


class _Harness:
    def __init__(self, *, controller, declared, native=None):
        self.handlers = _native_handlers()
        self.dispatcher = TypeBasedDispatcher(list(self.handlers.items()))
        self.layout = _LAYOUT
        self.after = MagicMock(side_effect=lambda req, ok, msg: (ok, msg + " +flush"))
        self.waits = 0

        def wait():
            self.waits += 1

        def set_layout(layout):
            self.layout = layout

        ModelRunnerControl(
            runner_name="reference" if native is None else "sglang",
            controller=controller,
            declared=frozenset(declared),
            native=native,
            layout=lambda: self.layout,
            set_layout=set_layout,
            after_weights_update=self.after,
            wait_for_launched_forward=wait,
        ).wrap(self.dispatcher)


class TestExternalControl(CustomTestCase):
    def test_undeclared_op_gets_a_failed_status(self):
        runner = _Recorder()
        h = _Harness(controller=runner, declared={ControlOp.WEIGHTS_UPDATE_FROM_DISK})
        reply = h.dispatcher(io.ReleaseMemoryOccupationReqInput())
        self.assertIsInstance(reply, io.ReleaseMemoryOccupationReqOutput)
        self.assertFalse(reply.success)
        self.assertEqual(
            reply.message,
            "--model-runner reference does not support release_memory_occupation "
            "(control op release; serve with --model-runner sglang)",
        )
        # Replies without a status keep their format.
        self.assertEqual(
            h.dispatcher(io.GetWeightsByNameReqInput(name="x")).parameter, None
        )
        self.assertEqual(runner.calls, [])
        h.handlers[io.ReleaseMemoryOccupationReqInput].assert_not_called()

    def test_declared_op_calls_control_with_args(self):
        runner = _Recorder(ControlResult(message="loaded"))
        h = _Harness(
            controller=runner,
            declared={ControlOp.WEIGHTS_UPDATE_FROM_DISK, ControlOp.WEIGHTS_GET},
        )
        reply = h.dispatcher(
            io.UpdateWeightFromDiskReqInput(model_path="/ckpt", load_format="auto")
        )
        self.assertEqual(
            runner.calls,
            [
                ControlRequest(
                    op=ControlOp.WEIGHTS_UPDATE_FROM_DISK,
                    args={"model_path": "/ckpt", "load_format": "auto"},
                )
            ],
        )
        # SGLang waited for launched steps, then flushed its cache after.
        self.assertEqual(h.waits, 1)
        self.assertTrue(reply.success)
        self.assertEqual(reply.message, "loaded +flush")
        runner.result = ControlResult(value=[[0.5, 0.25]])
        reply = h.dispatcher(io.GetWeightsByNameReqInput(name="w", truncate_size=1))
        self.assertEqual(reply.parameter, [[0.5, 0.25]])
        self.assertEqual(runner.calls[-1].args, {"name": "w", "truncate_size": 1})
        h.handlers[io.UpdateWeightFromDiskReqInput].assert_not_called()

    def test_failure_and_exception_map_to_status(self):
        runner = MagicMock()
        runner.control.side_effect = RuntimeError("disk gone")
        h = _Harness(controller=runner, declared={ControlOp.WEIGHTS_UPDATE_FROM_DISK})
        reply = h.dispatcher(io.UpdateWeightFromDiskReqInput(model_path="/x"))
        self.assertFalse(reply.success)
        self.assertIn("weights_update_from_disk failed: disk gone", reply.message)

    def test_sglang_owned_requests_are_not_routed(self):
        h = _Harness(controller=_Recorder(), declared=())
        self.assertEqual(h.dispatcher(io.FlushCacheReqInput()), "flushed")

    def test_release_narrows_external_control_ops(self):
        declared = RunnerCapabilities(control_ops=tuple(ControlOp))
        narrowed = capabilities_in_this_release(declared, native=False)
        self.assertEqual(
            set(narrowed.control_ops),
            {ControlOp.WEIGHTS_UPDATE_FROM_DISK, ControlOp.WEIGHTS_GET},
        )
        self.assertEqual(
            capabilities_in_this_release(declared, native=True).control_ops,
            tuple(ControlOp),
        )


class TestNativeControl(CustomTestCase):
    def _native(self):
        plugin = SGLangRunnerPlugin(host=MagicMock())
        h = _Harness(
            controller=plugin,
            declared=plugin.describe(None).control_ops,
            native=plugin.control_binding,
        )
        return plugin, h

    def test_native_handler_runs_inside_control(self):
        plugin, h = self._native()
        with patch.object(plugin, "control", wraps=plugin.control) as control:
            reply = h.dispatcher(io.UpdateWeightFromDiskReqInput(model_path="/p"))
        self.assertEqual(reply.message, "ok")
        self.assertEqual(
            control.call_args.args[0].op, ControlOp.WEIGHTS_UPDATE_FROM_DISK
        )
        h.handlers[io.UpdateWeightFromDiskReqInput].assert_called_once()
        # SGLang's after-update work stays inside the native handler.
        h.after.assert_not_called()

    def test_kv_release_and_resume_start_a_new_epoch(self):
        plugin, h = self._native()
        with patch(
            "sglang.srt.model_runner_plugins.native.native_kv_layout",
            return_value=_LAYOUT,
        ):
            h.dispatcher(io.ResumeMemoryOccupationReqInput())
            self.assertEqual(h.layout.epoch, 0)
            h.dispatcher(io.ReleaseMemoryOccupationReqInput(tags=["weights"]))
            h.dispatcher(io.ResumeMemoryOccupationReqInput(tags=["weights"]))
            self.assertEqual(h.layout.epoch, 0)
            h.dispatcher(io.ReleaseMemoryOccupationReqInput())
            h.dispatcher(io.ResumeMemoryOccupationReqInput())
            self.assertEqual(h.layout.epoch, 1)
            self.assertEqual(h.layout.placement(), _LAYOUT.placement())

    def test_refused_release_keeps_the_epoch(self):
        plugin, h = self._native()
        h.handlers[
            io.ReleaseMemoryOccupationReqInput
        ].return_value = io.ReleaseMemoryOccupationReqOutput(
            success=False, message="busy"
        )
        with patch(
            "sglang.srt.model_runner_plugins.native.native_kv_layout",
            return_value=_LAYOUT,
        ):
            reply = h.dispatcher(io.ReleaseMemoryOccupationReqInput())
            h.dispatcher(io.ResumeMemoryOccupationReqInput())
        self.assertEqual(reply.message, "busy")
        self.assertEqual(h.layout.epoch, 0)

    def test_a_new_epoch_keeps_the_sizes(self):
        plugin, h = self._native()
        smaller = KVLayout(page_size=1, num_slots=64)
        with patch(
            "sglang.srt.model_runner_plugins.native.native_kv_layout",
            return_value=smaller,
        ):
            h.dispatcher(io.ReleaseMemoryOccupationReqInput())
            with self.assertRaisesRegex(ValueError, "a new epoch keeps the sizes"):
                h.dispatcher(io.ResumeMemoryOccupationReqInput())


if __name__ == "__main__":
    unittest.main()
