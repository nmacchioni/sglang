"""Model runner lifecycle: phase order, close on failure, typed results, and the
built-in runner's delegation to the scheduler's startup phases."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_runner_plugins.api import (
    AdmissionLimits,
    Capacity,
    FullAttentionLayer,
    GraphBuckets,
    KVBuffer,
    KVKind,
    KVLayout,
    KVSubPool,
    LoadContext,
    LoadResult,
    MemoryNeeds,
    ModelInfo,
    ModelRunnerLifecycle,
    RunnerCapabilities,
    RuntimeConfig,
)
from sglang.srt.model_runner_plugins.native import SGLangRunnerPlugin
from sglang.srt.model_runner_plugins.native_results import native_kv_layout
from sglang.srt.model_runner_plugins.startup import (
    ModelRunnerStartup,
    StartupInputs,
    StartupPhase,
    settle_capacity,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_PHASES = ("load", "allocate", "capture", "finalize")
_LIMITS = AdmissionLimits(
    max_total_num_tokens=4096,
    max_running_requests=64,
    max_req_len=2047,
    max_req_input_len=2042,
)
_LAYER = FullAttentionLayer(
    layer_id=0, k_ptr=4096, v_ptr=8192, k_slot_bytes=256, v_slot_bytes=256
)
_INPUTS = StartupInputs(
    load=LoadContext(model_path="dummy", device="cuda", gpu_id=0),
    runtime=RuntimeConfig(page_size=1, max_total_tokens=4096),
    buckets=GraphBuckets(decode=(1, 2, 4)),
    capacity=Capacity(max_total_tokens=4096),
)


class _FakePlugin:
    """Records each lifecycle call; ``fail_in`` names the phase that raises."""

    api_version = 1

    def __init__(self, *, fail_in=None, results=None):
        self.calls = []
        self.fail_in = fail_in
        self.results = {
            "load": LoadResult(
                model_info=ModelInfo(context_len=2048, vocab_size=32000),
                memory=MemoryNeeds(weight_bytes=1 << 30),
            ),
            "allocate": KVLayout(page_size=1, num_slots=8192, full_attention=(_LAYER,)),
            "capture": None,
            "finalize": KVLayout(
                page_size=1, num_slots=4096, full_attention=(_LAYER,), limits=_LIMITS
            ),
            **(results or {}),
        }

    def describe(self, cfg):
        raise AssertionError("not part of startup")

    def _phase(self, name, arg):
        self.calls.append((name, arg))
        if self.fail_in == name:
            raise RuntimeError(f"{name} failed")
        return self.results[name]

    def load(self, ctx):
        return self._phase("load", ctx)

    def allocate(self, cfg):
        return self._phase("allocate", cfg)

    def capture(self, buckets):
        return self._phase("capture", buckets)

    def finalize(self, capacity):
        return self._phase("finalize", capacity)

    def close(self):
        self.calls.append(("close", None))


def _model_config():
    """The ``ModelConfig`` facts the resolved config reads, without a checkpoint."""
    return SimpleNamespace(
        hf_config=LlamaConfig(architectures=["LlamaForCausalLM"]),
        is_multimodal=False,
        is_generation=True,
        is_draft_model=False,
        is_hybrid_swa=False,
        sliding_window_size=None,
        attention_chunk_size=None,
        attention_arch=AttentionArch.MHA,
        is_encoder_decoder=False,
        linear_attn_registry_result=None,
    )


def _names(plugin):
    return [name for name, _ in plugin.calls]


class TestStartupOrder(CustomTestCase):
    def test_phases_run_in_native_order_with_their_inputs(self):
        plugin = _FakePlugin()
        startup = ModelRunnerStartup(runner_name="fake", plugin=plugin)

        final = startup.run(_INPUTS)

        self.assertIsInstance(plugin, ModelRunnerLifecycle)
        self.assertEqual(
            plugin.calls,
            [
                ("load", _INPUTS.load),
                ("allocate", _INPUTS.runtime),
                ("capture", _INPUTS.buckets),
                ("finalize", _INPUTS.capacity),
            ],
        )
        self.assertEqual(final.limits, _LIMITS)
        self.assertEqual(startup.layout, final)
        self.assertEqual(startup.phase, StartupPhase.FINALIZED)

    def test_close_runs_once_on_shutdown(self):
        plugin = _FakePlugin()
        startup = ModelRunnerStartup(runner_name="fake", plugin=plugin)
        startup.run(_INPUTS)

        startup.close()
        startup.close()

        self.assertEqual(_names(plugin), [*_PHASES, "close"])
        with self.assertRaisesRegex(RuntimeError, "already ran"):
            startup.run(_INPUTS)

    def test_close_before_load_does_not_reach_the_plugin(self):
        plugin = _FakePlugin()
        ModelRunnerStartup(runner_name="fake", plugin=plugin).close()
        self.assertEqual(plugin.calls, [])

    def test_failure_in_each_phase_closes_and_stops(self):
        for index, phase in enumerate(_PHASES):
            with self.subTest(phase=phase):
                plugin = _FakePlugin(fail_in=phase)
                startup = ModelRunnerStartup(runner_name="fake", plugin=plugin)

                with self.assertRaisesRegex(RuntimeError, f"{phase} failed"):
                    startup.run(_INPUTS)

                self.assertEqual(_names(plugin), [*_PHASES[: index + 1], "close"])
                self.assertEqual(startup.phase, StartupPhase.CLOSED)
                startup.close()
                self.assertEqual(_names(plugin).count("close"), 1)

    def test_a_failing_close_does_not_mask_the_startup_error(self):
        class _CloseFails(_FakePlugin):
            def close(self):
                raise RuntimeError("close failed")

        startup = ModelRunnerStartup(
            runner_name="fake", plugin=_CloseFails(fail_in="allocate")
        )
        with self.assertLogs("sglang.srt.model_runner_plugins.startup", "ERROR"):
            with self.assertRaisesRegex(RuntimeError, "allocate failed"):
                startup.run(_INPUTS)


class TestResultValidation(CustomTestCase):
    def _run(self, **results):
        plugin = _FakePlugin(results=results)
        startup = ModelRunnerStartup(runner_name="fake", plugin=plugin)
        return plugin, startup

    def test_wrong_result_types_are_refused_and_close(self):
        for phase, bad in (
            ("load", {"context_len": 2048}),
            ("allocate", None),
            ("finalize", _LIMITS),
        ):
            with self.subTest(phase=phase):
                plugin, startup = self._run(**{phase: bad})
                with self.assertRaisesRegex(
                    TypeError, rf"'fake' {phase}\(\) returned .*; expected"
                ):
                    startup.run(_INPUTS)
                self.assertEqual(_names(plugin)[-1], "close")

    def test_final_layout_needs_limits(self):
        plugin, startup = self._run(finalize=KVLayout(page_size=1, num_slots=4096))
        with self.assertRaisesRegex(ValueError, "limits set"):
            startup.run(_INPUTS)
        self.assertEqual(_names(plugin)[-1], "close")

    def test_finalize_may_resize_but_never_move(self):
        moved = FullAttentionLayer(
            layer_id=0, k_ptr=1 << 20, v_ptr=8192, k_slot_bytes=256, v_slot_bytes=256
        )
        for final, error in (
            (
                KVLayout(
                    page_size=1, num_slots=4096, full_attention=(moved,), limits=_LIMITS
                ),
                "moved KV buffers within an epoch",
            ),
            (
                KVLayout(
                    page_size=1,
                    num_slots=4096,
                    full_attention=(_LAYER,),
                    sub_pools=(
                        KVSubPool(
                            kind=KVKind.SLIDING_WINDOW,
                            num_items=1024,
                            buffers=(
                                KVBuffer(
                                    layer_id=1, name="k", ptr=1 << 30, item_bytes=256
                                ),
                            ),
                            window=512,
                        ),
                    ),
                    limits=_LIMITS,
                ),
                "moved KV buffers within an epoch",
            ),
            (
                KVLayout(
                    epoch=1,
                    page_size=1,
                    num_slots=4096,
                    full_attention=(_LAYER,),
                    limits=_LIMITS,
                ),
                "epoch or page size",
            ),
        ):
            with self.subTest(error=error):
                _, startup = self._run(finalize=final)
                with self.assertRaisesRegex(ValueError, error):
                    startup.run(_INPUTS)

        # A smaller or larger slot count at the same addresses is a resize.
        for num_slots in (2048, 16384):
            _, startup = self._run(
                finalize=KVLayout(
                    page_size=1,
                    num_slots=num_slots,
                    full_attention=(_LAYER,),
                    limits=_LIMITS,
                )
            )
            self.assertEqual(startup.run(_INPUTS).num_slots, num_slots)

    def test_struct_invariants(self):
        cases = (
            (lambda: ModelInfo(context_len=0, vocab_size=10), "positive context_len"),
            (lambda: MemoryNeeds(weight_bytes=-1), "must not be negative"),
            (lambda: KVLayout(page_size=0, num_slots=1), "page_size >= 1"),
            (
                lambda: KVLayout(
                    page_size=1, num_slots=1, full_attention=(_LAYER,) * 2
                ),
                "lists a layer twice",
            ),
            (
                lambda: AdmissionLimits(
                    max_total_num_tokens=0,
                    max_running_requests=1,
                    max_req_len=1,
                    max_req_input_len=1,
                ),
                "positive max_total_num_tokens",
            ),
            (
                lambda: AdmissionLimits(
                    max_total_num_tokens=10,
                    max_running_requests=1,
                    max_req_len=4,
                    max_req_input_len=5,
                ),
                "max_req_input_len <= max_req_len",
            ),
        )
        for build, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(ValueError, error):
                    build()


class _FakeHost:
    """The scheduler's native startup phases, recorded; ``fail_in`` raises."""

    def __init__(self, *, fail_in=None):
        self.calls = []
        self.fail_in = fail_in
        self.draft_worker = None
        pool = SimpleNamespace(page_size=1, size=4096)
        self.tp_worker = SimpleNamespace(
            model_config=SimpleNamespace(
                context_len=2048, vocab_size=32000, is_generation=True
            ),
            preloaded_weights_bytes=0,
            model_runner=SimpleNamespace(
                weight_load_mem_usage=1.5, token_to_kv_pool=pool
            ),
            get_worker_info=lambda: (
                4096,
                16384,
                64,
                None,
                2047,
                2042,
                7,
                "cuda",
                "stream",
                65,
                2048,
                4096,
            ),
        )

    def _phase(self, name):
        self.calls.append(name)
        if self.fail_in == name:
            raise RuntimeError(f"{name} failed")

    def load_model_workers(self):
        self._phase("load_model_workers")

    def alloc_model_memory_pools(self):
        self._phase("alloc_model_memory_pools")

    def capture_model_graphs(self):
        self._phase("capture_model_graphs")

    def finalize_model_capacity(self, *, agree_stages=True):
        self._phase("finalize_model_capacity")
        if agree_stages:
            self.agree_stage_capacity()

    def agree_stage_capacity(self):
        self._phase("agree_stage_capacity")


_HOST_PHASES = (
    "load_model_workers",
    "alloc_model_memory_pools",
    "capture_model_graphs",
    "finalize_model_capacity",
    # SGLang's, after the runner's finalize: PP stages agree, then limits.
    "agree_stage_capacity",
)


def _native_startup(host):
    plugin = SGLangRunnerPlugin(host=host)
    startup = ModelRunnerStartup(
        runner_name="sglang",
        plugin=plugin,
        settle=lambda final: settle_capacity(host, final),
    )
    return startup, plugin


class TestNativeRunner(CustomTestCase):
    def test_delegates_to_the_scheduler_phases_in_order(self):
        host = _FakeHost()
        startup, _ = _native_startup(host)

        final = startup.run(_INPUTS)

        self.assertEqual(host.calls, list(_HOST_PHASES))
        self.assertEqual(
            startup.load_result.model_info,
            ModelInfo(context_len=2048, vocab_size=32000),
        )
        self.assertEqual(startup.load_result.memory.weight_bytes, int(1.5 * (1 << 30)))
        # The limits are get_worker_info's, so both scheduler paths agree.
        self.assertEqual(final.limits, _LIMITS)
        # Not a slot-row MHA pool: no full-attention layers are described.
        self.assertEqual(final.full_attention, ())

    def test_failure_in_each_native_phase_closes_the_runner(self):
        for index, phase in enumerate(_HOST_PHASES):
            with self.subTest(phase=phase):
                host = _FakeHost(fail_in=phase)
                startup, plugin = _native_startup(host)

                with self.assertRaisesRegex(RuntimeError, f"{phase} failed"):
                    startup.run(_INPUTS)

                self.assertEqual(host.calls, list(_HOST_PHASES[: index + 1]))
                # Closed: the runner no longer drives the scheduler's phases.
                with self.assertRaisesRegex(RuntimeError, "after close"):
                    plugin.load(_INPUTS.load)

    def test_describe_only_instance_has_no_lifecycle(self):
        with self.assertRaisesRegex(RuntimeError, "inside the scheduler"):
            SGLangRunnerPlugin().allocate(_INPUTS.runtime)

    def test_full_attention_layout_of_a_slot_row_mha_pool(self):
        pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
        pool.page_size, pool.size, pool.use_hnd = 1, 7, False
        pool.model_layer_ids = (3, 4)
        # [rows, heads, head_dim] slot rows, as _create_buffers_normal lays them out.
        pool.k_buffer = [torch.zeros(8, 2, 4, dtype=torch.float16) for _ in range(2)]
        pool.v_buffer = [torch.zeros(8, 2, 8, dtype=torch.float16) for _ in range(2)]
        tp_worker = SimpleNamespace(model_runner=SimpleNamespace(token_to_kv_pool=pool))

        layout = native_kv_layout(tp_worker=tp_worker, with_limits=False)

        self.assertEqual(layout.num_slots, 7)
        self.assertIsNone(layout.limits)
        self.assertEqual(
            layout.full_attention,
            tuple(
                FullAttentionLayer(
                    layer_id=layer_id,
                    k_ptr=k.data_ptr(),
                    v_ptr=v.data_ptr(),
                    k_slot_bytes=2 * 4 * 2,
                    v_slot_bytes=2 * 8 * 2,
                )
                for layer_id, k, v in zip((3, 4), pool.k_buffer, pool.v_buffer)
            ),
        )


def _mha_pool(*, size, layer_ids, page_size=1):
    pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
    pool.page_size, pool.size, pool.use_hnd = page_size, size, False
    pool.model_layer_ids = tuple(layer_ids)
    rows = size + page_size
    pool.k_buffer = [torch.zeros(rows, 1, 4, dtype=torch.float16) for _ in layer_ids]
    pool.v_buffer = [torch.zeros(rows, 1, 4, dtype=torch.float16) for _ in layer_ids]
    return pool


class _MambaPool:
    """Two state tensors per layer, as MambaPool registers them for PD."""

    def __init__(self, *, slots, layer_ids):
        self.layer_ids = layer_ids
        self.tensors = [
            torch.zeros(slots, 3, dtype=torch.float32)
            for _ in range(2 * len(layer_ids))
        ]

    def get_contiguous_buf_infos(self):
        return (
            [t.data_ptr() for t in self.tensors],
            [t.nbytes for t in self.tensors],
            [t[0].nbytes for t in self.tensors],
        )

    def get_state_layer_ids(self):
        # Tensor-major, then layer, as MambaPool flattens its state list.
        return list(self.layer_ids) * 2


class TestNativeKVDescription(CustomTestCase):
    """The built-in runner describes the pools its steps index, and keeps the
    direct path for a pool it cannot describe rather than run a Step whose
    KVTables leave out KV the forward reads."""

    def _layout(self, model_runner):
        model_runner.token_to_kv_pool.page_size = 1
        return native_kv_layout(
            tp_worker=SimpleNamespace(model_runner=model_runner), with_limits=False
        )

    def test_sliding_window_layers_are_a_sub_pool_with_their_own_slot_count(self):
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
        from sglang.srt.model_runner_plugins.native_kv import NativeKV

        pool = SWAKVPool.__new__(SWAKVPool)
        pool.full_kv_pool = _mha_pool(size=16, layer_ids=(5,))
        pool.swa_kv_pool = _mha_pool(size=8, layer_ids=(0, 1, 2))
        pool.full_to_swa_index_mapping = torch.zeros(17, dtype=torch.int64)
        pool.size = 16
        runner = SimpleNamespace(token_to_kv_pool=pool, sliding_window_size=512)

        layout = self._layout(runner)

        self.assertEqual([layer.layer_id for layer in layout.full_attention], [5])
        (window,) = layout.sub_pools
        self.assertEqual((window.kind, window.window), (KVKind.SLIDING_WINDOW, 512))
        # The window pool's own slots, its padding page included.
        self.assertEqual(window.num_items, 9)
        self.assertEqual(
            [(b.layer_id, b.name) for b in window.buffers],
            [(i, n) for i in (0, 1, 2) for n in ("k", "v")],
        )
        self.assertEqual(window.buffers[2].ptr, pool.swa_kv_pool.k_buffer[1].data_ptr())
        # A window kept per request has no full-to-window slot map to hand over.
        pool.full_to_swa_index_mapping = None
        self.assertEqual(NativeKV(runner).refusal, "a sliding window kept per request")
        self.assertEqual(self._layout(runner).sub_pools, ())

    def test_recurrent_state_is_a_sub_pool_of_request_slots(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
        from sglang.srt.model_runner_plugins.native_kv import NativeKV

        pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
        pool.full_kv_pool = _mha_pool(size=16, layer_ids=(3,))
        pool.mamba_pool = _MambaPool(slots=5, layer_ids=(0, 1))
        pool.use_mla, pool.size = False, 16
        state = SimpleNamespace(ckpt_pool=None, own_state=None)
        runner = SimpleNamespace(
            token_to_kv_pool=pool,
            token_to_kv_pool_allocator=SimpleNamespace(state=state),
            req_to_token_pool=SimpleNamespace(state_sub_pool="mamba"),
            runner_ids=SimpleNamespace(
                state_ids=lambda sub_pool: SimpleNamespace(write=None)
            ),
        )

        (state_pool,) = self._layout(runner).sub_pools

        self.assertEqual(
            (state_pool.kind, state_pool.num_items), (KVKind.RECURRENT_STATE, 5)
        )
        self.assertEqual(
            [(b.layer_id, b.name) for b in state_pool.buffers],
            [(0, "state0"), (1, "state0"), (0, "state1"), (1, "state1")],
        )
        self.assertEqual(state_pool.buffers[0].item_bytes, 12)
        # Quantized checkpoints, and state slots behind a remapping table, have
        # no description: those models keep the direct path.
        state.ckpt_pool = object()
        self.assertIn("quantized checkpoints", NativeKV(runner).refusal)
        state.ckpt_pool = None
        runner.runner_ids = SimpleNamespace(
            state_ids=lambda sub_pool: SimpleNamespace(write=lambda ids: ids)
        )
        self.assertEqual(NativeKV(runner).refusal, "remapped recurrent-state slots")

    def test_an_undescribed_pool_keeps_the_direct_path(self):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
        from sglang.srt.model_runner_plugins.native_kv import NativeKV

        pool = MLATokenToKVPool.__new__(MLATokenToKVPool)
        runner = SimpleNamespace(token_to_kv_pool=pool)
        self.assertEqual(NativeKV(runner).refusal, "KV pool MLATokenToKVPool")


def _stub_scheduler(startup):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model_runner_startup = startup
    scheduler.tp_worker = _FakeHost().tp_worker
    scheduler.tp_worker.random_seed = 7
    scheduler.tp_worker.device = "cuda"
    scheduler.tp_worker.model_runner.forward_stream = "stream"
    return scheduler


_WORKER_INFO_FIELDS = (
    "max_total_num_tokens",
    "max_prefill_tokens",
    "max_running_requests",
    "max_queued_requests",
    "max_req_len",
    "max_req_input_len",
    "random_seed",
    "device",
    "forward_stream",
)


class TestSchedulerWiring(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            max_prefill_tokens=16384, max_queued_requests=None
        )
        override.install()
        self.addCleanup(override.restore)

    @patch(
        "sglang.srt.model_runner_plugins.registry.entry_points",
        side_effect=AssertionError("runner plugins enumerated without the flag"),
    )
    def test_omitted_flag_creates_no_lifecycle(self, _entry_points):
        self.assertIsNone(
            ModelRunnerStartup.maybe_create(
                native_host=_FakeHost(), model_config=_model_config()
            )
        )

    def test_native_flag_binds_the_scheduler_phases(self):
        with get_context().override_server_args(model_runner="sglang"):
            host = _FakeHost()
            startup = ModelRunnerStartup.maybe_create(
                native_host=host, model_config=_model_config()
            )
        self.assertEqual(startup.runner_name, "sglang")
        self.assertFalse(startup.is_external)
        startup.run(_INPUTS)
        self.assertEqual(host.calls, list(_HOST_PHASES))

    def test_external_runner_needs_the_step_methods(self):
        class _DescribeOnly:
            api_version = 1

            def describe(self, cfg):
                return RunnerCapabilities()

        class _Executor(_FakePlugin):
            def describe(self, cfg):
                return RunnerCapabilities()

            def plan(self, step): ...

            def execute(self, plan): ...

            def project(self, req): ...

            def release(self, out, after): ...

        for plugin, external in ((_DescribeOnly(), None), (_Executor(), True)):
            runner = SimpleNamespace(name="alpha", is_native=False, plugin=plugin)
            with (
                patch(
                    "sglang.srt.model_runner_plugins.startup.load_model_runner",
                    return_value=runner,
                ),
                get_context().override_server_args(model_runner="alpha"),
            ):
                if external is None:
                    with self.assertRaisesRegex(TypeError, "lacks the lifecycle"):
                        ModelRunnerStartup.maybe_create(
                            native_host=_FakeHost(), model_config=_model_config()
                        )
                    continue
                startup = ModelRunnerStartup.maybe_create(
                    native_host=_FakeHost(), model_config=_model_config()
                )
            self.assertTrue(startup.is_external)

    def test_inputs_carry_the_published_limits(self):
        with (
            get_context().override_server_args(
                max_total_tokens=12345, max_running_requests=17, page_size=16
            ),
            get_parallel().override(tp_rank=0, pp_rank=0),
        ):
            inputs = StartupInputs.from_published()
        self.assertEqual(inputs.runtime.max_total_tokens, 12345)
        self.assertEqual(inputs.runtime.max_running_requests, 17)
        self.assertEqual(inputs.runtime.page_size, 16)
        self.assertEqual(inputs.capacity.max_total_tokens, 12345)
        self.assertEqual(inputs.capacity.max_running_requests, 17)

    def test_worker_info_matches_on_both_paths(self):
        direct = _stub_scheduler(startup=None)
        direct.init_worker_info()

        startup, _ = _native_startup(_FakeHost())
        startup.run(_INPUTS)
        via_plugin = _stub_scheduler(startup=startup)
        # The typed path must not call get_worker_info again.
        via_plugin.tp_worker.get_worker_info = None
        via_plugin.init_worker_info()

        for field in _WORKER_INFO_FIELDS:
            with self.subTest(field=field):
                self.assertEqual(getattr(via_plugin, field), getattr(direct, field))

    def test_close_model_runner_is_a_no_op_without_the_flag(self):
        _stub_scheduler(startup=None).close_model_runner()
        plugin = _FakePlugin()
        startup = ModelRunnerStartup(runner_name="fake", plugin=plugin)
        startup.run(_INPUTS)
        _stub_scheduler(startup=startup).close_model_runner()
        self.assertEqual(_names(plugin)[-1], "close")


if __name__ == "__main__":
    unittest.main()
