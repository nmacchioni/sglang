"""Selection of --model-runner: flag, metadata-only discovery, and the launch check."""

import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from transformers import LlamaConfig

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.environ import envs
from sglang.srt.model_runner_plugins.api import MODEL_RUNNERS_GROUP
from sglang.srt.model_runner_plugins.registry import load_model_runner
from sglang.srt.model_runner_plugins.selection import (
    ModelRunnerAdmission,
    check_model_runner_at_launch,
)
from sglang.srt.runtime_context import get_context, get_exec
from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_PLUGIN_SOURCE = """
from sglang.srt.model_runner_plugins.api import RunnerCapabilities

class Plugin:
    api_version = {api_version}

    def describe(self, cfg):
        return {capabilities}

def create_plugin():
    return Plugin()
"""

# Importing this module is the failure: discovery must never reach it.
_POISON_SOURCE = "raise ImportError('an unselected model runner was imported')\n"


class _FakeDistributions:
    """Installed distributions on a temporary sys.path entry, read through real
    ``importlib.metadata``, so discovery sees only what a pip install would show."""

    def __init__(self):
        self.root = Path(tempfile.mkdtemp())
        self.modules = []
        sys.path.insert(0, str(self.root))

    def add(self, dist, runner_name, source):
        module = f"fake_runner_{uuid.uuid4().hex}"
        (self.root / f"{module}.py").write_text(source)
        # Wheel naming: the stem before the first "-" is the distribution name.
        info = self.root / f"{dist.replace('-', '_')}-1.0.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {dist}\nVersion: 1.0\n"
        )
        (info / "entry_points.txt").write_text(
            f"[{MODEL_RUNNERS_GROUP}]\n{runner_name} = {module}:create_plugin\n"
        )
        self.modules.append(module)
        return module

    def plugin(self, dist, runner_name, *, api_version=1, capabilities=None):
        return self.add(
            dist,
            runner_name,
            _PLUGIN_SOURCE.format(
                api_version=api_version,
                capabilities=capabilities or "RunnerCapabilities()",
            ),
        )

    def poison(self, dist, runner_name):
        return self.add(dist, runner_name, _POISON_SOURCE)

    def close(self):
        sys.path.remove(str(self.root))
        for module in self.modules:
            sys.modules.pop(module, None)
        shutil.rmtree(self.root, ignore_errors=True)


class _DistsCase(CustomTestCase):
    def setUp(self):
        self.dists = _FakeDistributions()
        self.addCleanup(self.dists.close)
        plugins = envs.SGLANG_PLUGINS.override(None)
        plugins.__enter__()
        self.addCleanup(plugins.__exit__, None, None, None)


class TestDiscovery(_DistsCase):
    def test_only_the_selected_provider_is_imported(self):
        selected = self.dists.plugin("fake-alpha", "alpha")
        unselected = self.dists.poison("fake-beta", "beta")

        runner = load_model_runner("alpha")

        self.assertEqual(runner.distribution, "fake-alpha")
        self.assertFalse(runner.is_native)
        self.assertIn(selected, sys.modules)
        self.assertNotIn(unselected, sys.modules)

    def test_native_selection_imports_no_installed_runner(self):
        unrelated = self.dists.poison("fake-beta", "beta")

        runner = load_model_runner("sglang")

        self.assertTrue(runner.is_native)
        self.assertIsNone(runner.distribution)
        self.assertNotIn(unrelated, sys.modules)

    def test_reserved_name_fails_only_when_it_is_selected(self):
        squatter = self.dists.poison("fake-squatter", "sglang")
        self.dists.plugin("fake-alpha", "alpha")

        # The squatter is irrelevant to another runner, and never imported.
        self.assertEqual(load_model_runner("alpha").name, "alpha")
        with self.assertRaisesRegex(RuntimeError, "reserved.*fake-squatter"):
            load_model_runner("sglang")
        # SGLANG_PLUGINS hides it, which is the documented way out.
        with envs.SGLANG_PLUGINS.override("alpha"):
            self.assertTrue(load_model_runner("sglang").is_native)
        self.assertNotIn(squatter, sys.modules)

    def test_two_providers_of_the_selected_name_fail(self):
        first = self.dists.poison("fake-one", "alpha")
        second = self.dists.poison("fake-two", "alpha")

        with self.assertRaisesRegex(RuntimeError, "fake-one, fake-two"):
            load_model_runner("alpha")
        self.assertNotIn(first, sys.modules)
        self.assertNotIn(second, sys.modules)

    def test_allowlist_filters_visible_runners(self):
        module = self.dists.plugin("fake-alpha", "alpha")

        with envs.SGLANG_PLUGINS.override("other"):
            with self.assertRaisesRegex(ValueError, "SGLANG_PLUGINS does not list"):
                load_model_runner("alpha")
        self.assertNotIn(module, sys.modules)
        with envs.SGLANG_PLUGINS.override("other, alpha"):
            self.assertEqual(load_model_runner("alpha").distribution, "fake-alpha")

    def test_unknown_name_lists_installed_runners_without_importing(self):
        module = self.dists.poison("fake-beta", "beta")

        with self.assertRaisesRegex(ValueError, "installed: beta"):
            load_model_runner("alpha")
        self.assertNotIn(module, sys.modules)

    def test_api_version_mismatch_fails(self):
        self.dists.plugin("fake-alpha", "alpha", api_version=2)

        with self.assertRaisesRegex(RuntimeError, "API version 2.*requires version 1"):
            load_model_runner("alpha")

    def test_factory_must_return_a_plugin(self):
        self.dists.add("fake-alpha", "alpha", "def create_plugin():\n    return 1\n")

        with self.assertRaisesRegex(TypeError, "lacks api_version or describe"):
            load_model_runner("alpha")


def _model_config(*, multimodal=False, generation=True):
    """The ``ModelConfig`` facts the resolved config reads, without a checkpoint."""
    return SimpleNamespace(
        hf_config=LlamaConfig(architectures=["LlamaForCausalLM"]),
        is_multimodal=multimodal,
        is_generation=generation,
        is_draft_model=False,
        is_hybrid_swa=False,
        sliding_window_size=None,
        attention_chunk_size=None,
        attention_arch=AttentionArch.MHA,
        is_encoder_decoder=False,
        linear_attn_registry_result=None,
    )


class TestFlag(CustomTestCase):
    def test_cli_flag_and_engine_kwarg_reach_the_exec_bag(self):
        cli = prepare_server_args(["--model-path", "dummy", "--model-runner", "alpha"])
        self.assertEqual(cli.model_runner, "alpha")
        # Engine(**kwargs) builds ServerArgs(**kwargs).
        self.assertEqual(
            ServerArgs(model_path="dummy", model_runner="alpha").model_runner, "alpha"
        )
        self.assertIsNone(ServerArgs(model_path="dummy").model_runner)
        with get_context().override_server_args(model_runner="alpha"):
            self.assertEqual(get_exec().kernel.model_runner, "alpha")

    @patch(
        "sglang.srt.model_runner_plugins.registry.entry_points",
        side_effect=AssertionError("runner plugins enumerated without the flag"),
    )
    def test_omitted_flag_enumerates_no_runner(self, _entry_points):
        with get_context().override_server_args(_model_config=_model_config()):
            check_model_runner_at_launch()
            self.assertIsNone(
                ModelRunnerAdmission.maybe_create(model_config=_model_config())
            )


class TestLaunchCheck(_DistsCase):
    def _launch(self, **fields):
        with get_context().override_server_args(
            _model_config=_model_config(multimodal=True), **fields
        ):
            check_model_runner_at_launch()

    def test_refusal_lists_every_reason(self):
        self.dists.plugin("fake-alpha", "alpha")

        with self.assertRaises(ValueError) as ctx:
            self._launch(
                model_runner="alpha",
                tp_size=2,
                disable_overlap_schedule=False,
                enable_memory_saver=True,
            )
        message = str(ctx.exception)
        self.assertTrue(
            message.startswith("--model-runner alpha does not support: "), message
        )
        for reason in (
            "tensor parallelism (tp_size=2)",
            "overlap scheduling (pass --disable-overlap-schedule)",
            "memory saver (drop --enable-memory-saver)",
            "multimodal inputs (pass --disable-multimodal)",
        ):
            self.assertIn(reason, message)

    def test_external_runner_is_held_to_this_release_whatever_it_declares(self):
        self.dists.plugin(
            "fake-alpha",
            "alpha",
            capabilities="RunnerCapabilities.all_supported()",
        )

        with self.assertRaises(ValueError) as ctx:
            self._launch(model_runner="alpha", tp_size=2)
        message = str(ctx.exception)
        self.assertIn("tensor parallelism (tp_size=2)", message)
        self.assertIn("multimodal inputs", message)

    def test_external_runner_accepts_a_tp1_eager_text_launch(self):
        self.dists.plugin("fake-alpha", "alpha")

        with get_context().override_server_args(
            _model_config=_model_config(),
            model_runner="alpha",
            disable_overlap_schedule=True,
            disable_decode_cuda_graph=True,
            disable_prefill_cuda_graph=True,
        ):
            check_model_runner_at_launch()

    def test_native_runner_accepts_the_launch(self):
        self._launch(
            model_runner="sglang",
            tp_size=2,
            disable_overlap_schedule=False,
            enable_memory_saver=True,
            enable_hierarchical_cache=True,
        )


if __name__ == "__main__":
    unittest.main()
