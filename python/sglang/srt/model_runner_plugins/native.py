"""The built-in ``sglang`` model runner plugin.

It declares everything SGLang's own runner supports; execution stays on the
direct path, which a launch with ``--model-runner sglang`` takes unchanged.
"""

from __future__ import annotations

from sglang.srt.model_runner_plugins.api import (
    MODEL_RUNNER_API_VERSION,
    Feature,
    RefusedCombination,
    ResolvedConfig,
    RunnerCapabilities,
)

_NATIVE_CAPABILITIES = RunnerCapabilities.all_supported(
    refused_combinations=(
        # The tokenizer manager refuses the same pair for the Python frontend.
        RefusedCombination(
            features=(Feature.INPUT_EMBEDS, Feature.RADIX_CACHE),
            hint="pass --disable-radix-cache",
        ),
    ),
)


class SGLangRunnerPlugin:
    api_version = MODEL_RUNNER_API_VERSION

    def describe(self, cfg: ResolvedConfig) -> RunnerCapabilities:
        return _NATIVE_CAPABILITIES
