"""Launch and admission checks for ``--model-runner``, read from the published config."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.model_executor.cuda_graph_config import Backend, Phase
from sglang.srt.model_runner_plugins.api import ResolvedConfig, RunnerCapabilities
from sglang.srt.model_runner_plugins.refusal import (
    RequestOptions,
    Uses,
    admission_refusals,
    launch_refusals,
    launch_uses,
    refusal_message,
    request_uses,
)
from sglang.srt.model_runner_plugins.registry import (
    LoadedModelRunner,
    load_model_runner,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_lora,
    get_memory,
    get_model,
    get_parallel,
    get_schedule,
    get_spec,
    process_model_config,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req


def check_model_runner_at_launch() -> None:
    """Refuse an unservable launch before any subprocess starts.

    A no-op without ``--model-runner``: nothing is enumerated or imported.
    """
    name = get_exec().kernel.model_runner
    if name is None:
        return
    runner = load_model_runner(name)
    cfg = resolved_config(process_model_config())
    reasons = launch_refusals(_describe(runner, cfg), cfg)
    if reasons:
        raise ValueError(refusal_message(name, reasons))
    if not runner.is_native:
        raise RuntimeError(external_runner_unavailable_message(name))


def external_runner_unavailable_message(name: str) -> str:
    return (
        f"--model-runner {name}: execution through external model runners is "
        "not available yet in this SGLang release. Omit --model-runner or "
        "pass --model-runner sglang."
    )


def resolved_config(model_config: ModelConfig) -> ResolvedConfig:
    parallel = get_parallel()
    graphs = get_exec().graph.cuda_graph_config
    return ResolvedConfig(
        model_path=get_model().model_path,
        architectures=tuple(model_config.hf_config.architectures or ()),
        tp_size=parallel.tp_size,
        pp_size=parallel.pp_size,
        attn_dp_size=parallel.attn_dp_size,
        moe_ep_size=parallel.moe_ep_size,
        attn_cp_size=parallel.attn_cp_size,
        dcp_size=parallel.dcp_size,
        cuda_graph_phases=tuple(
            phase
            for phase in Phase.ALL
            if graphs is not None and graphs[phase].backend != Backend.DISABLED
        ),
        overlap_schedule=not get_schedule().disable_overlap_schedule,
        speculative_algorithm=get_spec().speculative_algorithm,
        lora=bool(get_lora().enable_lora),
        multimodal=model_config.is_multimodal,
        pooling=not model_config.is_generation,
        disaggregation_mode=get_disagg().disaggregation_mode,
        hicache=get_memory().enable_hierarchical_cache,
        memory_saver=get_exec().features.enable_memory_saver,
        radix_cache=not get_memory().disable_radix_cache,
    )


class ModelRunnerAdmission:
    """Per-request refusal at the scheduler's admission boundary."""

    def __init__(
        self, *, runner_name: str, capabilities: RunnerCapabilities, launch: Uses
    ):
        self._runner_name = runner_name
        self._capabilities = capabilities
        self._launch = launch

    @classmethod
    def maybe_create(
        cls, *, model_config: ModelConfig
    ) -> Optional[ModelRunnerAdmission]:
        """None without ``--model-runner``, so the default path checks nothing."""
        name = get_exec().kernel.model_runner
        if name is None:
            return None
        cfg = resolved_config(model_config)
        return cls(
            runner_name=name,
            capabilities=_describe(load_model_runner(name), cfg),
            launch=launch_uses(cfg),
        )

    def check(self, req: Req) -> Optional[str]:
        """The refusal message for ``req``, or None when the runner serves it."""
        return self.check_options(RequestOptions.of_req(req))

    def check_options(self, opts: RequestOptions) -> Optional[str]:
        reasons = admission_refusals(
            self._capabilities, launch=self._launch, request=request_uses(opts)
        )
        return refusal_message(self._runner_name, reasons) if reasons else None


def _describe(runner: LoadedModelRunner, cfg: ResolvedConfig) -> RunnerCapabilities:
    capabilities = runner.plugin.describe(cfg)
    if not isinstance(capabilities, RunnerCapabilities):
        raise TypeError(
            f"Model runner {runner.name!r} describe() returned "
            f"{type(capabilities).__name__}; expected RunnerCapabilities."
        )
    return capabilities
