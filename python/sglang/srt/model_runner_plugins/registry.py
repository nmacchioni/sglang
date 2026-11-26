"""Selection of the model runner named by ``--model-runner``.

Discovery reads entry point metadata only and imports just the selected
provider, so an unrelated installed package cannot affect a launch. It runs
only when the flag names a runner other than the built-in one.
"""

from __future__ import annotations

import logging
from importlib.metadata import EntryPoint, entry_points
from typing import Optional

import msgspec

from sglang.srt.environ import envs
from sglang.srt.model_runner_plugins.api import (
    MODEL_RUNNER_API_VERSION,
    MODEL_RUNNERS_GROUP,
    NATIVE_MODEL_RUNNER,
    ModelRunnerPlugin,
)

logger = logging.getLogger(__name__)


class LoadedModelRunner(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    plugin: ModelRunnerPlugin
    # None for the built-in runner.
    distribution: Optional[str] = None

    @property
    def is_native(self) -> bool:
        return self.name == NATIVE_MODEL_RUNNER


def load_model_runner(name: str, *, explicit: bool = True) -> LoadedModelRunner:
    """Resolve ``name`` to a plugin, importing at most that one provider.

    ``explicit=False``: the built-in runner selected by default, without the
    flag; no entry point is enumerated then."""
    if name == NATIVE_MODEL_RUNNER and not explicit:
        from sglang.srt.model_runner_plugins.native import SGLangRunnerPlugin

        return LoadedModelRunner(name=name, plugin=SGLangRunnerPlugin())
    candidates = _visible_entry_points(name)
    if name == NATIVE_MODEL_RUNNER:
        if candidates:
            raise RuntimeError(
                f"Model runner name {name!r} is reserved for the built-in runner, "
                f"but {_providers(candidates)} also register it in "
                f"{MODEL_RUNNERS_GROUP!r}. Uninstall that package or exclude it "
                "with SGLANG_PLUGINS."
            )
        from sglang.srt.model_runner_plugins.native import SGLangRunnerPlugin

        return LoadedModelRunner(name=name, plugin=SGLangRunnerPlugin())
    if not candidates:
        raise ValueError(_unknown_runner_message(name))
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple distributions register model runner {name!r}: "
            f"{_providers(candidates)}. Uninstall all but one."
        )
    return _load_entry_point(name, candidates[0])


def _visible_entry_points(name: str) -> list[EntryPoint]:
    allowed = _allowlist()
    if allowed is not None and name not in allowed:
        return []
    return [ep for ep in entry_points(group=MODEL_RUNNERS_GROUP) if ep.name == name]


def _allowlist() -> Optional[frozenset[str]]:
    """``SGLANG_PLUGINS`` names, shared with general plugins; None when unset."""
    raw = envs.SGLANG_PLUGINS.get()
    if not raw:
        return None
    return frozenset(x.strip() for x in raw.split(",") if x.strip())


def _unknown_runner_message(name: str) -> str:
    allowed = _allowlist()
    installed = sorted(
        {ep.name for ep in entry_points(group=MODEL_RUNNERS_GROUP)}
        - {NATIVE_MODEL_RUNNER}
    )
    message = (
        f"Unknown model runner {name!r} for --model-runner. Built in: "
        f"{NATIVE_MODEL_RUNNER!r} (the default); "
        f"installed: {', '.join(installed) or 'none'}."
    )
    if allowed is not None and name in installed:
        message += " It is installed but SGLANG_PLUGINS does not list it."
    return message


def _load_entry_point(name: str, entry_point: EntryPoint) -> LoadedModelRunner:
    provider = _provider(entry_point)
    try:
        factory = entry_point.load()
        if not callable(factory):
            raise TypeError("the entry point must resolve to a callable factory")
        plugin = factory()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model runner {name!r} from {provider}: {exc}"
        ) from exc

    if not isinstance(plugin, ModelRunnerPlugin):
        raise TypeError(
            f"Model runner {name!r} factory from {provider} returned "
            f"{type(plugin).__name__}, which lacks api_version or describe()."
        )
    if plugin.api_version != MODEL_RUNNER_API_VERSION:
        raise RuntimeError(
            f"Model runner {name!r} from {provider} implements API version "
            f"{plugin.api_version}; this SGLang release requires version "
            f"{MODEL_RUNNER_API_VERSION}."
        )
    logger.info("Selected model runner %r from %s", name, provider)
    return LoadedModelRunner(
        name=name, plugin=plugin, distribution=_distribution(entry_point)
    )


def _distribution(entry_point: EntryPoint) -> Optional[str]:
    return entry_point.dist.name if entry_point.dist is not None else None


def _provider(entry_point: EntryPoint) -> str:
    return _distribution(entry_point) or entry_point.value


def _providers(candidates: list[EntryPoint]) -> str:
    return ", ".join(sorted(_provider(ep) for ep in candidates))
