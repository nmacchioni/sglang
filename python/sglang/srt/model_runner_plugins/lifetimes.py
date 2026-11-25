"""SGLang's bookkeeping of step output lifetimes: when each ``StepOutput`` is
released to the runner, and after which device event.

A step's output is consumed by sampling (on the runner stream, possibly
deferred until grammar is ready) and by host copies (on the copy stream). With
overlap scheduling off, every consumer has run on the host's side before the
next step is planned, so an output is released when the next step starts, with
no event. With overlap on, the scheduler marks an output consumed once its last
consumer is enqueued and hands over an event recorded behind it; the next step's
``execute`` may already be running by the time that event fires, so a runner
orders its overwrites of a shared buffer after it.

Outputs are released in production order, and one holding a shared buffer
before the next ``execute``, so a runner never sees step n+1 written into a
buffer whose step-n consumers it was not told about.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Deque, FrozenSet, Optional

from sglang.srt.model_runner_plugins.api import OutputBuffer, StepOutput

ReleaseFn = Callable[[StepOutput, Optional[Any]], None]


class OutputLease:
    """One step's output, until SGLang releases it to the runner."""

    __slots__ = ("out", "index", "after", "consumed", "released")

    def __init__(self, *, out: StepOutput, index: int):
        self.out = out
        self.index = index
        self.after: Optional[Any] = None
        self.consumed = False
        self.released = False


class OutputLifetimes:
    def __init__(
        self,
        *,
        runner_name: str,
        release: ReleaseFn,
        shared: FrozenSet[OutputBuffer],
        deferred: bool,
    ):
        self._runner_name = runner_name
        self._release = release
        self._shared = shared
        # True under overlap scheduling: the scheduler marks consumption.
        self.deferred = deferred
        self._pending: Deque[OutputLease] = deque()
        self._count = 0
        # Not deferred: the one output not yet released (the last step's).
        self._last: Optional[StepOutput] = None

    @property
    def pending(self) -> int:
        return len(self._pending) + (self._last is not None)

    def step_started(self) -> None:
        """Without overlap, the previous step's consumers have all run."""
        last = self._last
        if last is not None:
            self._last = None
            self._release(last, None)

    def before_execute(self) -> None:
        """No earlier output may still hold a buffer the next step overwrites."""
        # Not deferred, step_started released every earlier output.
        for lease in self._pending:
            held = lease.out.buffers() & self._shared
            if held:
                raise RuntimeError(
                    f"Model runner {self._runner_name!r}: step {lease.index}'s "
                    f"{', '.join(sorted(b.value for b in held))} is not released "
                    "before the next execute() reuses its buffer"
                )

    def produced(self, out: StepOutput) -> Optional[OutputLease]:
        """The lease the scheduler marks consumed under overlap; None when not
        deferred (the next step_started releases ``out``)."""
        if not self.deferred:
            self._count += 1
            self._last = out
            return None
        lease = OutputLease(out=out, index=self._count)
        self._count += 1
        self._pending.append(lease)
        return lease

    def consumers_enqueued(self, lease: OutputLease, after: Optional[Any]) -> None:
        """Every consumer of ``lease`` is enqueued; ``after`` fires behind them."""
        if lease.consumed:
            raise RuntimeError(f"step {lease.index}'s output was marked consumed twice")
        lease.consumed = True
        lease.after = after
        self._release_consumed()

    def close(self) -> None:
        """Release everything left; the caller has synchronized the device."""
        self.step_started()
        for lease in self._pending:
            lease.consumed = True
            lease.after = None
        self._release_consumed()

    def _release_consumed(self) -> None:
        # In production order: a consumed output waits for every earlier one.
        while self._pending and self._pending[0].consumed:
            lease = self._pending.popleft()
            lease.released = True
            self._release(lease.out, lease.after)
