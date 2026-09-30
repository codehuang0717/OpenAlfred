"""Cancellable post-goodbye grace period for a phone conversation."""

import asyncio
from collections.abc import Callable


class EndCallCountdown:
    def __init__(
        self,
        should_exit: asyncio.Event,
        speech_active: Callable[[], bool],
        grace_seconds: float = 2.0,
    ) -> None:
        self.should_exit = should_exit
        self.speech_active = speech_active
        self.grace_seconds = grace_seconds
        self.task: asyncio.Task[None] | None = None

    def arm(self) -> None:
        self.cancel()
        if not self.should_exit.is_set() and not self.speech_active():
            self.task = asyncio.create_task(self._wait())

    def cancel(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
        self.task = None

    async def _wait(self) -> None:
        try:
            await asyncio.sleep(self.grace_seconds)
            if not self.speech_active() and not self.should_exit.is_set():
                self.should_exit.set()
        except asyncio.CancelledError:
            pass
        finally:
            if self.task is asyncio.current_task():
                self.task = None
