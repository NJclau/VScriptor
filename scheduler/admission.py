"""Duration-based admission control for transcription jobs."""

import asyncio
from enum import Enum

from config import MAX_AUDIO_DURATION_SECS, MAX_PRIMARY_SLOTS, SHORT_JOB_THRESHOLD_SECS


class SlotType(Enum):
    """Admission slot class selected by job duration."""

    SHORT = "short"
    PRIMARY = "primary"


class AdmissionController:
    """Control concurrent admission using short and primary semaphores."""

    def __init__(self) -> None:
        self._short = asyncio.Semaphore(1)
        self._primary = asyncio.Semaphore(MAX_PRIMARY_SLOTS)

    async def acquire(self, duration_secs: float) -> SlotType:
        """Acquire slot by duration, rejecting files longer than max cap."""
        if duration_secs > MAX_AUDIO_DURATION_SECS:
            raise ValueError(
                f"Audio duration {duration_secs:.1f}s exceeds maximum "
                f"{MAX_AUDIO_DURATION_SECS:.1f}s. Please trim your file."
            )

        if duration_secs <= SHORT_JOB_THRESHOLD_SECS:
            await self._short.acquire()
            return SlotType.SHORT

        await self._primary.acquire()
        return SlotType.PRIMARY

    def release(self, slot: SlotType) -> None:
        """Release a previously-acquired slot."""
        if slot == SlotType.SHORT:
            self._short.release()
            return
        self._primary.release()
