"""Admission controller based on duration-aware semaphores."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from config import MAX_AUDIO_DURATION_SECS, MAX_PRIMARY_SLOTS, MAX_SHORT_SLOTS, SHORT_JOB_THRESHOLD_SECS


@dataclass(frozen=True)
class SlotHandle:
    """Represents an acquired slot in one of the admission pools."""

    slot_type: str


class AdmissionController:
    """Route short jobs and long jobs to dedicated semaphores."""

    def __init__(self) -> None:
        self._short_slots = asyncio.Semaphore(MAX_SHORT_SLOTS)
        self._primary_slots = asyncio.Semaphore(MAX_PRIMARY_SLOTS)

    async def acquire(self, duration_secs: float) -> SlotHandle:
        """Acquire an admission slot for a job duration."""
        if duration_secs > MAX_AUDIO_DURATION_SECS:
            raise ValueError(
                f"Audio duration {duration_secs:.2f}s exceeds cap {MAX_AUDIO_DURATION_SECS:.2f}s"
            )
        if duration_secs <= SHORT_JOB_THRESHOLD_SECS:
            await self._short_slots.acquire()
            return SlotHandle(slot_type="short")

        await self._primary_slots.acquire()
        return SlotHandle(slot_type="primary")

    def release(self, slot: SlotHandle) -> None:
        """Release a previously acquired slot."""
        if slot.slot_type == "short":
            self._short_slots.release()
        else:
            self._primary_slots.release()
