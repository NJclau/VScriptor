"""Volatile in-memory cache for UI read operations."""

from __future__ import annotations

from threading import Lock
from typing import Any

_cache: dict[str, dict[str, Any]] = {}
_cache_lock = Lock()


def seed_from_store(jobs: dict[str, dict[str, Any]]) -> None:
    """Replace cache contents with records loaded from durable storage."""
    with _cache_lock:
        _cache.clear()
        _cache.update(jobs)


def put(job_id: str, record: dict[str, Any]) -> None:
    """Store or update a single job record in cache."""
    with _cache_lock:
        _cache[job_id] = record


def get(job_id: str) -> dict[str, Any] | None:
    """Fetch a single job record from cache."""
    with _cache_lock:
        return _cache.get(job_id)


def get_system_status() -> dict[str, Any]:
    """Return lightweight status info for the UI status widget."""
    with _cache_lock:
        active_jobs = sum(1 for record in _cache.values() if record.get("status") == "processing")
        queued_jobs = sum(1 for record in _cache.values() if record.get("status") == "queued")

    load_level = "idle"
    if active_jobs > 0:
        load_level = "busy"
    if queued_jobs > 0:
        load_level = "queued"

    return {
        "active_jobs": active_jobs,
        "queued_jobs": queued_jobs,
        "load_level": load_level,
    }
