"""Volatile in-memory cache for UI reads."""

from __future__ import annotations

from copy import deepcopy

_CACHE: dict[str, dict] = {}


def seed_from_store(records: dict[str, dict]) -> None:
    """Seed cache from durable store records."""
    _CACHE.clear()
    for job_id, record in records.items():
        _CACHE[job_id] = deepcopy(record)


def put(job_id: str, record: dict) -> None:
    """Write a record snapshot into cache."""
    _CACHE[job_id] = deepcopy(record)


def get(job_id: str) -> dict | None:
    """Get one job from cache."""
    record = _CACHE.get(job_id)
    return deepcopy(record) if record else None


def get_system_status() -> dict:
    """Return UI status payload from cache only."""
    jobs = [deepcopy(item) for item in _CACHE.values()]
    active_jobs = sum(1 for item in jobs if item.get("status") == "processing")
    queued_jobs = sum(1 for item in jobs if item.get("status") == "queued")
    return {
        "active_jobs": active_jobs,
        "queued_jobs": queued_jobs,
        "total_jobs": len(jobs),
        "jobs": jobs,
    }
