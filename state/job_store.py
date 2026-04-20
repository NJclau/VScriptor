"""Durable job store implementation."""

from __future__ import annotations

import copy
import json
import os
import threading
import uuid
from datetime import datetime, timezone

from config import JOB_STATE_FILE

_WRITE_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jobs_file() -> dict[str, dict]:
    if not os.path.exists(JOB_STATE_FILE):
        return {}
    with open(JOB_STATE_FILE, "r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)
    return data if isinstance(data, dict) else {}


def _write_jobs_file(data: dict[str, dict]) -> None:
    with open(JOB_STATE_FILE, "w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2, ensure_ascii=False)


def load_all_jobs() -> dict[str, dict]:
    """Load all durable jobs."""
    with _WRITE_LOCK:
        return copy.deepcopy(_read_jobs_file())


def create_job(filename: str, duration_secs: float) -> tuple[str, dict]:
    """Create a queued job and persist it."""
    with _WRITE_LOCK:
        jobs = _read_jobs_file()
        job_id = str(uuid.uuid4())
        record = {
            "job_id": job_id,
            "filename": filename,
            "duration_secs": duration_secs,
            "status": "queued",
            "chunks": [],
            "result_text": "",
            "error": None,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        jobs[job_id] = record
        _write_jobs_file(jobs)
        return job_id, copy.deepcopy(record)


def update_job_chunk(job_id: str, chunk_index: int, text: str) -> dict:
    """Persist one chunk update and transition to processing."""
    with _WRITE_LOCK:
        jobs = _read_jobs_file()
        if job_id not in jobs:
            raise KeyError(f"Job {job_id} not found")
        record = jobs[job_id]
        record["status"] = "processing"
        record["chunks"].append({"chunk_index": chunk_index, "text": text})
        record["updated_at"] = _now_iso()
        _write_jobs_file(jobs)
        return copy.deepcopy(record)


def complete_job(job_id: str, result_text: str) -> dict:
    """Mark a job as complete."""
    with _WRITE_LOCK:
        jobs = _read_jobs_file()
        if job_id not in jobs:
            raise KeyError(f"Job {job_id} not found")
        record = jobs[job_id]
        record["status"] = "complete"
        record["result_text"] = result_text
        record["updated_at"] = _now_iso()
        _write_jobs_file(jobs)
        return copy.deepcopy(record)


def fail_job(job_id: str, reason: str) -> dict:
    """Mark a job as failed."""
    with _WRITE_LOCK:
        jobs = _read_jobs_file()
        if job_id not in jobs:
            raise KeyError(f"Job {job_id} not found")
        record = jobs[job_id]
        record["status"] = "failed"
        record["error"] = reason
        record["updated_at"] = _now_iso()
        _write_jobs_file(jobs)
        return copy.deepcopy(record)
