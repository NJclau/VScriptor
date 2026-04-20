"""Durable job state store backed by a Hugging Face dataset repository.

This module is the single write path for persisted job state. Callers should use
its transition APIs and avoid calling Hub write APIs directly.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, upload_file

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "mbaza-nlp/stt-job-state")
JOB_STATE_FILE = os.getenv("JOB_STATE_FILE", "jobs.json")
BACKOFF_BASE_SECS = float(os.getenv("BACKOFF_BASE_SECS", "1.0"))
BACKOFF_MAX_SECS = float(os.getenv("BACKOFF_MAX_SECS", "30.0"))
BACKOFF_MULTIPLIER = float(os.getenv("BACKOFF_MULTIPLIER", "2.0"))

_write_lock = threading.Lock()


class JobStatus(str, Enum):
    """Allowed persistent lifecycle statuses for a transcription job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


def _pull_from_hub() -> dict[str, dict[str, Any]]:
    """Download the latest persisted jobs map from the Hub dataset repo."""

    try:
        file_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=JOB_STATE_FILE,
            repo_type="dataset",
            force_download=True,
        )
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _is_rate_limited(error: Exception) -> bool:
    """Return True only for Hub HTTP 429 responses."""

    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code == 429:
        return True

    message = str(error).lower()
    return "429" in message or "rate limit" in message


def _push_to_hub(jobs: dict[str, dict[str, Any]]) -> None:
    """Persist the full jobs map with retries on HTTP 429 only."""

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump(jobs, tmp, ensure_ascii=False, indent=2, sort_keys=True)
            temp_path = tmp.name

        attempt_delay = BACKOFF_BASE_SECS
        while True:
            try:
                upload_file(
                    path_or_fileobj=temp_path,
                    path_in_repo=JOB_STATE_FILE,
                    repo_id=HF_DATASET_REPO,
                    repo_type="dataset",
                )
                return
            except Exception as err:
                if not _is_rate_limited(err):
                    raise
                time.sleep(min(attempt_delay, BACKOFF_MAX_SECS))
                attempt_delay = min(BACKOFF_MAX_SECS, attempt_delay * BACKOFF_MULTIPLIER)
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def create_job(job_id: str, audio_path: str, duration_secs: float, total_chunks: int = 0) -> dict[str, Any]:
    """Create a new queued job record and persist it."""

    with _write_lock:
        jobs = _pull_from_hub()
        record = {
            "job_id": job_id,
            "status": JobStatus.QUEUED.value,
            "audio_path": audio_path,
            "duration_secs": duration_secs,
            "total_chunks": total_chunks,
            "chunks_done": 0,
            "transcripts": [],
            "final_transcript": "",
            "error": None,
        }
        jobs[job_id] = record
        _push_to_hub(jobs)
        return record


def update_job_chunk(job_id: str, chunk_index: int, transcript: str) -> dict[str, Any]:
    """Record a processed chunk and move job into processing state."""

    with _write_lock:
        jobs = _pull_from_hub()
        record = jobs[job_id]
        record["status"] = JobStatus.PROCESSING.value
        transcripts = record.setdefault("transcripts", [])
        while len(transcripts) <= chunk_index:
            transcripts.append("")
        transcripts[chunk_index] = transcript
        record["chunks_done"] = int(record.get("chunks_done", 0)) + 1
        _push_to_hub(jobs)
        return record


def complete_job(job_id: str, final_transcript: str) -> dict[str, Any]:
    """Mark a job complete and save its final transcript."""

    with _write_lock:
        jobs = _pull_from_hub()
        record = jobs[job_id]
        record["status"] = JobStatus.COMPLETE.value
        record["final_transcript"] = final_transcript
        record["error"] = None
        _push_to_hub(jobs)
        return record


def fail_job(job_id: str, reason: str) -> dict[str, Any]:
    """Mark a job failed and persist the failure reason."""

    with _write_lock:
        jobs = _pull_from_hub()
        record = jobs[job_id]
        record["status"] = JobStatus.FAILED.value
        record["error"] = reason
        _push_to_hub(jobs)
        return record


def load_all_jobs() -> dict[str, dict[str, Any]]:
    """Load all persisted jobs for startup recovery."""

    return _pull_from_hub()
