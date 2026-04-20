"""
Single source of durable job state.

All reads and writes go to the HF Dataset repo defined in config.

This module is the ONLY writer. cache.py is the only reader for the UI.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from config import BACKOFF_BASE_SECS, BACKOFF_MAX_SECS, BACKOFF_MULTIPLIER, HF_DATASET_REPO, JOB_STATE_FILE

_api = HfApi()
_write_lock = threading.Lock()


def _fetch_jobs() -> dict:
    """Pull the current jobs.json from the Hub. Returns empty dict if not found."""
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=JOB_STATE_FILE,
            repo_type="dataset",
            force_download=True,
        )
        with open(path, encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except EntryNotFoundError:
        return {}
    except Exception as exc:
        raise RuntimeError(f"job_store: failed to fetch state from Hub: {exc}") from exc


def _push_jobs(jobs: dict) -> None:
    """Write jobs dict back to Hub as jobs.json, retrying only HTTP 429 with backoff."""
    delay_secs = BACKOFF_BASE_SECS

    while True:
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
                json.dump(jobs, tmp, indent=2)
                tmp_path = tmp.name
            _api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=JOB_STATE_FILE,
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                commit_message=f"state: job update at {int(time.time())}",
            )
            return
        except Exception as exc:
            if "429" in str(exc) and delay_secs <= BACKOFF_MAX_SECS:
                time.sleep(delay_secs)
                delay_secs *= BACKOFF_MULTIPLIER
                continue
            raise RuntimeError(f"job_store: failed to push state to Hub: {exc}") from exc
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)


def create_job(job_id: str, audio_path: str, duration_secs: float, total_chunks: int) -> dict:
    with _write_lock:
        jobs = _fetch_jobs()
        record = {
            "job_id": job_id,
            "audio_path": audio_path,
            "duration_secs": duration_secs,
            "total_chunks": total_chunks,
            "completed_chunks": 0,
            "chunk_transcripts": [],
            "status": "running",
            "created_at": int(time.time()),
            "final_transcript": None,
            "error": None,
        }
        jobs[job_id] = record
        _push_jobs(jobs)
        return record


def update_job_chunk(job_id: str, chunk_index: int, transcript_text: str) -> dict:
    with _write_lock:
        jobs = _fetch_jobs()
        if job_id not in jobs:
            raise KeyError(f"job_store: job {job_id} not found")
        jobs[job_id]["chunk_transcripts"].append(transcript_text)
        jobs[job_id]["completed_chunks"] = chunk_index + 1
        _push_jobs(jobs)
        return jobs[job_id]


def complete_job(job_id: str, final_transcript: str) -> dict:
    with _write_lock:
        jobs = _fetch_jobs()
        if job_id not in jobs:
            raise KeyError(f"job_store: job {job_id} not found")
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["final_transcript"] = final_transcript
        jobs[job_id]["completed_at"] = int(time.time())
        _push_jobs(jobs)
        return jobs[job_id]


def fail_job(job_id: str, reason: str) -> dict:
    with _write_lock:
        jobs = _fetch_jobs()
        if job_id not in jobs:
            raise KeyError(f"job_store: job {job_id} not found")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = reason
        jobs[job_id]["failed_at"] = int(time.time())
        _push_jobs(jobs)
        return jobs[job_id]


def load_all_jobs() -> list[dict]:
    """Called on Space startup to seed the volatile cache."""
    jobs = _fetch_jobs()
    return list(jobs.values())
