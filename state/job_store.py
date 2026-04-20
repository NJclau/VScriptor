"""
Sharded durable job store backed by HF Dataset repo.
Each job is isolated in jobs/{job_id}.json — concurrent jobs never share a file.
All Hub I/O retries on 429 / 5xx with exponential backoff via tenacity.
Public API is unchanged from the previous monolithic implementation.
"""

import json
import os
import tempfile
import time

from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import HF_DATASET_REPO, JOB_STATE_DIR

_api = HfApi()

# ---------------------------------------------------------------------------
# Retry policy — wraps every Hub call
# Retries on network errors and HF rate limits (429).
# Gives up after 5 attempts (~2 min total with backoff).
# ---------------------------------------------------------------------------
_hub_retry = retry(
    retry=retry_if_exception_type((Exception,)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=10, min=10, max=120),
    reraise=True,
)


def _job_path(job_id: str) -> str:
    """Hub path for this job's isolated state file."""
    return f"{JOB_STATE_DIR}/{job_id}.json"


@_hub_retry
def _fetch_job(job_id: str) -> dict | None:
    """Pull a single job record from Hub. Returns None if not yet created."""
    try:
        path = _api.hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=_job_path(job_id),
            repo_type="dataset",
        )
        with open(path) as f:
            return json.load(f)
    except EntryNotFoundError:
        return None
    except Exception as e:
        raise RuntimeError(
            f"job_store: failed to fetch {job_id} from Hub: {e}"
        ) from e


@_hub_retry
def _push_job(job_id: str, record: dict) -> None:
    """Write a single job record to Hub as jobs/{job_id}.json."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            json.dump(record, tmp, indent=2)
            tmp_path = tmp.name
        _api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=_job_path(job_id),
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            commit_message=f"state: {job_id} @ {int(time.time())}",
        )
    except Exception as e:
        raise RuntimeError(
            f"job_store: failed to push {job_id} to Hub: {e}"
        ) from e
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Public API — signatures unchanged from monolithic implementation
# ---------------------------------------------------------------------------

def create_job(
    job_id: str,
    audio_path: str,
    duration_secs: float,
    total_chunks: int,
) -> dict:
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
    _push_job(job_id, record)
    return record


def update_job_chunk(job_id: str, chunk_index: int, transcript_text: str) -> dict:
    record = _fetch_job(job_id)
    if record is None:
        raise KeyError(f"job_store: job {job_id} not found")
    record["chunk_transcripts"].append(transcript_text)
    record["completed_chunks"] = chunk_index + 1
    _push_job(job_id, record)
    return record


def complete_job(job_id: str, final_transcript: str) -> dict:
    record = _fetch_job(job_id)
    if record is None:
        raise KeyError(f"job_store: job {job_id} not found")
    record["status"] = "complete"
    record["final_transcript"] = final_transcript
    record["completed_at"] = int(time.time())
    _push_job(job_id, record)
    return record


def fail_job(job_id: str, reason: str) -> dict:
    record = _fetch_job(job_id)
    if record is None:
        raise KeyError(f"job_store: job {job_id} not found")
    record["status"] = "failed"
    record["error"] = reason
    record["failed_at"] = int(time.time())
    _push_job(job_id, record)
    return record


@_hub_retry
def load_all_jobs() -> list[dict]:
    """Called on Space startup to seed the volatile cache from Hub."""
    try:
        files = _api.list_repo_tree(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            path_in_repo=JOB_STATE_DIR,
        )
        jobs = []
        for f in files:
            job_id = f.rfilename.replace(f"{JOB_STATE_DIR}/", "").replace(".json", "")
            record = _fetch_job(job_id)
            if record:
                jobs.append(record)
        return jobs
    except EntryNotFoundError:
        return []
    except Exception as e:
        raise RuntimeError(f"job_store: failed to load all jobs from Hub: {e}") from e
