"""Stress simulation for concurrent job lifecycle writes through state.job_store."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import random
import threading
import time
from unittest.mock import patch

from state import job_store


def test_concurrent_job_lifecycle_stress_audit_reports_low_risk_completion() -> None:
    """Launch 10 concurrent jobs and audit final durable state completeness."""
    job_count = 10
    chunks_per_job = 3
    rng = random.Random(7)

    durable_store: dict[str, dict] = {}
    write_log: list[dict[str, float | str | int]] = []
    store_lock = threading.Lock()

    expected_ids = [f"stress-job-{index:02d}" for index in range(job_count)]

    def mocked_fetch_job(job_id: str) -> dict | None:
        with store_lock:
            return deepcopy(durable_store.get(job_id))

    def mocked_push_job(job_id: str, record: dict) -> None:
        latency_secs = rng.uniform(0.05, 0.2)
        time.sleep(latency_secs)

        with store_lock:
            durable_store[job_id] = deepcopy(record)
            write_log.append(
                {
                    "ts": time.perf_counter(),
                    "latency_ms": int(latency_secs * 1000),
                    "job_id": job_id,
                }
            )

    def run_job_lifecycle(job_id: str) -> None:
        job_store.create_job(job_id, f"{job_id}.wav", 9.0, chunks_per_job)
        for chunk_index in range(chunks_per_job):
            job_store.update_job_chunk(job_id, chunk_index, f"{job_id}-chunk-{chunk_index}")
        final_text = " ".join(f"{job_id}-chunk-{chunk_index}" for chunk_index in range(chunks_per_job))
        job_store.complete_job(job_id, final_text)

    async def run_all_lifecycles() -> None:
        await asyncio.gather(*(asyncio.to_thread(run_job_lifecycle, job_id) for job_id in expected_ids))

    started_at = time.perf_counter()
    with patch("state.job_store._fetch_job", side_effect=mocked_fetch_job), \
         patch("state.job_store._push_job", side_effect=mocked_push_job):
        asyncio.run(run_all_lifecycles())
    wall_time_secs = time.perf_counter() - started_at

    writes_total = len(write_log)
    writes_per_sec = writes_total / wall_time_secs if wall_time_secs else 0.0
    completed_jobs_count = sum(1 for job in durable_store.values() if job.get("status") == "complete")

    print(f"wall time: {wall_time_secs:.3f}s")
    print(f"total writes: {writes_total}")
    print(f"writes/sec: {writes_per_sec:.2f}")
    print(f"completed jobs count: {completed_jobs_count}/{job_count}")

    incomplete_or_missing: list[dict[str, str]] = []

    for job_id in expected_ids:
        expected_final = " ".join(f"{job_id}-chunk-{chunk_index}" for chunk_index in range(chunks_per_job))
        record = durable_store.get(job_id)

        if record is None:
            incomplete_or_missing.append({"job_id": job_id, "finding": "missing job record"})
            continue

        if record.get("status") != "complete":
            incomplete_or_missing.append(
                {"job_id": job_id, "finding": f"status={record.get('status')} expected=complete"}
            )

        if record.get("final_transcript") != expected_final:
            incomplete_or_missing.append(
                {
                    "job_id": job_id,
                    "finding": "final transcript mismatch or missing",
                }
            )

        if record.get("completed_chunks") != chunks_per_job:
            incomplete_or_missing.append(
                {
                    "job_id": job_id,
                    "finding": f"completed_chunks={record.get('completed_chunks')} expected={chunks_per_job}",
                }
            )

        chunk_transcripts = record.get("chunk_transcripts", [])
        if len(chunk_transcripts) != chunks_per_job:
            incomplete_or_missing.append(
                {
                    "job_id": job_id,
                    "finding": f"chunk_transcripts count={len(chunk_transcripts)} expected={chunks_per_job}",
                }
            )

    if incomplete_or_missing:
        print("incomplete/missing jobs table:")
        print("job_id | finding")
        print("---|---")
        for finding in incomplete_or_missing:
            print(f"{finding['job_id']} | {finding['finding']}")
    else:
        print("incomplete/missing jobs table: none")

    risk_classification = "LOW" if not incomplete_or_missing else "HIGH"
    print(f"risk classification: {risk_classification}")

    assert completed_jobs_count == job_count, (
        f"REGRESSION: {job_count - completed_jobs_count} jobs failed to complete. "
        f"The sharding fix did not resolve the concurrency issue."
    )
    assert risk_classification in ("LOW", "MEDIUM"), (
        f"REGRESSION: stress risk is still {risk_classification}. "
        f"Expected LOW or MEDIUM after sharding fix."
    )
