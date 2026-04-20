from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from state import job_store

FREE_TIER_GUIDANCE_WRITES_PER_SEC = 1.0
REPORT_DIR = Path("qa-reports")
REPORT_PATH = REPORT_DIR / "write_frequency_report.txt"


def _risk_band(writes_per_sec: float) -> str:
    if writes_per_sec <= FREE_TIER_GUIDANCE_WRITES_PER_SEC * 0.8:
        return "LOW"
    if writes_per_sec <= FREE_TIER_GUIDANCE_WRITES_PER_SEC:
        return "MODERATE"
    if writes_per_sec <= FREE_TIER_GUIDANCE_WRITES_PER_SEC * 1.5:
        return "HIGH"
    return "CRITICAL"


def _run_lifecycle_for_frequency(concurrent_jobs: int, chunks_per_job: int) -> dict[str, Any]:
    fake_store: dict[str, dict[str, Any]] = {}

    def fake_fetch() -> dict[str, dict[str, Any]]:
        return {
            job_id: {
                key: (value.copy() if isinstance(value, list) else value)
                for key, value in record.items()
            }
            for job_id, record in fake_store.items()
        }

    push_calls: list[dict[str, dict[str, Any]]] = []

    def fake_push(payload: dict[str, dict[str, Any]]) -> None:
        push_calls.append(payload)
        fake_store.clear()
        fake_store.update(payload)

    with patch("state.job_store._fetch_jobs", side_effect=fake_fetch), patch(
        "state.job_store._push_jobs", side_effect=fake_push
    ):
        started = time.perf_counter()
        for index in range(concurrent_jobs):
            job_id = f"job-{index:03d}"
            job_store.create_job(job_id, f"/tmp/{job_id}.wav", 30.0, chunks_per_job)
            for chunk_index in range(chunks_per_job):
                job_store.update_job_chunk(job_id, chunk_index, f"transcript-{chunk_index}")
            job_store.complete_job(job_id, "final transcript")
        elapsed_secs = time.perf_counter() - started

    total_writes = len(push_calls)
    writes_per_sec = total_writes / elapsed_secs if elapsed_secs > 0 else float("inf")
    return {
        "concurrent_jobs": concurrent_jobs,
        "chunks_per_job": chunks_per_job,
        "total_writes": total_writes,
        "elapsed_secs": elapsed_secs,
        "writes_per_sec": writes_per_sec,
        "risk_band": _risk_band(writes_per_sec),
    }


@pytest.mark.parametrize(
    ("concurrent_jobs", "chunks_per_job"),
    [
        (1, 1),
        (2, 4),
        (4, 6),
    ],
)
def test_write_frequency_report_uses_expected_formula(concurrent_jobs: int, chunks_per_job: int) -> None:
    metrics = _run_lifecycle_for_frequency(concurrent_jobs, chunks_per_job)
    expected_writes = concurrent_jobs * (chunks_per_job + 2)

    assert metrics["total_writes"] == expected_writes

    report_line = (
        f"scenario(concurrent_jobs={concurrent_jobs}, chunks_per_job={chunks_per_job}) | "
        f"total writes={metrics['total_writes']} | "
        f"elapsed seconds={metrics['elapsed_secs']:.6f} | "
        f"writes/sec={metrics['writes_per_sec']:.2f} | "
        f"risk band={metrics['risk_band']} "
        f"(guidance≈{FREE_TIER_GUIDANCE_WRITES_PER_SEC:.1f} write/sec)"
    )

    print(report_line)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("a", encoding="utf-8") as report_file:
        report_file.write(report_line + "\n")
