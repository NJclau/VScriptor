"""Characterize blob-write contention risk for concurrent job updates."""

from __future__ import annotations

from copy import deepcopy
from threading import Barrier, Lock, Thread
from time import sleep
from unittest.mock import patch

from state import job_store


TRIALS = 100
FETCH_LATENCY_SECS = 0.003
PUSH_LATENCY_SECS = 0.003


def _risk_band(lost_update_rate_pct: float) -> str:
    if lost_update_rate_pct > 50:
        return "CRITICAL"
    if lost_update_rate_pct > 10:
        return "HIGH"
    if lost_update_rate_pct > 1:
        return "MODERATE"
    return "LOW"


def test_blob_contention_lost_update_rate_report(capsys) -> None:
    """Run repeated two-thread updates and report lost-update risk bands."""
    lost_updates = 0

    for trial_index in range(TRIALS):
        durable_store = {
            "job-a": {
                "job_id": "job-a",
                "chunk_transcripts": [],
                "completed_chunks": 0,
                "status": "running",
            },
            "job-b": {
                "job_id": "job-b",
                "chunk_transcripts": [],
                "completed_chunks": 0,
                "status": "running",
            },
        }

        fetch_barrier = Barrier(2)
        push_lock = Lock()

        def delayed_fetch_jobs() -> dict:
            snapshot = deepcopy(durable_store)
            fetch_barrier.wait(timeout=1)
            sleep(FETCH_LATENCY_SECS)
            return snapshot

        def delayed_push_jobs(jobs: dict) -> None:
            sleep(PUSH_LATENCY_SECS)
            with push_lock:
                durable_store.clear()
                durable_store.update(deepcopy(jobs))

        with patch("state.job_store._fetch_jobs", side_effect=delayed_fetch_jobs), patch(
            "state.job_store._push_jobs", side_effect=delayed_push_jobs
        ):
            thread_a = Thread(target=job_store.update_job_chunk, args=("job-a", 0, f"a-{trial_index}"))
            thread_b = Thread(target=job_store.update_job_chunk, args=("job-b", 0, f"b-{trial_index}"))
            thread_a.start()
            thread_b.start()
            thread_a.join()
            thread_b.join()

        both_updates_persisted = (
            durable_store["job-a"]["completed_chunks"] == 1
            and durable_store["job-b"]["completed_chunks"] == 1
        )
        if not both_updates_persisted:
            lost_updates += 1

    lost_update_rate_pct = (lost_updates / TRIALS) * 100
    report_risk = _risk_band(lost_update_rate_pct)

    print(f"total trials: {TRIALS}")
    print(f"lost updates: {lost_updates}")
    print(f"lost update rate (%): {lost_update_rate_pct:.2f}")
    print("risk thresholds: >10% HIGH, >50% CRITICAL")
    print(f"observed risk: {report_risk}")

    report = capsys.readouterr().out
    assert "total trials:" in report
    assert "lost updates:" in report
    assert "lost update rate (%):" in report
    assert "risk thresholds: >10% HIGH, >50% CRITICAL" in report
    assert "observed risk:" in report
    assert lost_updates == TRIALS
