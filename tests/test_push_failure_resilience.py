"""Resilience characterization tests for push failures in state.job_store."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from state import job_store


def _make_fetch_from(durable_store: dict):
    """Return a _fetch_jobs mock that emulates pull-by-value semantics."""

    def _fetch_jobs() -> dict:
        return deepcopy(durable_store)

    return _fetch_jobs


def _actionable_error(exc: Exception) -> bool:
    """Heuristic for whether an exception gives a caller enough context to act."""
    message = str(exc).lower()
    return any(token in message for token in ("push", "hub", "failed", "create", "update", "complete"))


def _print_scenario_report(name: str, exception_kind: str, durable_store: dict, actionable: bool) -> None:
    print(f"scenario={name}")
    print(f"exception_behavior={exception_kind}")
    print(f"durable_state={durable_store}")
    print(f"actionable_error_signal={actionable}")


def test_push_failure_on_create_reports_exception_and_no_durable_write(capsys):
    durable_store: dict = {}

    def fail_create_push(jobs: dict) -> None:
        raise RuntimeError("simulated push failure at create")

    with patch("state.job_store._fetch_jobs", side_effect=_make_fetch_from(durable_store)), patch(
        "state.job_store._push_jobs", side_effect=fail_create_push
    ):
        with pytest.raises(RuntimeError, match="create") as exc_info:
            job_store.create_job("job-create-fail", "audio.wav", 9.0, 3)

    actionable = _actionable_error(exc_info.value)
    _print_scenario_report("create", "raised", durable_store, actionable)

    out = capsys.readouterr().out
    assert "scenario=create" in out
    assert "exception_behavior=raised" in out
    assert durable_store == {}
    assert actionable is True


def test_push_failure_mid_chunk_update_leaves_last_durable_checkpoint(capsys):
    durable_store: dict = {}
    push_call_counter = {"count": 0}

    def checkpoint_then_fail_mid_update(jobs: dict) -> None:
        push_call_counter["count"] += 1
        if push_call_counter["count"] == 3:
            raise RuntimeError("simulated push failure at mid-chunk update")
        durable_store.clear()
        durable_store.update(deepcopy(jobs))

    with patch("state.job_store._fetch_jobs", side_effect=_make_fetch_from(durable_store)), patch(
        "state.job_store._push_jobs", side_effect=checkpoint_then_fail_mid_update
    ):
        job_store.create_job("job-mid-fail", "audio.wav", 12.0, 3)
        job_store.update_job_chunk("job-mid-fail", 0, "chunk-0")

        with pytest.raises(RuntimeError, match="mid-chunk update") as exc_info:
            job_store.update_job_chunk("job-mid-fail", 1, "chunk-1")

    actionable = _actionable_error(exc_info.value)
    _print_scenario_report("mid-chunk update", "raised", durable_store, actionable)

    out = capsys.readouterr().out
    assert "scenario=mid-chunk update" in out
    assert "exception_behavior=raised" in out
    assert durable_store["job-mid-fail"]["completed_chunks"] == 1
    assert durable_store["job-mid-fail"]["chunk_transcripts"] == ["chunk-0"]
    assert durable_store["job-mid-fail"]["status"] == "running"
    assert actionable is True


def test_push_failure_on_complete_preserves_running_state_and_signals_error(capsys):
    durable_store: dict = {}
    push_call_counter = {"count": 0}

    def fail_on_complete(jobs: dict) -> None:
        push_call_counter["count"] += 1
        if push_call_counter["count"] == 4:
            raise RuntimeError("simulated push failure at complete")
        durable_store.clear()
        durable_store.update(deepcopy(jobs))

    with patch("state.job_store._fetch_jobs", side_effect=_make_fetch_from(durable_store)), patch(
        "state.job_store._push_jobs", side_effect=fail_on_complete
    ):
        job_store.create_job("job-complete-fail", "audio.wav", 20.0, 2)
        job_store.update_job_chunk("job-complete-fail", 0, "chunk-0")
        job_store.update_job_chunk("job-complete-fail", 1, "chunk-1")

        with pytest.raises(RuntimeError, match="complete") as exc_info:
            job_store.complete_job("job-complete-fail", "final text")

    actionable = _actionable_error(exc_info.value)
    _print_scenario_report("complete", "raised", durable_store, actionable)

    out = capsys.readouterr().out
    assert "scenario=complete" in out
    assert "exception_behavior=raised" in out
    assert durable_store["job-complete-fail"]["status"] == "running"
    assert durable_store["job-complete-fail"]["final_transcript"] is None
    assert durable_store["job-complete-fail"]["completed_chunks"] == 2
    assert actionable is True


def test_source_inspection_reports_missing_retry_backoff_as_high_risk(capsys):
    source_path = Path("state/job_store.py")
    source_text = source_path.read_text(encoding="utf-8").lower()

    keywords = ("retry", "backoff", "429", "sleep")
    missing = [keyword for keyword in keywords if keyword not in source_text]
    risk = "HIGH" if missing else "LOW"

    print(f"source_inspection_path={source_path}")
    print(f"missing_keywords={missing}")
    print(f"risk_classification={risk}")

    out = capsys.readouterr().out
    assert "risk_classification=HIGH" in out
    assert missing
    assert risk == "HIGH"
