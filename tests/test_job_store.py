from unittest.mock import patch

import pytest

from state import job_store


def test_load_all_jobs_returns_records_from_hub_payload():
    mock_data = {
        "job-1": {"status": "processing"},
        "job-2": {"status": "queued"},
    }

    with patch("state.job_store._pull_from_hub", return_value=mock_data):
        loaded = job_store.load_all_jobs()

    assert loaded == mock_data


def test_create_job_persists_queued_record_with_push():
    with patch("state.job_store._pull_from_hub", return_value={}), patch(
        "state.job_store._push_to_hub"
    ) as push:
        record = job_store.create_job("job-123", "audio.wav", 21.0)

    assert record["status"] == "queued"
    assert record["job_id"] == "job-123"
    assert push.call_count == 1


def test_push_to_hub_retries_on_429_before_success(monkeypatch):
    calls = {"count": 0}

    def flaky_push(payload):
        calls["count"] += 1
        if calls["count"] < 3:
            raise Exception("429")

    monkeypatch.setattr(job_store, "_push_to_hub", flaky_push)
    monkeypatch.setattr(job_store.time, "sleep", lambda _secs: None)

    with patch("state.job_store._pull_from_hub", return_value={}):
        record = job_store.create_job("job-429", "a.wav", 10.0)

    assert record["job_id"] == "job-429"
    assert calls["count"] == 3


def test_update_job_chunk_transitions_job_to_processing():
    store = {
        "job-1": {
            "job_id": "job-1",
            "status": "queued",
            "chunks": [],
        }
    }

    with patch("state.job_store._pull_from_hub", return_value=store), patch(
        "state.job_store._push_to_hub"
    ):
        record = job_store.update_job_chunk("job-1", 0, "text")

    assert record["status"] == "processing"
    assert len(record["chunks"]) == 1
