from unittest.mock import patch

from state import job_store


def test_load_all_jobs_returns_records_from_hub_payload():
    mock_data = {
        "job-1": {"job_id": "job-1", "status": "processing"},
        "job-2": {"job_id": "job-2", "status": "queued"},
    }

    with patch("state.job_store._fetch_job", side_effect=lambda _job_id: mock_data, create=True), patch(
        "state.job_store._fetch_jobs", return_value=mock_data
    ):
        loaded = job_store.load_all_jobs()

    assert loaded == list(mock_data.values())


def test_create_job_persists_record_with_push():
    fake_store: dict[str, dict] = {}
    pushed: list[tuple[str, dict]] = []

    def fake_fetch_job(job_id: str) -> dict | None:
        return fake_store.get(job_id)

    def fake_push_job(job_id: str, record: dict) -> None:
        pushed.append((job_id, record.copy()))
        fake_store[job_id] = record.copy()

    with patch("state.job_store._fetch_job", side_effect=fake_fetch_job, create=True), patch(
        "state.job_store._push_job", side_effect=fake_push_job, create=True
    ), patch("state.job_store._fetch_jobs", return_value={}), patch("state.job_store._push_jobs"):
        record = job_store.create_job("job-123", "audio.wav", 21.0, 3)

    assert record["status"] == "running"
    assert record["job_id"] == "job-123"
    assert pushed == []


def test_push_to_hub_retries_on_429_before_success(monkeypatch):
    calls = {"count": 0}

    def flaky_upload_file(**_kwargs) -> None:
        calls["count"] += 1
        if calls["count"] < 3:
            raise Exception("429")

    monkeypatch.setattr(job_store._api, "upload_file", flaky_upload_file)
    monkeypatch.setattr(job_store.time, "sleep", lambda _secs: None)

    with patch("state.job_store._fetch_job", return_value=None, create=True), patch(
        "state.job_store._fetch_jobs", return_value={}
    ):
        record = job_store.create_job("job-429", "a.wav", 10.0, 2)

    assert record["job_id"] == "job-429"
    assert calls["count"] == 3


def test_update_job_chunk_transitions_job_to_processing():
    store = {
        "job-1": {
            "job_id": "job-1",
            "status": "queued",
            "completed_chunks": 0,
            "chunk_transcripts": [],
        }
    }

    def fake_fetch_job(job_id: str) -> dict | None:
        return store.get(job_id)

    with patch("state.job_store._fetch_job", side_effect=fake_fetch_job, create=True), patch(
        "state.job_store._push_job", create=True
    ), patch("state.job_store._fetch_jobs", return_value=store), patch("state.job_store._push_jobs"):
        record = job_store.update_job_chunk("job-1", 0, "text")

    assert record["completed_chunks"] == 1
    assert len(record["chunk_transcripts"]) == 1
