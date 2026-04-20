from unittest.mock import patch, MagicMock
from state import job_store

def test_load_all_jobs_returns_records_from_hub_shards():
    mock_files = [
        MagicMock(rfilename="jobs/job-1.json"),
        MagicMock(rfilename="jobs/job-2.json"),
    ]
    mock_data = {
        "job-1": {"job_id": "job-1", "status": "processing"},
        "job-2": {"job_id": "job-2", "status": "queued"},
    }

    def fake_fetch(job_id):
        return mock_data.get(job_id)

    with patch("state.job_store._api.list_repo_tree", return_value=mock_files), \
         patch("state.job_store._fetch_job", side_effect=fake_fetch):
        loaded = job_store.load_all_jobs()

    assert len(loaded) == 2
    assert any(j["job_id"] == "job-1" for j in loaded)
    assert any(j["job_id"] == "job-2" for j in loaded)


def test_create_job_persists_isolated_record():
    pushed = []

    def fake_push(job_id, record):
        pushed.append((job_id, record.copy()))

    with patch("state.job_store._push_job", side_effect=fake_push):
        record = job_store.create_job("job-123", "audio.wav", 21.0, 3)

    assert record["status"] == "running"
    assert record["job_id"] == "job-123"
    assert len(pushed) == 1
    assert pushed[0][0] == "job-123"
    assert pushed[0][1]["job_id"] == "job-123"


def test_update_job_chunk_appends_to_isolated_record():
    store = {
        "job-1": {
            "job_id": "job-1",
            "status": "running",
            "completed_chunks": 0,
            "chunk_transcripts": [],
        }
    }

    def fake_fetch(job_id):
        return store.get(job_id)

    def fake_push(job_id, record):
        store[job_id] = record

    with patch("state.job_store._fetch_job", side_effect=fake_fetch), \
         patch("state.job_store._push_job", side_effect=fake_push):
        record = job_store.update_job_chunk("job-1", 0, "text")

    assert record["completed_chunks"] == 1
    assert "text" in record["chunk_transcripts"]
    assert store["job-1"]["completed_chunks"] == 1


def test_state_lifecycle_create_update_complete():
    store = {}

    def fake_fetch(job_id):
        return dict(store.get(job_id, {})) or None

    def fake_push(job_id, record):
        store[job_id] = dict(record)

    with patch("state.job_store._fetch_job", side_effect=fake_fetch), \
         patch("state.job_store._push_job", side_effect=fake_push):

        record = job_store.create_job("job-001", "/audio/test.wav", 25.0, 2)
        assert record["status"] == "running"

        record = job_store.update_job_chunk("job-001", 0, "muraho")
        assert record["completed_chunks"] == 1

        record = job_store.complete_job("job-001", "muraho neza")
        assert record["status"] == "complete"
        assert record["final_transcript"] == "muraho neza"
