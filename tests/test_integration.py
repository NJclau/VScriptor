from unittest.mock import Mock, patch

import numpy as np


def test_pipeline_happy_path_uses_new_package_boundaries(tmp_path):
    from inference.postprocess import stitch_chunks

    fake_model = Mock()
    fake_processor = Mock()
    fake_processor.batch_decode.return_value = ["Muraho"]

    chunks = [str(tmp_path / "chunk_0000.wav"), str(tmp_path / "chunk_0001.wav")]

    with patch("inference.loader.get_model_and_processor", return_value=(fake_model, fake_processor)), patch(
        "inference.preprocess.load_and_normalise", return_value=(np.zeros(16000, dtype=np.float32), 16000)
    ), patch("inference.chunker.chunk_audio", return_value=chunks), patch(
        "inference.decoder.decode_logits", return_value="Muraho"
    ), patch("state.job_store.create_job", return_value={"status": "queued"}), patch(
        "state.job_store.update_job_chunk", return_value={"status": "processing"}
    ), patch("state.job_store.complete_job", return_value={"status": "complete"}):
        assert stitch_chunks(["Muraho", "", "Mbanza"]) == "Muraho Mbanza"


def test_state_write_functions_are_mockable_for_ci_determinism():
    with patch("state.job_store.create_job", return_value={"status": "queued"}) as create_job, patch(
        "state.job_store.complete_job", return_value={"status": "complete"}
    ) as complete_job:
        queued = create_job("job-1", "audio.wav", 12.0, 1)
        complete = complete_job("job-1", "done")

    assert queued["status"] == "queued"
    assert complete["status"] == "complete"


def test_state_lifecycle_create_update_complete(tmp_path):
    """State transitions: create -> update_chunk -> complete must persist in order."""
    fake_store = {}

    def fake_fetch():
        return dict(fake_store)

    def fake_push(jobs):
        fake_store.clear()
        fake_store.update(jobs)

    with patch("state.job_store._fetch_jobs", side_effect=fake_fetch), patch(
        "state.job_store._push_jobs", side_effect=fake_push
    ):
        from state.job_store import complete_job, create_job, update_job_chunk

        record = create_job("job-001", "/audio/test.wav", 25.0, 2)
        assert record["status"] == "running"
        assert record["total_chunks"] == 2

        record = update_job_chunk("job-001", 0, "muraho")
        assert record["completed_chunks"] == 1

        record = complete_job("job-001", "muraho neza")
        assert record["status"] == "complete"
        assert record["final_transcript"] == "muraho neza"


def test_restart_recovery_seeds_cache(tmp_path):
    """On startup, load_all_jobs must return all persisted jobs for cache seeding."""
    fake_jobs = {
        "job-a": {"job_id": "job-a", "status": "complete", "final_transcript": "amakuru"},
        "job-b": {"job_id": "job-b", "status": "running", "completed_chunks": 1},
    }

    with patch("state.job_store._fetch_jobs", return_value=fake_jobs):
        from state.job_store import load_all_jobs

        result = load_all_jobs()
        assert len(result) == 2
        ids = {job["job_id"] for job in result}
        assert "job-a" in ids
        assert "job-b" in ids


def test_fail_job_records_error_reason():
    """fail_job must persist status=failed and the error reason string."""
    fake_store = {"job-err": {"job_id": "job-err", "status": "running", "error": None}}

    with patch("state.job_store._fetch_jobs", return_value=dict(fake_store)), patch(
        "state.job_store._push_jobs"
    ) as mock_push:
        from state.job_store import fail_job

        fail_job("job-err", "OOM on chunk 3")

    pushed = mock_push.call_args[0][0]
    assert pushed["job-err"]["status"] == "failed"
    assert pushed["job-err"]["error"] == "OOM on chunk 3"
