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
        queued = create_job("job-1", "audio.wav", 12.0)
        complete = complete_job("job-1", "done")

    assert queued["status"] == "queued"
    assert complete["status"] == "complete"
