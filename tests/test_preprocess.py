import numpy as np
import pytest
import soundfile as sf

from inference.preprocess import get_audio_duration_secs, load_and_normalise


def test_get_audio_duration_secs_returns_expected_duration(tmp_path):
    samples = np.zeros(32000, dtype=np.float32)
    audio_path = tmp_path / "duration.wav"
    sf.write(audio_path, samples, 16000)

    duration_secs = get_audio_duration_secs(str(audio_path))

    assert duration_secs == pytest.approx(2.0, abs=0.05)


def test_load_and_normalise_resamples_and_returns_target_rate(tmp_path):
    samples = np.zeros(44100, dtype=np.float32)
    audio_path = tmp_path / "resample.wav"
    sf.write(audio_path, samples, 44100)

    array, sample_rate = load_and_normalise(str(audio_path))

    assert sample_rate == 16000
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.float32


def test_load_and_normalise_missing_file_raises_file_not_found_error():
    with pytest.raises(FileNotFoundError):
        load_and_normalise("/tmp/does-not-exist.wav")
