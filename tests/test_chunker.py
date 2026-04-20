import numpy as np
import soundfile as sf

from inference.chunker import chunk_audio


def test_chunk_audio_writes_at_least_one_chunk_for_short_audio(tmp_path):
    audio_path = tmp_path / "short.wav"
    output_dir = tmp_path / "chunks"
    samples = np.zeros(1600, dtype=np.float32)
    sf.write(audio_path, samples, 16000)

    chunk_paths = chunk_audio(str(audio_path), str(output_dir))

    assert len(chunk_paths) == 1
    assert (output_dir / "chunk_0000.wav").exists()


def test_chunk_audio_returns_ordered_wav_paths(tmp_path):
    audio_path = tmp_path / "long.wav"
    output_dir = tmp_path / "chunks"
    samples = np.random.uniform(-0.2, 0.2, 16000 * 30).astype(np.float32)
    sf.write(audio_path, samples, 16000)

    chunk_paths = chunk_audio(str(audio_path), str(output_dir))

    assert chunk_paths == sorted(chunk_paths)
    assert all(path.endswith(".wav") for path in chunk_paths)
