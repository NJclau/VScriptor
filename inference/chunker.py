"""Audio chunking utilities for long-form transcription."""

import os

import soundfile as sf

from config import CHUNK_DURATION_MS, OVERLAP_DURATION_MS, TARGET_SAMPLE_RATE
from inference.preprocess import load_and_normalise


def chunk_audio(audio_path: str, output_dir: str) -> list[str]:
    """Split audio into overlapping WAV chunks and return ordered chunk paths."""
    os.makedirs(output_dir, exist_ok=True)

    audio_array, sample_rate = load_and_normalise(audio_path)
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"Unexpected sample rate {sample_rate}; expected {TARGET_SAMPLE_RATE}.")

    chunk_size = int((CHUNK_DURATION_MS / 1000) * sample_rate)
    overlap_size = int((OVERLAP_DURATION_MS / 1000) * sample_rate)
    step_size = max(1, chunk_size - overlap_size)

    total_samples = audio_array.shape[0]
    chunk_paths: list[str] = []

    if total_samples <= chunk_size:
        chunk_path = os.path.join(output_dir, "chunk_0000.wav")
        sf.write(chunk_path, audio_array, sample_rate)
        return [chunk_path]

    chunk_index = 0
    start_sample = 0
    while start_sample < total_samples:
        end_sample = min(start_sample + chunk_size, total_samples)
        chunk = audio_array[start_sample:end_sample]
        if chunk.size == 0:
            break

        chunk_path = os.path.join(output_dir, f"chunk_{chunk_index:04d}.wav")
        sf.write(chunk_path, chunk, sample_rate)
        chunk_paths.append(chunk_path)

        if end_sample >= total_samples:
            break

        start_sample += step_size
        chunk_index += 1

    return chunk_paths
