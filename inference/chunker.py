"""Audio chunking utilities."""

from __future__ import annotations

import math
import os
import tempfile

import torch
import torchaudio


def chunk_audio(audio_path: str, chunk_duration_secs: int) -> list[str]:
    """Split audio into fixed-length WAV chunks and return chunk file paths."""
    waveform, sample_rate = torchaudio.load(audio_path)
    total_samples = waveform.shape[1]
    chunk_samples = int(chunk_duration_secs * sample_rate)
    if chunk_samples <= 0:
        raise ValueError("chunk_duration_secs must be positive")

    num_chunks = max(1, math.ceil(total_samples / chunk_samples))
    output_dir = tempfile.mkdtemp(prefix="stt_chunks_")
    chunk_paths: list[str] = []
    for chunk_index in range(num_chunks):
        start = chunk_index * chunk_samples
        end = min(total_samples, start + chunk_samples)
        chunk_waveform = waveform[:, start:end]
        if chunk_waveform.numel() == 0:
            chunk_waveform = torch.zeros((1, 1), dtype=waveform.dtype)
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_index:04d}.wav")
        torchaudio.save(chunk_path, chunk_waveform, sample_rate)
        chunk_paths.append(chunk_path)
    return chunk_paths
