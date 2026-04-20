"""Audio preprocessing helpers."""

import numpy as np
import soundfile as sf
import torch
import torchaudio

from config import TARGET_SAMPLE_RATE


def get_audio_duration_secs(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    info = sf.info(audio_path)
    return info.duration


def load_and_normalise(audio_path: str) -> tuple[np.ndarray, int]:
    """Load audio, resample to target rate if needed, return as float32 array and sample rate."""
    data, sr = sf.read(audio_path, dtype='float32')
    if sr != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)
        data = resampler(torch.tensor(data)).numpy()
    return data, TARGET_SAMPLE_RATE
