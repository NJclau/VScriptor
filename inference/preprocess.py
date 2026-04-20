"""Audio preprocessing helpers for ASR inference."""

import os

import numpy as np
import torch
import torchaudio

from config import AUDIO_PEAK_NORMALISATION_TARGET, TARGET_SAMPLE_RATE


def get_audio_duration_secs(audio_path: str) -> float:
    """Return audio duration in seconds using metadata only."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    metadata = torchaudio.info(audio_path)
    if metadata.sample_rate <= 0:
        raise ValueError(f"Invalid sample rate in metadata: {audio_path}")

    return metadata.num_frames / metadata.sample_rate


def load_and_normalise(audio_path: str) -> tuple[np.ndarray, int]:
    """Load audio, convert to mono float32, resample to target rate, and peak-normalise."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
        sample_rate = TARGET_SAMPLE_RATE

    waveform = waveform.squeeze(0).to(torch.float32)
    peak = torch.max(torch.abs(waveform)).item()
    if peak > 0.0:
        waveform = waveform * (AUDIO_PEAK_NORMALISATION_TARGET / peak)

    return waveform.cpu().numpy().astype(np.float32, copy=False), sample_rate
