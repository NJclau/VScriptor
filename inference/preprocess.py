"""Audio preprocessing utilities."""

import os
import torchaudio


def get_audio_duration_secs(audio_path: str) -> float:
    """Return audio duration in seconds using file metadata."""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    info = torchaudio.info(audio_path)
    return float(info.num_frames) / float(info.sample_rate)
