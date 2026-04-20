"""Simple sliding window audio chunking."""

import os

from config import CHUNK_DURATION_MS, OVERLAP_DURATION_MS
from inference.preprocess import get_audio_duration_secs, load_and_normalise
import soundfile as sf


def chunk_audio(audio_path: str, output_dir: str) -> list[str]:
    """Chunk audio into overlapping windows, save as WAV files, return list of paths."""
    os.makedirs(output_dir, exist_ok=True)
    
    data, sr = load_and_normalise(audio_path)
    duration = get_audio_duration_secs(audio_path)
    
    chunk_duration_secs = CHUNK_DURATION_MS / 1000.0
    overlap_secs = OVERLAP_DURATION_MS / 1000.0
    step = chunk_duration_secs - overlap_secs
    
    chunks = []
    start = 0.0
    idx = 0
    
    while start < duration:
        end = min(start + chunk_duration_secs, duration)
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk_data = data[start_sample:end_sample]
        
        chunk_path = os.path.join(output_dir, f"chunk_{idx:04d}.wav")
        sf.write(chunk_path, chunk_data, sr)
        chunks.append(chunk_path)
        
        start += step
        idx += 1
    
    return chunks
