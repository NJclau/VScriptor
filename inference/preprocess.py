"""Audio preprocessing helpers."""

import os
from subprocess import run

from pydub import AudioSegment

from config import TARGET_SAMPLE_RATE


def prepare_audio_for_transcription(audio_input: str) -> str:
    """Convert audio to mono WAV at target sample rate and return output path."""
    if not audio_input:
        raise ValueError("Audio input is required.")

    input_root, input_extension = os.path.splitext(audio_input)
    output_path = f"{input_root}.wav"
    extension = input_extension.lower().lstrip(".")

    if extension != "wav":
        run(["ffmpeg", "-y", "-i", audio_input, output_path], check=True)
        audio = AudioSegment.from_file(output_path, format="wav")
    else:
        audio = AudioSegment.from_file(audio_input, format="wav")

    audio = audio.set_channels(1)
    if audio.frame_rate != TARGET_SAMPLE_RATE:
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)

    audio.export(output_path, format="wav")
    return output_path
