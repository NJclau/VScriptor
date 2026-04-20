"""Streaming decoder for long audio files."""

from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR


def decode_long_audio(filename: str, asr_model, stream_config: dict) -> str:
    """Run frame-batch streaming decode for a long audio file."""
    asr_model.eval()
    asr_model = asr_model.to(asr_model.device)

    frame_asr = FrameBatchASR(
        asr_model=asr_model,
        frame_len=stream_config["chunk_len_secs"],
        total_buffer=stream_config["total_buffer_secs"],
        batch_size=stream_config["batch_size"],
    )
    frame_asr.read_audio_file(
        filename,
        stream_config["mid_delay"],
        stream_config["model_stride_secs"],
    )
    return frame_asr.transcribe(
        stream_config["tokens_per_chunk"],
        stream_config["mid_delay"],
    )
