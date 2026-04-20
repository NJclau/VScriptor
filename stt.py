"""Chunked STT inference helpers."""

from config import CHUNK_DURATION_SECS, OVERLAP_DURATION_SECS, TARGET_SAMPLE_RATE
from inference.decoder import decode_logits_ctc_greedy
from inference.loader import get_model_and_processor


def chunk_audio(audio_path: str):
    """Split audio into overlapping chunks and return temporary chunk file paths."""
    import torchaudio

    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)

    chunk_size = int(CHUNK_DURATION_SECS * TARGET_SAMPLE_RATE)
    overlap = int(OVERLAP_DURATION_SECS * TARGET_SAMPLE_RATE)
    step = max(1, chunk_size - overlap)
    total = waveform.shape[1]

    if total <= chunk_size:
        return [waveform]

    chunks = []
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        chunk = waveform[:, start:end]
        if chunk.shape[1] == 0:
            break
        chunks.append(chunk)
        if end >= total:
            break
        start += step
    return chunks


def stitch_transcriptions(chunk_texts: list[str]) -> str:
    """Join chunk transcriptions into a single whitespace-normalized string."""
    return " ".join(part.strip() for part in chunk_texts if part and part.strip()).strip()


def transcribe_long_audios(audio_path: str) -> str:
    """Transcribe long audio by chunking and CTC-greedy decoding each chunk."""
    try:
        import torch
        no_grad_context = torch.no_grad
    except ModuleNotFoundError:
        from contextlib import nullcontext
        no_grad_context = nullcontext

    model, processor = get_model_and_processor()
    model.eval()

    chunk_waveforms = chunk_audio(audio_path)
    transcripts = []

    with no_grad_context():
        for chunk in chunk_waveforms:
            inputs = processor(chunk.squeeze(0).numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")
            logits = model(inputs.input_values).logits
            transcripts.append(decode_logits_ctc_greedy(logits, processor))

    return stitch_transcriptions(transcripts)
