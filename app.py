"""
Gradio UI — stateless execution surface.
"""

import asyncio
import hashlib
import time
import tempfile
import logging

import gradio as gr
import torch

from config import CACHE_POLL_INTERVAL_SECS
from inference.preprocess import get_audio_duration_secs, load_and_normalise
from inference.chunker import chunk_audio
from inference.loader import get_model_and_processor
from inference.decoder import decode_logits
from inference.postprocess import stitch_chunks
from state import job_store, cache
from state.write_queue import enqueue_write, configure_flush, start_queue_worker, queue_depth

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Seed cache from Hub on startup
try:
    _startup_jobs = job_store.load_all_jobs()
    cache.seed_from_store({j["job_id"]: j for j in _startup_jobs})
except Exception as e:
    logger.error(f"Failed to seed cache: {e}")

# Global initialization
configure_flush(job_store._push_job)


def _make_job_id(audio_path: str) -> str:
    """Deterministic job ID from file content hash + timestamp."""
    with open(audio_path, "rb") as f:
        content_hash = hashlib.sha256(f.read(4096)).hexdigest()[:12]
    return f"{content_hash}-{int(time.time())}"


async def transcribe(audio_path: str | None, progress=gr.Progress()) -> str:
    """Main transcription handler. Called by Gradio on button click."""
    if not audio_path:
        return "Please upload an audio file first."

    try:
        duration = get_audio_duration_secs(audio_path)
    except Exception:
        return "Failed to read audio duration."

    job_id = _make_job_id(audio_path)

    try:
        progress(0.05, desc="Loading model...")
        model, processor = get_model_and_processor()

        with tempfile.TemporaryDirectory() as tmp_dir:
            progress(0.10, desc="Chunking audio...")
            chunk_paths = chunk_audio(audio_path, tmp_dir)
            total = len(chunk_paths)

            record = {
                "job_id": job_id,
                "audio_path": audio_path,
                "duration_secs": duration,
                "total_chunks": total,
                "completed_chunks": 0,
                "chunk_transcripts": [],
                "status": "running",
                "created_at": int(time.time()),
                "final_transcript": None,
                "error": None,
            }

            await enqueue_write(job_id, record)
            cache.put(job_id, record)

            transcripts = []
            for i, chunk_path in enumerate(chunk_paths):
                progress(0.10 + 0.85 * (i / total), desc=f"Transcribing chunk {i + 1}/{total}...")
                array, sr = load_and_normalise(chunk_path)
                inputs = processor(array, sampling_rate=sr, return_tensors="pt")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    logits = model(**inputs).logits
                text = decode_logits(logits, processor)
                transcripts.append(text)

                record["chunk_transcripts"].append(text)
                record["completed_chunks"] = i + 1
                await enqueue_write(job_id, record)
                cache.put(job_id, record)

            progress(0.97, desc="Finalising...")
            final = stitch_chunks(transcripts)
            record["status"] = "complete"
            record["final_transcript"] = final
            record["completed_at"] = int(time.time())
            await enqueue_write(job_id, record)
            cache.put(job_id, record)

        return final

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        fail_record = {"job_id": job_id, "status": "failed", "error": str(e), "failed_at": int(time.time())}
        await enqueue_write(job_id, fail_record)
        cache.put(job_id, fail_record)
        return f"Transcription failed. Job ID: {job_id}"


def _system_status_html() -> str:
    """Lightweight status string read from volatile cache."""
    status = cache.get_system_status()
    active, queued = status["active_jobs"], status["queued_jobs"]
    level = "low" if active == 0 else "medium" if active < 2 else "high"
    color = {"low": "#10B981", "medium": "#F59E0B", "high": "#E11D48"}[level]
    q_depth = queue_depth()
    queue_suffix = f" (Hub Queue: {q_depth})" if q_depth > 0 else ""
    return f'<span style="color:{color};font-weight:600;">System: {level.upper()} — {active} active, {queued} queued{queue_suffix}</span>'


if __name__ == "__main__":
    with gr.Blocks(title="MbanzaNLP — Kinyarwanda ASR") as demo:
        gr.Markdown("## Kinyarwanda Speech Transcription")
        with gr.Row():
            audio_input = gr.Audio(label="Upload Audio", type="filepath")
            with gr.Column():
                transcription = gr.Textbox(label="Transcription", interactive=True, lines=10)
                status_html = gr.HTML(value=_system_status_html())
        transcribe_btn = gr.Button("Transcribe", variant="primary")
        transcribe_btn.click(fn=transcribe, inputs=[audio_input], outputs=[transcription])
        demo.load(fn=_system_status_html, outputs=[status_html], every=CACHE_POLL_INTERVAL_SECS)

    loop = asyncio.get_event_loop()
    loop.create_task(start_queue_worker())
    demo.launch(server_name="0.0.0.0")
