"""
Gradio UI — stateless execution surface.

Design contracts:
  - The UI reads ONLY from state/cache.py (volatile read model).
  - All durable writes go through state/job_store.py via state/write_queue.py.
  - No inference logic lives here — all delegated to inference/ modules.
  - Model loading is lazy: first request triggers loader.get_model_and_processor().
  - On startup, the cache is seeded from the Hub dataset repo.
"""

import asyncio
import hashlib
import os
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
from scheduler.admission import AdmissionController
from state import job_store, cache
from state.write_queue import enqueue_write, configure_flush, start_queue_worker, queue_depth

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Seed cache from Hub on startup (recovers jobs from before last restart)
_startup_jobs = job_store.load_all_jobs()
cache.seed_from_store({j["job_id"]: j for j in _startup_jobs})

# Global initialization
configure_flush(job_store._push_job)

# Single shared admission controller — module-level singleton
_admission = AdmissionController()


def _make_job_id(audio_path: str) -> str:
    """Deterministic job ID from file content hash + timestamp."""
    with open(audio_path, "rb") as f:
        content_hash = hashlib.sha256(f.read(4096)).hexdigest()[:12]
    return f"{content_hash}-{int(time.time())}"


async def transcribe(audio_path: str | None, progress=gr.Progress()) -> str:
    """Main transcription handler. Called by Gradio on button click."""
    if not audio_path:
        return "Please upload an audio file first."

    # 1. Validate duration before touching the model
    try:
        duration = get_audio_duration_secs(audio_path)
    except FileNotFoundError:
        return "File not found. Please re-upload."

    # 2. Acquire slot — raises ValueError if over cap
    try:
        slot = await _admission.acquire(duration_secs=duration)
    except ValueError as e:
        return str(e)

    job_id = _make_job_id(audio_path)

    try:
        progress(0.05, desc="Loading model...")
        model, processor = get_model_and_processor()

        # 3. Create job record and enqueue initial write
        with tempfile.TemporaryDirectory() as tmp_dir:
            progress(0.10, desc="Chunking audio...")
            chunk_paths = chunk_audio(audio_path, tmp_dir)
            total = len(chunk_paths)

            # Create record based on job_store schema
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

            # Non-blocking enqueue for Hub push
            await enqueue_write(job_id, record)
            # Immediate local cache update for UI responsiveness
            cache.put(job_id, record)

            # 4. Process each chunk; write state transition after each
            transcripts = []
            for i, chunk_path in enumerate(chunk_paths):
                progress(
                    0.10 + 0.85 * (i / total),
                    desc=f"Transcribing chunk {i + 1}/{total}..."
                )
                array, sr = load_and_normalise(chunk_path)
                inputs = processor(
                    array, sampling_rate=sr, return_tensors="pt"
                )

                device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    logits = model(**inputs).logits
                text = decode_logits(logits, processor)
                transcripts.append(text)

                # Update record
                record["chunk_transcripts"].append(text)
                record["completed_chunks"] = i + 1

                # Enqueue Hub update
                await enqueue_write(job_id, record)
                cache.put(job_id, record)

            # 5. Stitch and finalise
            progress(0.97, desc="Finalising...")
            final = stitch_chunks(transcripts)

            record["status"] = "complete"
            record["final_transcript"] = final
            record["completed_at"] = int(time.time())

            await enqueue_write(job_id, record)
            cache.put(job_id, record)

        return final

    except Exception as e:
        logger.error(f"Transcription failed for {job_id}: {e}")
        # Construct failure record
        fail_record = {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "failed_at": int(time.time()),
        }
        # Attempt to get partial record if available
        existing = cache.get(job_id)
        if existing:
            existing.update(fail_record)
            fail_record = existing

        await enqueue_write(job_id, fail_record)
        cache.put(job_id, fail_record)
        return f"Transcription failed. Job ID: {job_id}. Error: {str(e)}"
    finally:
        _admission.release(slot)


def _system_status_html() -> str:
    """Lightweight status string read from volatile cache — never from Hub."""
    status = cache.get_system_status()
    # Align with cache.py keys
    active = status["active_jobs"]
    queued = status["queued_jobs"]

    level = "low" if active == 0 else "medium" if active < 2 else "high"
    color = {"low": "#10B981", "medium": "#F59E0B", "high": "#E11D48"}[level]

    q_depth = queue_depth()
    queue_suffix = f" (Hub Queue: {q_depth})" if q_depth > 0 else ""

    return (
        f'<span style="color:{color};font-weight:600;">'
        f'System: {level.upper()} — '
        f'{active} active, {queued} queued{queue_suffix}'
        f'</span>'
    )


if __name__ == "__main__":
    with gr.Blocks(title="MbanzaNLP — Kinyarwanda ASR") as demo:
        gr.Markdown("## Kinyarwanda Speech Transcription\nMax 3 minutes · WAV recommended")

        with gr.Row():
            audio_input = gr.Audio(label="Upload Audio", type="filepath")
            with gr.Column():
                transcription = gr.Textbox(
                    label="Transcription", interactive=True, lines=10
                )
                status_html = gr.HTML(value=_system_status_html())

        transcribe_btn = gr.Button("Transcribe", variant="primary")
        transcribe_btn.click(
            fn=transcribe,
            inputs=[audio_input],
            outputs=[transcription],
        )

        # Refresh system status from volatile cache every 5 seconds
        demo.load(
            fn=_system_status_html,
            outputs=[status_html],
            every=CACHE_POLL_INTERVAL_SECS,
        )

    # Start background worker
    loop = asyncio.get_event_loop()
    loop.create_task(start_queue_worker())

    demo.launch(server_name="0.0.0.0")
