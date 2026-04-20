"""Gradio entrypoint with orchestration-only workflow."""

from __future__ import annotations

import os

import gradio as gr

from config import CACHE_POLL_INTERVAL_SECS, CHUNK_DURATION_SECS
from inference.chunker import chunk_audio
from inference.decoder import decode_chunk
from inference.loader import get_model_and_processor
from inference.postprocess import stitch_chunks
from inference.preprocess import get_audio_duration_secs
from scheduler.admission import AdmissionController
from state import cache
from state import job_store


try:
    _persisted_jobs = job_store.load_all_jobs()
except Exception:
    _persisted_jobs = {}
cache.seed_from_store(_persisted_jobs)

_admission = AdmissionController()
_model, _processor = get_model_and_processor()


async def transcribe_audio(audio_input: str | None) -> tuple[str, str]:
    """Transcribe one uploaded audio file with durable state transitions."""
    if not audio_input:
        return "", "Please upload an audio file first."

    duration_secs = get_audio_duration_secs(audio_input)
    job_id, queued_record = job_store.create_job(os.path.basename(audio_input), duration_secs)
    cache.put(job_id, queued_record)

    slot = await _admission.acquire(duration_secs=duration_secs)
    try:
        chunk_paths = chunk_audio(audio_input, chunk_duration_secs=CHUNK_DURATION_SECS)
        chunk_texts: list[str] = []
        for chunk_index, chunk_path in enumerate(chunk_paths):
            chunk_text = decode_chunk(chunk_path, model=_model, processor=_processor)
            chunk_texts.append(chunk_text)
            processing_record = job_store.update_job_chunk(job_id, chunk_index, chunk_text)
            cache.put(job_id, processing_record)

        final_text = stitch_chunks(chunk_texts)
        complete_record = job_store.complete_job(job_id, result_text=final_text)
        cache.put(job_id, complete_record)
        return job_id, final_text
    except Exception as error:
        failed_record = job_store.fail_job(job_id, reason=str(error))
        cache.put(job_id, failed_record)
        return job_id, f"Transcription failed for job {job_id}: {error}"
    finally:
        _admission.release(slot)


def get_status_view() -> str:
    """Render status information from volatile cache only."""
    status = cache.get_system_status()
    return (
        f"Active: {status['active_jobs']} | "
        f"Queued: {status['queued_jobs']} | "
        f"Total: {status['total_jobs']}"
    )


with gr.Blocks() as demo:
    gr.Markdown("## VScriptor — Kinyarwanda ASR")
    audio_input = gr.Audio(label="Upload Audio", type="filepath")
    job_id_output = gr.Textbox(label="Job ID", interactive=False)
    transcription_output = gr.Textbox(label="Transcription", lines=6)
    status_output = gr.Textbox(label="System Status", interactive=False)
    transcribe_button = gr.Button("Transcribe")

    transcribe_button.click(
        fn=transcribe_audio,
        inputs=[audio_input],
        outputs=[job_id_output, transcription_output],
    )
    demo.load(fn=get_status_view, outputs=[status_output], every=CACHE_POLL_INTERVAL_SECS)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
