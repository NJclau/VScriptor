import gradio as gr
import stt
from subprocess import run
import torch
from pydub import AudioSegment
import nemo.collections.asr as nemo_asr

from state import cache


CACHE_POLL_INTERVAL_SECS = 5


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = device.type
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_rw_conformer_ctc_large")
asr_model = asr_model.to(device)


def transcription_text(audio_input):
    if audio_input is None or audio_input == "":
        return "please input the audio first"
    file_name = audio_input.split(".")[0] + ".wav"
    file_extension = audio_input.split(".")[-1]
    try:
        if file_extension != "wav":
            run(["ffmpeg", "-i", audio_input, file_name], check=True)
            audio = AudioSegment.from_file(file_name, format="wav").set_channels(1)
        else:
            audio = AudioSegment.from_file(file_name, format=file_extension).set_channels(1)
    except Exception:
        return "convertion failed, try to use a .wav file format instead"

    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    audio.export(file_name, format="wav")
    try:
        transcription = stt.transcribe_long_audios(file_name, asr_model)
    except Exception:
        return "Something went wrong, please contact the admin for support"

    return transcription


def system_status_text() -> str:
    """Format cache-only system status for the UI widget."""
    status = cache.get_system_status()
    return (
        f"System status: {status['load_level']} | "
        f"active: {status['active_jobs']} | "
        f"queued: {status['queued_jobs']}"
    )


if __name__ == "__main__":
    with gr.Blocks() as demo:
        status_widget = gr.Markdown(value=system_status_text())
        audio_input = gr.Audio(label="Upload Audio", type="filepath")
        transcription = gr.Textbox(label="Transcription")
        transcription.interactive = True
        transcribe_button = gr.Button("Transcribe")
        transcribe_button.click(transcription_text, audio_input, transcription)
        demo.load(
            fn=system_status_text,
            outputs=status_widget,
            every=CACHE_POLL_INTERVAL_SECS,
        )

    demo.launch(server_name="0.0.0.0")
