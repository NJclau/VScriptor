import asyncio
from subprocess import run

import gradio as gr
import nemo.collections.asr as nemo_asr
import torch
from pydub import AudioSegment

import stt
from scheduler.admission import AdmissionController


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = device.type
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_rw_conformer_ctc_large")
asr_model = asr_model.to(device)
_admission = AdmissionController()


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

    slot = None
    try:
        slot = asyncio.run(_admission.acquire(duration_secs=audio.duration_seconds))
        transcription = stt.transcribe_long_audios(file_name, asr_model)
    except ValueError as error:
        return str(error)
    except Exception:
        return "Something went wrong, please contact the admin for support"
    finally:
        if slot is not None:
            _admission.release(slot)

    return transcription


if __name__ == "__main__":
    with gr.Blocks() as demo:
        audio_input = gr.Audio(label="Upload Audio", type="filepath")
        transcription = gr.Textbox(label="Transcription")
        transcription.interactive = True
        transcribe_button = gr.Button("Transcribe")
        transcribe_button.click(transcription_text, audio_input, transcription)

    demo.launch(server_name="0.0.0.0")
