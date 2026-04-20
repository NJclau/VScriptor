import gradio as gr
from subprocess import run

from pydub import AudioSegment

import stt


def transcription_text(audio_input):
    """Preprocess an uploaded file and run transcription."""
    if audio_input is None or audio_input == "":
        return "please input the audio first"

    file_name = audio_input.rsplit(".", 1)[0] + ".wav"
    file_extension = audio_input.rsplit(".", 1)[-1]

    try:
        if file_extension != "wav":
            run(["ffmpeg", "-i", audio_input, file_name], check=True)
            audio = AudioSegment.from_file(file_name, format="wav").set_channels(1)
        else:
            audio = AudioSegment.from_file(audio_input, format=file_extension).set_channels(1)
    except Exception:
        return "convertion failed, try to use a .wav file format instead"

    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)

    audio.export(file_name, format="wav")

    try:
        transcription = stt.transcribe_long_audios(file_name)
    except Exception:
        return "Something went wrong, please contact the admin for support"

    return transcription


if __name__ == "__main__":
    with gr.Blocks() as demo:
        audio_input = gr.Audio(label="Upload Audio", type="filepath")
        transcription = gr.Textbox(label="Transcription")
        transcription.interactive = True
        transcribe_button = gr.Button("Transcribe")
        transcribe_button.click(transcription_text, audio_input, transcription)

    demo.launch(server_name="0.0.0.0")
