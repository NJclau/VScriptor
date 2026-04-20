import gradio as gr

from inference.chunker import build_streaming_chunk_config
from inference.decoder import decode_long_audio
from inference.loader import get_asr_model
from inference.postprocess import stitch_chunk_transcripts
from inference.preprocess import prepare_audio_for_transcription



def transcription_text(audio_input):
    if audio_input is None or audio_input == "":
        return "please input the audio first"

    try:
        file_name = prepare_audio_for_transcription(audio_input)
    except Exception:
        return "convertion failed, try to use a .wav file format instead"

    try:
        asr_model = get_asr_model()
        stream_config = build_streaming_chunk_config(asr_model)
        transcript = decode_long_audio(file_name, asr_model, stream_config)
        return stitch_chunk_transcripts([transcript])
    except Exception:
        return "Something went wrong, please contact the admin for support"

with gr.Blocks() as demo:
    gr.Markdown("## VScriptor — Kinyarwanda ASR")
    audio_input = gr.Audio(label="Upload Audio", type="filepath")
    job_id_output = gr.Textbox(label="Job ID", interactive=False)
    transcription_output = gr.Textbox(label="Transcription", lines=6)
    status_output = gr.Textbox(label="System Status", interactive=False)
    transcribe_button = gr.Button("Transcribe")

if __name__ == '__main__':
    with gr.Blocks() as demo:
        audio_input = gr.Audio(label="Upload Audio", type="filepath")
        transcription = gr.Textbox(label="Transcription")
        transcription.interactive = True
        transcribe_button = gr.Button("Transcribe")
        transcribe_button.click(transcription_text, audio_input, transcription)

    demo.launch(server_name="0.0.0.0")
