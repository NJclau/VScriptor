import gradio as gr

from inference.loader import get_model_and_processor
from inference.preprocess import load_and_normalise, get_audio_duration_secs
from inference.chunker import chunk_audio
from inference.decoder import decode_logits
from inference.postprocess import stitch_chunks



def transcription_text(audio_input):
    if audio_input is None or audio_input == "":
        return "please input the audio first"

    try:
        audio_array, sample_rate = load_and_normalise(audio_input)
    except Exception:
        return "convertion failed, try to use a .wav file format instead"

    try:
        model, processor = get_model_and_processor()
        transcript = decode_logits(model, processor)
        return stitch_chunks([transcript])
    except Exception:
        return "Something went wrong, please contact the admin for support"

if __name__ == '__main__':
    with gr.Blocks() as demo:
        audio_input = gr.Audio(label="Upload Audio", type="filepath")
        transcription = gr.Textbox(label="Transcription")
        transcription.interactive = True
        transcribe_button = gr.Button("Transcribe")
        transcribe_button.click(transcription_text, audio_input, transcription)

    demo.launch(server_name="0.0.0.0")
