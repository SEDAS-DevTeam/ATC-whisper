import gradio as gr
import torch
from transformers import pipeline


def transcribe(audio):
    return whisper_pipeline(audio)["text"]


iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=['microphone', 'upload'], type='filepath'),
    outputs='text'
)


# runtime
device = "cuda:0" if torch.cuda.is_available() else "cpu"
whisper_pipeline = pipeline(task="automatic-speech-recognition", model="BUT-FIT/whisper-ATC-czech-full", chunk_length_s=30, device=device)
whisper_pipeline.model.config.forced_decoder_ids = whisper_pipeline.tokenizer.get_decoder_prompt_ids(task="transcribe", language="english")

iface.launch(share=False)
