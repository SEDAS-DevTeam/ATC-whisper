import gradio as gr


def transcribe(audio):
    return "TODO: rework later"


iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=['microphone', 'upload'], type='filepath'),
    outputs='text'
)

iface.launch(share=False)
