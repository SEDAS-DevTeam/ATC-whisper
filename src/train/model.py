# imports
import whisper
import torch
import lightning


class WhisperPipeline:
    def __init__(self,
                 model_type,
                 cuda,
                 checkpoint_path):
        self.model_type = model_type
        self.cuda = cuda
        self.checkpoint_path = checkpoint_path

    def clean_cache(self):
        if self.cuda == "cuda":
            torch.cuda.empty_cache()

    def load_pipeline(self):
        self.model = whisper.load_model(self.model_type)
        self.model.load_state_dict(torch.load(self.checkpoint_path))

        self.model = self.model.to(self.cuda)
