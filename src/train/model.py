#!/usr/bin/env python

# imports
import torch
import lightning

# whisper import
from transformers import (
    WhisperTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration
)

# tools
import sys


class WhisperPipeline:
    def __init__(self,
                 model_type,
                 cuda=True,
                 checkpoint_path=None):
        self.model_type = model_type
        self.model_id = "openai/whisper-" + self.model_type
        self.cuda = cuda
        self.checkpoint_path = checkpoint_path

    def clean_cache(self):
        if self.cuda == "cuda":
            torch.cuda.empty_cache()

    def load_pipeline(self):
        print(f"Checking from transformers URI: {self.model_id}")
        # just check if all model parts work fine
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_id)
        self.tokenizer = WhisperTokenizer.from_pretrained(self.model_id)
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_id)

        # try setting some vars
        self.model.generation_config.language = "english"
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None

        print("Whisper loaded successfully")


if __name__ == "__main__":
    test_whisper_pipeline = WhisperPipeline(sys.argv[1])
    test_whisper_pipeline.clean_cache()
    test_whisper_pipeline.load_pipeline()
