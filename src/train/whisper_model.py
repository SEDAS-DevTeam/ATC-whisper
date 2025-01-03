#!/usr/bin/env python

# imports
import torch

# whisper import
from transformers import (
    # Whisper
    WhisperTokenizer,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,

    # Huggingface trainer
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
# WER metric import
import evaluate

# tools
import sys


#
# Whisper model
#
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

    def create_training_args(self):
        self.arguments = Seq2SeqTrainingArguments(

        )


#
# Metric
#
def create_metric(name):
    return evaluate.load(name)


def calculate(metric: evaluate.Metric, tokenizer, pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


#
# Trainer
#
def create_trainer(training_arguments):
    pass


# Runtime test
if __name__ == "__main__":
    test_whisper_pipeline = WhisperPipeline(sys.argv[1])
    test_whisper_pipeline.clean_cache()
    test_whisper_pipeline.load_pipeline()
