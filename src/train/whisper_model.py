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

    def create_training_args(self, batch_size, epochs):
        self.arguments = Seq2SeqTrainingArguments(
            output_dir=self.checkpoint_path, #TODO: Check
            per_device_eval_batch_size=batch_size,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=0.00001,
            warmup_steps=500,
            max_steps=5000,
            gradient_checkpointing=True,
            fp16=True,
            eval_strategy="epoch",
            predict_with_generate=True,
            generation_max_length=225,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            push_to_hub=False,
            save_total_limit=2,
            report_to=["tensorboard"],
            num_train_epochs=epochs
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
def create_trainer(training_arguments,
                   loaded_model,
                   train_dataset,
                   eval_dataset,
                   metrics,
                   data_collator,
                   tokenizer
                   ):
    return Seq2SeqTrainer(
        args=training_arguments,
        model=loaded_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metrics,
        data_collator=data_collator,
        tokenizer=tokenizer
    )


# Runtime test
if __name__ == "__main__":
    test_whisper_pipeline = WhisperPipeline(sys.argv[1])
    test_whisper_pipeline.clean_cache()
    test_whisper_pipeline.load_pipeline()
