#!/usr/bin/env python

# imports
import whisper_model
import data
from functools import partial

import torch
import lightning
import sys
import time

# vars
cuda = "cpu" # set cpu by default

if __name__ == "__main__":
    # parse arguments
    args = sys.argv[1:]
    model_type = args[0]
    cuda = args[1]
    checkpoint_path = args[2]
    dataset_path = args[3]

    # preprocess dataset and load to memory
    dataset = data.ATCOSIM_dataset(dataset_path)
    train_dataset, test_dataset, val_dataset = data.split_to_subsets(dataset, [0.7, 0.2, 0.1])
    train_dataloader, test_dataloader, val_dataloader = data.create_dataloaders(train_dataset,
                                                                                test_dataset,
                                                                                val_dataset)

    # load whisper model
    whisper_pipeline = whisper_model.WhisperPipeline(model_type,
                                                     cuda)
    whisper_pipeline.clean_cache() # clean any residue cache
    whisper_pipeline.load_pipeline() # load whisper and its tokenizer

    wer_metric = whisper_model.create_metric("wer")
    compute_metric = partial(whisper_model.calculate(metric=wer_metric,
                                                     tokenizer=whisper_pipeline.tokenizer))

    whisper_pipeline.create_training_args()
    trainer = whisper_model.create_trainer(whisper_pipeline.arguments)
    trainer.train()
