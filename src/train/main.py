#!/usr/bin/env python

# imports
from train import model, data

import torch
import sys

# vars
CUDA = "cpu" # set cpu by default

if __name__ == "__main__":
    # parse arguments
    args = sys.argv[1:]
    model_type = args[0]
    CUDA = args[1]

    # preprocess dataset and load to memory

    # load whisper model
    whisper_pipeline = model.WhisperPipeline(model_type,
                                             CUDA)
    whisper_pipeline.clean_cache() # clean any residue cache
    whisper_pipeline.load_pipeline() # load whisper and its tokenizer

