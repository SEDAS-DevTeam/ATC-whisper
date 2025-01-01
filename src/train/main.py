#!/usr/bin/env python

# imports
import model
import data

import torch
import lightning
import sys

# vars
cuda = "cpu" # set cpu by default

if __name__ == "__main__":
    # parse arguments
    args = sys.argv[1:]
    model_type = args[0]
    cuda = args[1]
    checkpoint_path = args[2]
    txt_path = args[3]
    wav_path = args[4]

    # preprocess dataset and load to memory
    dataset = data.ATCOSIM_dataset(wav_path,
                                   txt_path)

    exit(0) #TODO: just for testing

    # load whisper model
    whisper_pipeline = model.WhisperPipeline(model_type,
                                             cuda,
                                             checkpoint_path)
    whisper_pipeline.clean_cache() # clean any residue cache
    whisper_pipeline.load_pipeline() # load whisper and its tokenizer
