#!/usr/bin/env python

# imports
import whisper
import torch

import sys

if __name__ == "__main__":
    # parse arguments
    args = sys.argv[1:]

    model_type = [0]

    # load whisper model
    model = whisper.load_model(model_type)
