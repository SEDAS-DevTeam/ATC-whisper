#!/usr/bin/env python

# imports
from tabulate import tabulate
from pathlib import Path
import pycdlib
import requests
import shutil
from invoke import task

# os library import
from os.path import join, isfile
from os import makedirs, mkdir, remove, listdir

# data
import yaml
import pandas as pd
import torch
import torchaudio
from torchaudio.transforms import Resample

abs_path = str(Path(__file__).parents[1])
abs_path_src = join(abs_path, "src/")

# config paths
dataset_path = join(abs_path, "configs/dataset_config.yaml")
model_path = join(abs_path, "configs/model_config.yaml")

model_url = "https://huggingface.co/BUT-FIT/whisper-ATC-czech-full/resolve/main/model.safetensors?download=true"


# functions
def print_color(color, text):
    print(color + text + colors.ENDC)


def load_config(path):
    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exception:
            print("Error reading yaml file:")
            print(exception)
            exit(0)


def parse_input(input_string):
    arr = input_string.split(" ")
    command = arr[0]
    params = arr[1:]

    return command, params


def reparse_annotation(annot_string: str):
    # filler removal
    irrelevant_fillers = ["[HNOISE]", "[FRAGMENT]", "[NONSENSE]", "[UNKNOWN]", "[EMPTY]"]
    for tag in irrelevant_fillers:
        if tag in annot_string:
            annot_string = annot_string.replace(tag, "")

    # tag removal
    annot_string = annot_string.replace("<OT>", "").replace("</OT>", "")

    # word prefix removal
    annot_string = annot_string.replace("@", "")

    curr_word = ""
    prev_word = ""
    prev2_word = ""

    for char in annot_string + " ":
        if char.isspace():
            prev2_word = prev_word
            prev_word = curr_word
            curr_word = ""

            if len(prev2_word) == 0 or len(prev_word) == 0:
                continue
            else:
                if prev2_word[-1] == "=" and prev_word[0] == "=":
                    # in case of *= and =* : join together
                    annot_string = annot_string.replace(prev_word, "").replace(prev2_word, prev2_word[:-1] + prev_word[1:])
                # in case of *= : remove from sentence
                elif prev2_word[-1] == "=":
                    annot_string = annot_string.replace(prev2_word, "")
                elif prev_word[-1] == "=":
                    annot_string = annot_string.replace(prev_word, "")
        else:
            curr_word += char

    annot_string = " ".join(annot_string.split()) # take care of duplicate spaces

    return annot_string


# commands in dict
def print_info(*args):
    out_string = ""
    for item in info:
        out_string += colors.UNDERLINE + item["name"] + colors.ENDC + " " + chars.ARROW + "\n" + chars.TAB + item["desc"] + "\n"
    print(out_string)


def add_args(command, *args):
    res_command = command
    for arg in args:
        res_command += " " + arg

    return res_command


@task
def run_infer(context):
    print("Running infer!")


@task
def download_model(context):
    print_color(colors.BLUE, "Starting model download...")

    model_output_path = join(abs_path_src, "model/source/atc-whisper.safetensors")

    # Remove original one if downloaded
    try:
        remove(model_path)
    except FileNotFoundError:
        pass

    # fetch from huggingface
    response = requests.get(model_url)
    if response.status_code == 200:
        with open(model_output_path, "wb") as model_file:
            model_file.write(response.content)

    print_color(colors.BLUE, "Model download finished")


@task
def convert_model(context, conversion_type):
    st_path = join(abs_path_src, "model/source/atc-whisper.safetensors")
    pt_path = join(abs_path_src, "model/source/atc-whisper.pt")
    ggml_path = join(abs_path_src, "model/source/atc-whisper.bin")

    if conversion_type == "st-to-ggml":
        print_color(colors.BLUE, "Converting safetensor to pytorch")
        context.run()
        print_color(colors.BLUE, "Converting pytorch to ggml")
        context.run()
    elif conversion_type == "ggml-to-st":
        print_color(colors.BLUE, "Converting safetensor to pytorch")
        context.run()
        print_color(colors.BLUE, "Converting pytorch to ggml")
        context.run()


# definitions
class colors:
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLUE = '\033[94m'


class chars:
    ARROW = '\u2192'
    TAB = '\t'


info = [
    {
        "name": "exit",
        "desc": "exit project manager",
        "call": exit
    },
    {
        "name": "help",
        "desc": "prints out helper",
        "call": print_info
    },
    {
        "name": "run-infer",
        "desc": "run example infer, params: [mic, local]",
        "call": run_infer
    },
    {
        "name": "download-model",
        "desc": "download Whisper model from BUT-FIT source",
        "call": download_model
    },
    {
        "name": "convert-model",
        "desc": "convert model format",
        "call": convert_model
    }
]

# start program
print(colors.BLUE)
print(tabulate([["ATC-whisper project manager"]]))
print(colors.ENDC)
