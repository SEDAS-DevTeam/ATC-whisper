#!/usr/bin/env python

# imports
from tabulate import tabulate
import subprocess
from pathlib import Path
import pycdlib
import requests
import shutil

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

dataset_url = "http://www2.spsc.tugraz.at/databases/ATCOSIM/.ISO/atcosim.iso"
model_url = "https://ggml.ggerganov.com/"


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


def run_script(command: str):
    command_split = command.split(" ")
    process = subprocess.Popen(command_split, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
    except KeyboardInterrupt:
        print("\n")
        print_color(colors.BLUE, "Exiting...")

        process.terminate()
        process.wait()
    finally:
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            for line in iter(process.stderr.readline, ''):
                print(line, end='')
            process.stderr.close()
        process.terminate()
        process.wait()

        print_color(colors.BLUE, "Process terminated")


def add_args(command, *args):
    res_command = command
    for arg in args:
        res_command += " " + arg

    return res_command


# commands in dict
def print_info(*args):
    out_string = ""
    for item in info:
        out_string += colors.UNDERLINE + item["name"] + colors.ENDC + " " + chars.ARROW + "\n" + chars.TAB + item["desc"] + "\n"
    print(out_string)


def run_model_train(*args):
    # reparse config into args
    model_type   = model_config["type"]
    cuda         = model_config["compute"]
    checkpoint   = model_config["checkpoint_path"]
    dataset_path = dataset_config["reparsed_path"]

    command = add_args(join(abs_path_src, "train/main.py"),
                       model_type,
                       cuda,
                       join(abs_path, checkpoint),
                       join(abs_path, dataset_path))
    run_script(command)


def run_model_infer(*args):
    print("Running infer!")


def download_dataset(*args):
    def extract_directory(iso: pycdlib.PyCdlib, path, output_path):
        for entry in iso.list_children(iso_path=path):
            name = entry.file_identifier().decode('utf-8')
            if name == "." or name == "..":
                continue

            child_path = join(path, name)
            if entry.is_dir():
                new_output_path = join(output_path, name)
                makedirs(new_output_path, exist_ok=True)
                extract_directory(iso, child_path, new_output_path)
            else:
                new_output_path = join(output_path, name)
                iso.get_file_from_iso(local_path=new_output_path, iso_path=child_path)

    iso_output_path = join(abs_path_src, "dataset/atcosim.iso")
    dataset_output_path = join(abs_path_src, "dataset/src_data")

    # remove existing .iso file
    try:
        remove(iso_output_path)
        shutil.rmtree(dataset_output_path)
    except FileNotFoundError:
        pass

    # recreate source directory
    mkdir(dataset_output_path)

    # download dataset .iso file
    print_color(colors.BLUE, "Starting dataset download...")

    response = requests.get(dataset_url)
    if response.status_code == 200:
        with open(iso_output_path, "wb") as dataset_file:
            dataset_file.write(response.content)

    print_color(colors.BLUE, "Finished dataset download")

    # extract dataset .iso file
    print_color(colors.BLUE, "Starting dataset extraction...")
    iso_extractor = pycdlib.PyCdlib()
    iso_extractor.open(iso_output_path)

    extract_directory(iso_extractor, "/", dataset_output_path)
    iso_extractor.close()

    print_color(colors.BLUE, "Dataset extracted, done")


def parse_dataset(*args):
    def read_annot(path):
        with open(path) as file:
            return file.read()

    def read_wav(path):
        duration = 30

        waveform, sample_rate = torchaudio.load(path)

        # resample
        if sample_rate != target_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
            waveform = resampler(waveform)

        # truncate or pad to 30 seconds
        num_samples = target_sample_rate * duration
        if waveform.size(1) > num_samples:
            waveform = waveform[:, :num_samples]  # truncate
        elif waveform.size(1) < num_samples:
            padding = num_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))  # pad

        return waveform

    print_color(colors.BLUE, "Starting dataset parsing...")

    annot_path = join(abs_path, dataset_config["txt_path"])
    reparsed_dataset_path = join(abs_path, dataset_config["reparsed_path"])
    reparsed_data_path = join(reparsed_dataset_path, "data")
    target_sample_rate = 16000

    try:
        shutil.rmtree(reparsed_dataset_path)
    except FileNotFoundError:
        pass

    # recreate source directory
    mkdir(reparsed_dataset_path)
    mkdir(reparsed_data_path)

    data_list = {
        "wav_source": [],
        "annot": []
    }

    data_i = 0

    # setup annot files list
    for category in listdir(annot_path):
        category_dir = join(annot_path, category)
        if isfile(category_dir):
            continue
        for sub_category in listdir(category_dir):
            subcategory_dir = join(category_dir, sub_category)
            for annot_source in listdir(subcategory_dir):

                # get annotation full path
                annot_source_full_path = join(subcategory_dir, annot_source)

                # get wav full path
                wav_source_full_path = join(
                    subcategory_dir.replace("TXTDATA", "WAVDATA"),
                    annot_source.replace(".TXT", ".WAV"))

                annot = read_annot(annot_source_full_path)

                #
                # Waveform modifications
                #

                wav = read_wav(wav_source_full_path)
                torchaudio.save(join(reparsed_data_path, f"data{data_i}.wav"), wav, target_sample_rate)

                #
                # Annotation modifications
                #
                if "<FL>" in annot or "~" in annot: # skip french, spelled characters
                    continue

                # rework the rest of the annot
                annot = reparse_annotation(annot)

                # last cleaning
                if len(annot) == 0:
                    continue # skip empty annots

                data_list["wav_source"].append(wav_source_full_path)
                data_list["annot"].append(annot)

                data_i += 1

    print_color(colors.BLUE, "Dataset listing done, writing to CSV...")
    data_frame = pd.DataFrame(data_list)
    data_frame.to_csv(join(reparsed_dataset_path, "dataset.csv"))
    print_color(colors.BLUE, "Done")


def download_model(*args):
    # DEPRECATED
    """
    def parse_url():
        global model_url
        return model_url + "ggml-model-whisper-" + model_config["type"] + ".bin"

    source_dir_path = join(abs_path_src, "model/source/")
    model_output_path = join(source_dir_path, model_config["type"] + ".bin")
    pt_output_path = join(source_dir_path, model_config["type"] + ".pt")

    # delete model cache (if exists)
    for entry in listdir(source_dir_path):
        if ".gitkeep" in entry:
            continue

        local_model_path = join(source_dir_path, entry)
        remove(local_model_path)

    # download model .bin file
    print_color(colors.BLUE, "Starting model download...")

    response = requests.get(parse_url())
    if response.status_code == 200:
        with open(model_output_path, "wb") as dataset_file:
            dataset_file.write(response.content)

    # finished model .bin file
    print_color(colors.BLUE, "Finished model download")

    # converting .bin to .pt for load
    print_color(colors.BLUE, "Starting model conversion...")
    command = add_args("python3", join(abs_path_src, "model/ggml_to_pt.py"), model_output_path, pt_output_path)
    run_script(command)

    print_color(colors.BLUE, "Finished model conversion")
    """

    # instead use this :)
    model_type = model_config["type"]

    command = add_args(join(abs_path_src, "train/model.py"),
                       model_type)
    run_script(command)


def stash_model(*args):
    model_old_name = args[0]
    model_new_name = args[1]

    print_color(colors.BLUE, f"Stashing model: {model_old_name} to {model_new_name} ...")

    source_dir_path = join(abs_path_src, "model/source/")
    models_dir_path = join(abs_path_src, "model/models")

    ggml_path_old = join(source_dir_path, model_old_name + ".bin")
    pt_path_old = join(source_dir_path, model_old_name + ".pt")

    ggml_path_new = join(models_dir_path, model_new_name + ".bin")
    pt_path_new = join(models_dir_path, model_new_name + ".pt")

    # move ggml file
    shutil.move(ggml_path_old, ggml_path_new)

    # move checkpoint file
    shutil.move(pt_path_old, pt_path_new)

    print_color(colors.BLUE, "Model stashed")


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
        "name": "run-train",
        "desc": "run model training sequence",
        "call": run_model_train
    },
    {
        "name": "run-infer",
        "desc": "run example infer, params: [mic, local]",
        "call": run_model_infer
    },
    {
        "name": "download-dataset",
        "desc": "Download ATCOSIM dataset for whisper training",
        "call": download_dataset
    },
    {
        "name": "parse-dataset",
        "desc": "Create a CSV from ATCOSIM dataset",
        "call": parse_dataset
    },
    {
        "name": "stash-model",
        "desc": "Stash model into /models directory for later usage",
        "call": stash_model
    },
    {
        "name": "download-model",
        "desc": "Download Whisper model from whisper.cpp source",
        "call": download_model
    },
]

# load all configs
dataset_config = load_config(dataset_path)
model_config = load_config(model_path)

# start to in program loop
print(colors.BLUE)
print(tabulate([["ATC-whisper project manager"]]))
print(colors.ENDC)

while True:
    try:
        inp = input(f"{colors.BOLD}Run command (or type 'help' for more info]) {colors.ENDC}")
        command, params = parse_input(inp)

        for command_info in info:
            if command == command_info["name"]:
                command_info["call"](*params)
    except KeyboardInterrupt:
        print("\n")
        continue
