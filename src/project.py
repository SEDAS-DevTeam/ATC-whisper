#!/usr/bin/env python

# imports
from tabulate import tabulate
import subprocess
from pathlib import Path
import pycdlib
import requests
import shutil

# os library import
from os.path import join
from os import makedirs, mkdir

abs_path = str(Path(__file__).parent)

dataset_url = "http://www2.spsc.tugraz.at/databases/ATCOSIM/.ISO/atcosim.iso"

# functions
def print_color(color, text):
    print(color + text + colors.ENDC)

def parse_input(input_string):
    arr = input_string.split(" ")
    command = arr[0]
    params = arr[1:]

    return command, params

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
            process.stderr.close()
        process.terminate()
        process.wait()

        print_color(colors.BLUE, "Process terminated")

        exit(0)

# commands in dict
def print_info():
    out_string = ""
    for item in info:
        out_string += colors.UNDERLINE + item["name"] + colors.ENDC + " " + chars.ARROW + "\n" + chars.TAB + item["desc"] + "\n"
    print(out_string)

def run_model_train():
    run_script(join(abs_path, "train/main.py"))

def run_model_infer():
    print("Running infer!")

def download_dataset():
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



    iso_output_path = join(abs_path, "dataset/atcosim.iso")
    dataset_output_path = join(abs_path, "dataset/src_data")

    # download dataset .iso file
    """
    print_color(colors.BLUE, "Starting dataset download...")

    response = requests.get(dataset_url)
    if response.status_code == 200:
        with open(iso_output_path, "wb") as dataset_file:
            dataset_file.write(response.content)
    
    print_color(colors.BLUE, "Finished dataset download")
    """

    # recreate source directory
    shutil.rmtree(dataset_output_path)
    mkdir(dataset_output_path)

    # extract dataset .iso file
    print_color(colors.BLUE, "Starting dataset extraction...")
    iso_extractor = pycdlib.PyCdlib()
    iso_extractor.open(iso_output_path)

    extract_directory(iso_extractor, "/", dataset_output_path)
    iso_extractor.close()

    print_color(colors.BLUE, "Dataset extracted, done")

def download_model():
    pass

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
        "name": "download-model",
        "desc": "Download Whisper model for its training and infer",
        "call": download_model
    }
]

# start to in program loop
print(colors.BLUE)
print(tabulate([["ATC-whisper project manager"]]))
print(colors.ENDC)

while True:
    inp = input(f"{colors.BOLD}Run command (or type 'help' for more info]) {colors.ENDC}")
    command, params = parse_input(inp)

    for command_info in info:
        if command == command_info["name"]:
            command_info["call"]()