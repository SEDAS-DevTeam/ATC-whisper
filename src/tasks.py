#!/usr/bin/env python

# imports
import shutil
from tabulate import tabulate
from pathlib import Path
import requests
from invoke import task
import time
from functools import partial

# os library import
from os import mkdir, chdir
from os.path import join, exists

# data
import yaml
from git import Repo

# conversion
from model.conversion import bin_to_ggml, pretrained_to_pt, st_to_pt, pt_to_st, pt_to_ggml, ggml_to_pt

abs_path = str(Path(__file__).parents[1])
abs_path_src = join(abs_path, "src/")

model_config_path = join(abs_path, "configs/model_config.yaml")
modeltest_config_path = join(abs_path, "configs/modeltest_config.yaml")

model_path = join(abs_path_src, "model/source/fine_whisper")
openai_path = join(abs_path_src, "model/source/whisper")
whisper_cpp_path = join(abs_path_src, "model/source/whisper_cpp")

openai_url = "https://github.com/openai/whisper"
whisper_cpp_url = "https://github.com/ggerganov/whisper.cpp"


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


def add_args(command, *args):
    res_command = command
    for arg in args:
        res_command += " " + arg

    return res_command


def pull_from_repo(path):
    repo = Repo(path)
    origin = repo.remotes.origin
    pull_info = origin.pull()

    for info in pull_info:
        print(f"Branch: {info.ref.name}")
        print(f"Commit: {info.commit.hexsha}")
        print(f"Summary: {info.commit.summary}")


def check_repo(path, url, repo_name):
    if exists(path): # repo already exists, rewrite
        shutil.rmtree(path)

    print_color(colors.BLUE, f"Cloning {repo_name} repo in {path}")
    Repo.clone_from(url, path)

    print_color(colors.BLUE, "Done!")


def fetch_resource(url, path, in_chunks):
    response = requests.get(url)
    if response.status_code == 200:
        response.raise_for_status() # Raise an error for bad status codes
        with open(path, "wb") as model_file:
            if in_chunks:
                for chunk in response.iter_content(chunk_size=model_config["chunk_size"]):
                    if chunk: model_file.write(chunk)
            else:
                model_file.write(response.content)


def reformat_url(url, filename):
    return url + filename + "?download=true"


def fetch_model_file(url, path, filename, is_model=False):
    # just an top-level abstraction because I am lazy
    fetch = partial(fetch_resource,
                    reformat_url(url, filename),
                    join(path, filename))

    if is_model: fetch(True)
    else: fetch(False)


# invoke tasks
@task
def run_infer(context):
    inference_args: list = model_config["infer_args"]

    ggml_model_path = join(abs_path_src, "model/source/atc-whisper-ggml.bin")
    ggml_script_path = join(abs_path_src, "model/source/whisper-stream")
    inference_args.insert(0, f"-m {ggml_model_path}")

    context.run(add_args(ggml_script_path, *inference_args))


@task
def run_modeltest(context):
    dataset_path = modeltest_config["test_dataset_path"]
    models = modeltest_config["test_models"]

    print(dataset_path)
    print(models)


@task
def download_model_files(context):
    print_color(colors.BLUE, "Starting model download...")
    model_type = model_config["type"]
    model_source = model_config["model_source"]

    if model_type == "huggingface":
        # prefer installing pytorch .bin files with json configurations
        if exists(model_path):
            # rewrite whole directory (when trying different models, etc.)
            shutil.rmtree(model_path)
            mkdir(model_path)
        else:
            mkdir(model_path)

    print_color(colors.BLUE, "Starting model files download...")
    fetch_model_file(model_source, model_path, "added_tokens.json")
    fetch_model_file(model_source, model_path, "vocab.json")
    fetch_model_file(model_source, model_path, "config.json")

    print_color(colors.BLUE, "Fetching main model file...")
    fetch_model_file(model_source, model_path, "pytorch_model.bin", is_model=True)

    print_color(colors.BLUE, "Done!")


@task
def download_toolkits(context):
    check_repo(openai_path, openai_url, "Openai/whisper")
    check_repo(whisper_cpp_path, whisper_cpp_url, "Whisper.cpp")


@task
def build_whisper_inference(context):
    chdir(whisper_cpp_path)
    print_color(colors.BLUE, "Building whisper.cpp...")
    context.run(add_args("cmake -B build", *model_config["args"]))
    print_color(colors.BLUE, "Building whisper.cpp Release...")
    time.sleep(1) # why does this delay let whisper.cpp compile :( (otherwise, wont work for some reason)
    context.run("cmake --build build --config Release")

    # moving whisper stream binary to source
    whisper_stream_path = join(whisper_cpp_path, "build/bin/whisper-stream")
    whisper_source_path = join(abs_path_src, "model/source/whisper-stream")

    shutil.move(whisper_stream_path, whisper_source_path)

    print_color(colors.BLUE, "Done!")
    chdir(abs_path_src)


@task
def convert_model(context, conversion_type):
    st_path = join(abs_path_src, "model/source/atc-whisper.safetensors")
    pt_path = join(abs_path_src, "model/source/atc-whisper.pt")
    ggml_path = join(abs_path_src, "model/source/atc-whisper-ggml.bin")
    bin_path = join(abs_path_src, "model/source/fine_whisper/")

    #
    # NOTE and TODO: Only bin-to-ggml conversion works
    #

    if conversion_type == "st-to-ggml":
        print_color(colors.BLUE, "Converting safetensor to pytorch")
        st_to_pt(st_path, pt_path)
        print_color(colors.BLUE, "Converting pytorch to ggml")
        pt_to_ggml(pt_path, ggml_path)
    elif conversion_type == "ggml-to-st":
        print_color(colors.BLUE, "Converting ggml to pytorch")
        ggml_to_pt(ggml_path, pt_path)
        print_color(colors.BLUE, "Converting pytorch to safetensor")
        pt_to_st(pt_path, st_path)
    elif conversion_type == "pretrained-to-ggml":
        print_color(colors.BLUE, "Converting bin to pytorch")
        pretrained_to_pt(pt_path)
        print_color(colors.BLUE, "Converting pytorch to ggml")
        pt_to_ggml(pt_path, openai_path, ggml_path)
    elif conversion_type == "bin-to-ggml":
        print_color(colors.BLUE, "Converting huggingface binary model to ggml")
        bin_to_ggml(bin_path, openai_path, ggml_path)


# definitions
class colors:
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLUE = '\033[94m'


class chars:
    ARROW = '\u2192'
    TAB = '\t'


model_config = load_config(model_config_path)
modeltest_config = load_config(modeltest_config_path)

# start program
print(colors.BLUE)
print(tabulate([["ATC-whisper project manager"]]))
print(colors.ENDC)
