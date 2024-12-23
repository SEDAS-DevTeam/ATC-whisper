#!/usr/bin/env python

# imports
from tabulate import tabulate
import subprocess
from pathlib import Path

abs_path = str(Path(__file__).parent)

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
    run_script(abs_path + "/train/main.py")

def run_model_infer():
    print("Running infer!")

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