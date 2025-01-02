import torchaudio
from torchaudio.transforms import Resample
import lightning as L

import torch
import os

from torch.utils.data import Dataset


def reparse_annotation(annot_string: str):
    # filler removal
    irrelevant_fillers = ["[HNOISE]", "[EMPTY]", "[FRAGMENT]", "[NONSENSE]", "[UNKNOWN]"]
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

            if len(prev2_word) == 0:
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


class ATCOSIM_dataset(Dataset):
    def __init__(self, wav_path, annot_path):
        def read_annot(path):
            with open(path) as file:
                return file.read()

        def read_wav(path):
            duration = 30
            target_sample_rate = 16000

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

        self.data_X = [] # paths to wav files
        self.data_y = [] # annotations corresponding with wav files

        # setup annot files list
        for category in os.listdir(annot_path):
            category_dir = os.path.join(annot_path, category)
            if os.path.isfile(category_dir):
                continue
            for sub_category in os.listdir(category_dir):
                subcategory_dir = os.path.join(category_dir, sub_category)
                for annot_source in os.listdir(subcategory_dir):
                    # get annotation full path
                    annot_source_full_path = os.path.join(subcategory_dir, annot_source)

                    # get wav full path
                    wav_source_full_path = os.path.join(
                        subcategory_dir.replace("TXTDATA", "WAVDATA"),
                        annot_source.replace(".TXT", ".WAV"))

                    annot = read_annot(annot_source_full_path)
                    audio = read_wav(wav_source_full_path)

                    # dataset modifications, excluding, etc.
                    if "<FL>" in annot or "~" in annot: # skip french and spelled characters
                        continue

                    annot = reparse_annotation(annot)

                    # TODO: modify
                    self.data_X.append(audio)
                    self.data_y.append(annot)

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self):
        pass


class ATCOSIM_datamodule(L.LightningDataModule):
    pass