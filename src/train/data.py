import torchaudio
import torch
import os

from torch.utils.data import Dataset


class ATCOSIM_dataset(Dataset):
    def __init__(self, wav_path, annot_path):
        self.data_X = [] # paths to wav files
        self.data_y = [] # annotations corresponding with wav files

        # setup wav files list
        for category in os.listdir(annot_path):
            print(category)

        # setup annot files list
        for category in os.listdir(wav_path):
            pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass
