import torchaudio
from torchaudio.transforms import Resample
import lightning as L

import torch

from torch.utils.data import Dataset


class ATCOSIM_dataset(Dataset):
    def __init__(self, wav_path, annot_path):
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

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self):
        pass


class ATCOSIM_datamodule(L.LightningDataModule):
    pass
