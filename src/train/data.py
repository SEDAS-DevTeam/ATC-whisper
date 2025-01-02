import torchaudio
from torchaudio.transforms import Resample
import lightning as L

from torch.utils.data import Dataset


class ATCOSIM_dataset(Dataset):
    def __init__(self, wav_path, annot_path):
        self.data_X = [] # paths to wav files
        self.data_y = [] # annotations corresponding with wav files

    def __len__(self):
        return len(self.data_X)

    def __getitem__(self):
        pass


class ATCOSIM_datamodule(L.LightningDataModule):
    pass
