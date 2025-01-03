import torchaudio
import pandas as pd

from torch.utils.data import Dataset, DataLoader, random_split
import torch

import os


class ATCOSIM_dataset(Dataset):
    def __init__(self, dataset_path):
        self.annot_data_path = os.path.join(dataset_path, "dataset.csv")

        # read annot csv
        self.annot_data = pd.read_csv(self.annot_data_path)

        print("dataset created")

    def __len__(self):
        return self.annot_data.shape[0]

    def __getitem__(self, idx):
        # TODO: do tokenizing and other stuff

        n_row = self.annot_data.iloc[idx]
        wav_source = n_row["wav_source"]

        X_data, sample_rate = torchaudio.load(wav_source)
        y_data = n_row["annot"]

        return X_data, y_data


def split_to_subsets(dataset, partitions):
    generator = torch.Generator().manual_seed(42) # setting to fix seed
    return random_split(dataset, partitions, generator=generator)


def create_dataloaders(train, test, val):
    train_dataloader = DataLoader(train,
                                  batch_size=8,
                                  shuffle=True)
    # TODO: check num_workers argument

    test_dataloader = DataLoader(test,
                                 batch_size=8,
                                 shuffle=True)

    val_dataloader = DataLoader(val,
                                batch_size=8,
                                shuffle=True)
    
    print("dataloaders created")

    return train_dataloader, test_dataloader, val_dataloader
