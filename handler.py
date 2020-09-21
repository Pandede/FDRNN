import os
from glob import glob

import pandas as pd
import torch
from torch.utils.data import Dataset


class IndexDataset(Dataset):
    def __init__(self, folder_path, lag):
        self.index_file_list = sorted(glob(os.path.join(folder_path, '*.csv')))
        self.lag = lag

    def __getitem__(self, idx):
        index_data = pd.read_csv(self.index_file_list[idx])['CloseDiff'].values
        n_fragments = len(index_data) - self.lag
        fragments = torch.zeros((n_fragments, self.lag))
        for i in range(n_fragments):
            fragments[i] = torch.from_numpy(index_data[i:i + self.lag])
        return index_data[self.lag:], fragments

    def __len__(self):
        return len(self.index_file_list)