import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class IndexDataset(Dataset):
    def __init__(self, folder_path, lag):
        self.index_file_list = sorted(glob(os.path.join(folder_path, '*.csv')))
        self.lag = lag

    def __getitem__(self, idx):
        self.dataframe = pd.read_csv(self.index_file_list[idx])
        index_data = self.dataframe['CloseDiff'].values
        n_fragments = len(index_data) - self.lag
        fragments = torch.zeros((n_fragments, self.lag))
        for i in range(n_fragments):
            fragments[i] = torch.from_numpy(index_data[i:i + self.lag])
        return index_data[self.lag:], fragments

    def __len__(self):
        return len(self.index_file_list)


class FuzzyIndexDataset(IndexDataset):
    def __init__(self, folder_path, lag):
        super(FuzzyIndexDataset, self).__init__(folder_path, lag)
        self.lag = lag

    def __getitem__(self, idx):
        returns, fragments = super().__getitem__(idx)
        mean = self.dataframe.filter(regex='^mean').values[self.lag:]
        var = self.dataframe.filter(regex='^var').values[self.lag:]
        return returns, fragments, mean, var


class FuzzyStreamer:
    def __init__(self, lag, fuzzy_degree):
        self.lag = lag
        self.fuzzy_degree = fuzzy_degree

    @staticmethod
    def __groupby(array, label):
        perm = label.argsort()
        sorted_array, sorted_label = array[perm], label[perm]
        label_count = np.unique(sorted_label, return_index=True)[1]
        groups = np.split(sorted_array, label_count)[1:]
        return groups

    def transform(self, src_folder, dst_folder):
        # Make the destination directory if it does not exist
        if not os.path.exists(dst_folder):
            os.makedirs(dst_folder)

        # Construct the source and destination of CSVs
        src_path_list = glob(os.path.join(src_folder, '*.csv'))
        dst_path_list = [os.path.join(dst_folder, os.path.basename(src)) for src in src_path_list]

        with tqdm(total=len(src_path_list)) as progress_bar:
            for src, dst in zip(src_path_list, dst_path_list):
                dataframe = pd.read_csv(src)
                index_data = dataframe['CloseDiff'].values

                # Reserve the space for mean and variance
                params = np.full((len(index_data), self.fuzzy_degree * 2), np.nan)
                for i in range(len(index_data) - self.lag):
                    # Clustering with K-Means, computing the mean and variance of each group
                    fragment = index_data[i:i + self.lag].reshape(-1, 1)
                    labels = KMeans(self.fuzzy_degree).fit(fragment).labels_

                    groups = self.__groupby(fragment, labels)
                    mean = [np.mean(group) for group in groups]
                    var = [np.var(group) for group in groups]

                    params[i + self.lag] = mean + var

                # Initialize the columns name
                mean_cols = ['mean_%d' % i for i in range(self.fuzzy_degree)]
                var_cols = ['var_%d' % i for i in range(self.fuzzy_degree)]

                # Convert the array into dataframe
                params = pd.DataFrame(params, columns=mean_cols + var_cols)

                # Append it to the original dataframe
                dataframe = pd.concat((dataframe, params), axis=1)
                dataframe.to_csv(dst, index=False)

                progress_bar.set_description("[Transforming '%s'...]" % src)
                progress_bar.update()
