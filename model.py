import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class FDDR(nn.Module):
    def __init__(self, lag, fuzzy_degree=3):
        super(FDDR, self).__init__()

        self.fuzzy_layer = FuzzyLayer(fuzzy_degree)
        self.autoencoder = AutoEncoder(lag * fuzzy_degree, 10)
        self.rnn = SequentialLayer(10)

    def forward(self, x):
        h1 = self.fuzzy_layer(x)
        h2 = self.autoencoder(h1)
        output, _ = self.rnn(h2)
        return output


class AutoEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(AutoEncoder, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, 512),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Linear(512, 256),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Linear(256, 128),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Linear(128, 64),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Linear(64, output_size),
                                    nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


class FuzzyLayer(nn.Module):
    def __init__(self, fuzzy_degree):
        super(FuzzyLayer, self).__init__()
        self.fuzzy_degree = fuzzy_degree

    @staticmethod
    def fuzzy_function(x, m, v, eps=1e-4):
        return torch.exp(-torch.pow(x - m, 2) / (v + eps))

    @staticmethod
    def groupby(array, label):
        perm = label.argsort()
        sorted_array, sorted_label = array[perm], label[perm]
        _, label_count = torch.unique(sorted_label, return_counts=True)
        groups = torch.split(sorted_array, label_count.tolist())
        return groups

    def forward(self, x):
        # Clustering with K-Means, computing the mean and variance of each group
        o = torch.zeros(*x.size(), self.fuzzy_degree)
        for i in range(x.size(1)):
            fragment = x[:, i].T
            label = KMeans(self.fuzzy_degree).fit(fragment).labels_
            label = torch.tensor(label)

            groups = self.groupby(fragment, label)
            mean = [torch.mean(group) for group in groups]
            var = [torch.var(group, unbiased=False) for group in groups]

            # Propagating through fuzzy operation
            o[:, i] = torch.cat([self.fuzzy_function(fragment, m, v) for m, v in zip(mean, var)], -1)
        return torch.flatten(o, -2)


class SequentialLayer(nn.Module):
    def __init__(self, n_features):
        super(SequentialLayer, self).__init__()
        self.rnn = nn.LSTM(n_features, 3, 1)
        signal = torch.tensor([[-1.], [0.], [1.]])
        self.register_buffer('signal', signal)

    def forward(self, x):
        output, hidden_state = self.rnn(x)
        output = F.gumbel_softmax(output, hard=True, dim=-1)
        delta = torch.matmul(output, self.signal)
        return delta, hidden_state
