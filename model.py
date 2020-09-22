import torch
import torch.nn as nn
import torch.nn.functional as F


class FDDR(nn.Module):
    def __init__(self, lag, fuzzy_degree=3):
        super(FDDR, self).__init__()

        self.autoencoder = AutoEncoder(lag * fuzzy_degree, 10)
        self.rnn = SequentialLayer(10)

    def forward(self, x):
        h = self.autoencoder(x)
        output, _ = self.rnn(h)
        return output


class AutoEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(AutoEncoder, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_size, 128),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Linear(128, 64),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Linear(64, 32),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Linear(32, 16),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Linear(16, output_size))

    def forward(self, x):
        return self.layers(x)


class FuzzyLayer(nn.Module):
    def __init__(self):
        super(FuzzyLayer, self).__init__()

    @staticmethod
    def fuzzy_function(x, m, v, eps=1e-4):
        return torch.exp(-torch.pow(x - m, 2) / (v + eps))

    def forward(self, x, mean, var):
        return torch.cat([self.fuzzy_function(x, m, v) for m, v in zip(mean, var)], -1)


class SequentialLayer(nn.Module):
    def __init__(self, n_features):
        super(SequentialLayer, self).__init__()
        self.rnn = nn.RNN(n_features, 3, 1)
        signal = torch.tensor([[-1.], [0.], [1.]])
        self.register_buffer('signal', signal)

    def forward(self, x):
        output, hidden_state = self.rnn(x)
        output = F.gumbel_softmax(output, hard=True, dim=-1)
        delta = torch.matmul(output, self.signal)
        return delta, hidden_state
