import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

from handler import IndexDataset
from helper import AverageMeter
from model import FDDR, FuzzyLayer

# Parameters
c = 0.01
lag = 50
fuzzy_degree = 3

# Dataset
dataset = IndexDataset('./Data', lag)
dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

# Models
kmeans = KMeans(3)
fuzzy_layers = FuzzyLayer()
fddr = FDDR(lag)

# Tools
optimizer = torch.optim.Adam(fddr.parameters())
reward_meter = AverageMeter(100, len(dataloader))

# Training Phase
for e in range(10):
    with tqdm(total=len(dataloader)) as progress_bar:
        for i, (returns, fragments) in enumerate(dataloader):
            # Clustering with K-Means, computing the mean and variance of each group
            o = torch.zeros(*fragments.size(), fuzzy_degree, requires_grad=False)
            for f in range(fragments.size(1)):
                fragment = fragments[:, f].T
                label = kmeans.fit(fragment).labels_

                mean = torch.zeros(3, requires_grad=False)
                var = torch.zeros(3, requires_grad=False)
                for k in range(fuzzy_degree):
                    group = fragment[label == k]
                    mean[k] = torch.mean(group) if len(group) > 0 else 0
                    var[k] = torch.var(group) if len(group) > 1 else 0

                # Propagating through fuzzy operation
                o[:, f] = fuzzy_layers(fragment, mean, var)
            o = torch.flatten(o, -2)

            # Computing actions by using FDDR
            delta = fddr(o).double().squeeze(-1)

            # Computing reward
            pad_delta = F.pad(delta, [1, 0])
            delta_diff = (pad_delta[:, 1:] - pad_delta[:, :-1])
            reward = torch.sum(delta * returns - c * torch.abs(delta_diff))

            # Updating FDDR
            optimizer.zero_grad()
            (-reward).backward()
            optimizer.step()

            # Recording and showing the information
            reward_meter.append(reward.item())
            progress_bar.set_description(
                '[Epoch %d][Iteration %d][Reward: %.4f]' % (e, i, reward_meter.get_average(-1)))
            progress_bar.update()
        reward_meter.step()

# Plot the reward curve
plt.plot(reward_meter.get_average())
plt.show()
