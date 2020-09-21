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

dataset = IndexDataset('./Data', lag)
dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
kmeans = KMeans(3)

fuzzy_layers = FuzzyLayer()
fddr = FDDR(lag)
optimizer = torch.optim.Adam(fddr.parameters())
reward_meter = AverageMeter(100, len(dataloader))

for e in range(10):
    with tqdm(total=len(dataloader)) as progress_bar:
        for i, (returns, fragments) in enumerate(dataloader):
            # Clustering with K-Means, computing the mean and variance of each group
            o = torch.zeros(*fragments.size(), fuzzy_degree)
            for f in range(fragments.size(1)):
                fragment = fragments[:, f].T
                label = kmeans.fit(fragment).labels_
                mean = [torch.mean(fragment[label == k]) for k in range(fuzzy_degree)]
                var = [torch.var(fragment[label == k]) for k in range(fuzzy_degree)]

                # Propagating through fuzzy operation
                o[:, f] = fuzzy_layers(fragment, mean, var)
            o = torch.flatten(o, -2)

            # Computing actions by using FDDR
            delta_prob = fddr(o)
            _, delta = torch.max(delta_prob, -1)
            delta = delta.double() - 1

            # Computing reward
            pad_delta = F.pad(delta, [1, 0])
            delta_diff = (pad_delta[:, 1:] - pad_delta[:, :-1])
            reward = torch.sum(delta * returns - c * torch.abs(delta_diff))
            reward = reward.requires_grad_()

            # Updating FDDR
            optimizer.zero_grad()
            (-reward).backward()
            optimizer.step()

            reward_meter.append(reward.item())
            progress_bar.set_description(
                '[Epoch %d][Iteration %d][Reward: %.4f]' % (e, i, reward_meter.get_average(-1)))
            progress_bar.update()
        reward_meter.step()

plt.plot(reward_meter.get_average())
plt.show()
