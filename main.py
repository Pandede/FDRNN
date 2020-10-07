import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from handler import FuzzyIndexDataset
# from handler import FuzzyStreamer
from helper import AverageMeter
from model import FDDR

# Parameters
epochs = 1000
save_per_epoch = 20
c = 0.05
lag = 50
fuzzy_degree = 3

# streamer = FuzzyStreamer(lag, fuzzy_degree)
# streamer.transform('./Data/futures/train', './Data/fuzzy_futures/train')

# Dataset
train_dataset = FuzzyIndexDataset('./Data/fuzzy_futures/train', lag)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1)
test_dataset = FuzzyIndexDataset('./Data/fuzzy_futures/test', lag)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

# Models
fddr = FDDR(lag, fuzzy_degree)

# Tools
optimizer = torch.optim.Adam(fddr.parameters())
train_reward_meter = AverageMeter(epochs, len(train_dataloader))
test_reward_meter = AverageMeter(epochs, len(test_dataloader))

# Training Phase
for e in range(epochs):
    with tqdm(total=len(train_dataloader), ncols=130) as progress_bar:
        fddr.train()
        for i, (returns, fragments, mean, var) in enumerate(train_dataloader):
            # Computing actions by using FDDR
            delta = fddr(fragments, running_mean=mean, running_var=var).double().squeeze(-1)

            # Computing reward
            pad_delta = F.pad(delta, [1, 0])
            delta_diff = (pad_delta[:, 1:] - pad_delta[:, :-1])
            reward = torch.sum(delta * returns - c * torch.abs(delta_diff))

            # Updating FDDR
            optimizer.zero_grad()
            (-reward).backward()
            optimizer.step()

            # Recording and showing the information
            train_reward_meter.append(reward.item())
            progress_bar.set_description(
                '[Epoch %d][Iteration %d][Reward: train = %.4f]' %
                (e, i, train_reward_meter.get_average(-1)))
            progress_bar.update()

        fddr.eval()
        with torch.no_grad():
            for i, (returns, fragments, mean, var) in enumerate(test_dataloader):
                # Computing actions by using FDDR
                delta = fddr(fragments, running_mean=mean, running_var=var).double().squeeze(-1)

                # Computing reward
                pad_delta = F.pad(delta, [1, 0])
                delta_diff = (pad_delta[:, 1:] - pad_delta[:, :-1])
                reward = torch.sum(delta * returns - c * torch.abs(delta_diff))

                test_reward_meter.append(reward.item())

        progress_bar.set_description(
            '[Epoch %d][Iteration %d][Reward: train = %.4f, test = %.4f]' %
            (e, i, train_reward_meter.get_average(-1), test_reward_meter.get_average(-1)))

        if e % save_per_epoch == 0:
            torch.save(fddr.state_dict(), './Pickle/fddrl.pkl')
        train_reward_meter.step()
        test_reward_meter.step()

# Save the model and reward history
torch.save(fddr.state_dict(), './Pickle/fddrl.pkl')
np.save('./Pickle/fddrl_reward.npy', train_reward_meter.get_average())

# Plot the reward curve
plt.plot(train_reward_meter.get_average())
plt.plot(test_reward_meter.get_average())
plt.show()
