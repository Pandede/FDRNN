import os
from configparser import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from handler import IndexDataset
from helper import AverageMeter
from model import DDRL

cfg = ConfigParser()
cfg.read('./config.ini')

# Parameters
epochs = cfg.getint('default', 'epochs')
save_per_epoch = cfg.getint('default', 'save_per_epoch')
c = cfg.getfloat('default', 'c')
lag = cfg.getint('default', 'lag')

data_src = cfg.get('default', 'data_src')
log_src = cfg.get('default', 'log_src')

# Dataset
dataset = IndexDataset(os.path.join(data_src, 'futures', 'train'), lag)
dataloader = DataLoader(dataset, shuffle=False, batch_size=1)

# Models
ddrl = DDRL(lag)

# Tools
optimizer = torch.optim.Adam(ddrl.parameters())
reward_meter = AverageMeter(epochs, len(dataloader))

# Training Phase
for e in range(epochs):
    with tqdm(total=len(dataloader)) as progress_bar:
        for i, (returns, fragments) in enumerate(dataloader):
            # Computing actions by using FDDR
            delta = ddrl(fragments).double().squeeze(-1)

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

        if e % save_per_epoch == 0:
            torch.save(ddrl.state_dict(), os.path.join(log_src, 'ddrl.pkl'))
        reward_meter.step()

# Save the model and reward history
torch.save(ddrl.state_dict(), os.path.join(log_src, 'ddrl.pkl'))
np.save(os.path.join(log_src, 'ddrl_reward.npy'), reward_meter.get_average())

# Plot the reward curve
plt.plot(reward_meter.get_average())
plt.show()
