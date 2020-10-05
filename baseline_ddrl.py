import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from handler import IndexDataset
from helper import AverageMeter
from model import DDRL

# Parameters
epochs = 1000
save_per_epoch = 20
c = 0.01
lag = 50

# Dataset
dataset = IndexDataset('Data/futures/train', lag)
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
            torch.save(ddrl.state_dict(), './Pickle/ddrl.pkl')
        reward_meter.step()

# Save the model and reward history
torch.save(ddrl.state_dict(), './Pickle/ddrl.pkl')
np.save('./Pickle/ddrl_reward.npy', reward_meter.get_average())

# Plot the reward curve
plt.plot(reward_meter.get_average())
plt.show()
