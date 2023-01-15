import torch
import numpy as np

def LunarLander_pos(state, dtype, device):

    costs = torch.tensor(np.zeros(state.shape[0]), dtype=dtype).to(device)
    idx = (-0.15 > state[:, 0]) | (0.15 < state[:, 0])
    costs[idx] = 1

    return costs


def LunarLander_vel(state, dtype, device):

    costs = torch.tensor(np.zeros(state.shape[0]), dtype=dtype).to(device)
    idx = torch.abs(state[:, 2]) + torch.abs(state[:, 3]) > 1
    costs[idx] = 1

    return costs

