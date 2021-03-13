import torch

def bce_loss(r, x):
    """ Binary Cross Entropy Loss """
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

