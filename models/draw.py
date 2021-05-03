import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image

from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


def sigmoid(x):
    return torch.where(
        x >= 0, 
        1 / (1 + torch.exp(-x)), 
        torch.exp(x) / (1 + torch.exp(x))
    )


def compute_filterbanks(A, B, log_var, gt_X, gt_Y, log_dt, N):
    # retrieve non-log versions
    var = torch.exp(log_var + 1e-8)

    # calculate grid center
    g_X = (A + 1) * (gt_X + 1) / 2
    g_Y = (B + 1) * (gt_Y + 1) / 2

    # calculate stride
    d = torch.exp(log_dt) * (np.max([A, B]) - 1) / (N - 1)
    
    # compute filters
    F_X = torch.zeros((N, A))
    F_Y = torch.zeros((N, B))

    # construct mean vectors
    mu_X = torch.linspace(
        g_X + (- N/2 - 0.5) * d, 
        g_X + (N-1 - N/2 - 0.5) * d,
        N
    )
    mu_Y = torch.linspace(
        g_Y + (- N/2 - 0.5) * d, 
        g_Y + (N-1 - N/2 - 0.5) * d,
        N
    )

    # Compute filter matrices
    for a in range(A):
        F_X[:, a] = torch.exp( -(a - mu_X)**2 / (2 * var))

    for b in range(B):
        F_Y[:, b] = torch.exp( -(b - mu_Y)**2 / (2 * var))

    # normalize filters (should each sum to 1)
    F_X = F_X / torch.sum(F_X)
    F_Y = F_Y / torch.sum(F_Y)
    return F_X, F_Y


class BaseAttention(nn.Module):
    """ No attention module """
    def __init__(self, h_dim, x_dim):
        super(BaseAttention, self).__init__()
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.write_head = nn.Linear(h_dim, x_dim)

    def read(self, x, x_hat, h):
        return torch.cat([x, x_hat], dim=1)

    def write(self, h_dec):
        return self.write_head(h_dec)


class FilterbankAttention(nn.Module):
    """ Attention module using Filterbank matrices """
    def __init__(self, h_dim, x_dim, x_shape):
        super(FilterbankAttention, self).__init__()
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.N = 10
        self.A = x_shape[0]
        self.B = x_shape[1]
        self.W_read = nn.Linear(h_dim, 5)
        self.W_write = nn.Linear(h_dim, self.N**2)

    def read(self, x, x_hat, h):
        """ Performs the read operation with attention """
        params = self.W_read(h)
        gamma = torch.exp(params[4])
        
        # filter x and x_hat
        F_X, F_Y = compute_filterbanks(
            self.A, self.B, params[2], params[0].numpy(), 
            params[1].numpy(), params[3], self.N)
        x_filt = gamma * F_Y * x * F_X.T
        x_hat_filt = gamma * F_Y * x_hat * F_X.T
        return torch.cat([x_filt, x_hat_filt], dim=1)

    def write(self, h_dec):
        params = self.W_read(h_dec)
        F_X, F_Y = compute_filterbanks(
            self.A, self.B, params[2], params[0].numpy(), 
            params[1].numpy(), params[3], self.N)
        w_t = self.W_write(h_dec)
        return F_Y.T * w_t * F_X / torch.exp(params[4])


class DRAW(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, T=10, x_shape=None):
        super(DRAW, self).__init__()
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.T = T
        
        # instantiate distribution layers
        self.variational = nn.Linear(h_dim, 2*z_dim)
        self.observation = nn.Linear(x_dim, x_dim)

        # Recurrent encoder/decoder using LSTM
        self.encoder = nn.LSTMCell(2*x_dim + h_dim, h_dim)
        self.decoder = nn.LSTMCell(z_dim, h_dim)

        # define attention module
        if x_shape == None:
            self.attention = BaseAttention(h_dim, x_dim)
        else:
            self.attention = FilterbankAttention(h_dim, x_dim, x_shape)

    def forward(self, x):
        batch_size = x.size(0)

        # define initial state as zeros
        h_enc = x.new_zeros((batch_size, self.h_dim))
        h_dec = x.new_zeros((batch_size, self.h_dim))
        c_enc = x.new_zeros((batch_size, self.h_dim))
        c_dec = x.new_zeros((batch_size, self.h_dim))
        
        # prior
        p_mu = x.new_zeros((batch_size, self.z_dim))
        p_std = x.new_ones((batch_size, self.z_dim))
        self.prior = Normal(p_mu, p_std)

        # initial canvas
        canvas = x.new_zeros((batch_size, self.x_dim))

        # initial loss
        kl = 0

        for _ in range(self.T):
            # calculate error
            x_hat = x - sigmoid(canvas)

            # use attention to read image
            r_t = self.attention.read(x, x_hat, h_dec)

            # pass throuygh encoder
            h_enc, c_enc = self.encoder(
                torch.cat([r_t, h_dec], dim=1),
                [h_enc, c_enc]
            )

            # sample from distribution
            q_mu, q_log_std = torch.split(
                self.variational(h_enc), self.z_dim, dim=1
            )
            Q_t = Normal(q_mu, torch.exp(q_log_std)) 
            z_t = Q_t.rsample()

            # pass through decoder
            h_dec, c_dec = self.decoder(z_t, [h_dec, c_dec])

            # write on canvas
            canvas += self.attention.write(h_dec)

            # add loss
            kl += kl_divergence(Q_t, self.prior)

        # reconstruction
        x_mu = self.observation(canvas)
        return [x_mu, kl]

    def sample(self, z=None):
        """ Generate a sample from the distribution """
        if z is None:
            z = self.prior.sample()
        batch_size = z.size(0)

        # initial 
        canvas = z.new_zeros((batch_size, self.x_dim))
        h_dec = z.new_zeros((batch_size, self.h_dim))
        c_dec = z.new_zeros((batch_size, self.h_dim))

        for _ in range(self.T):
            h_dec, c_dec = self.decoder(z, [h_dec, c_dec])
            canvas += self.attention.write(h_dec)
        
        x_mu = self.observation(canvas)
        return x_mu







