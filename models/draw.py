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


class BaseAttention(nn.Module):
    """ No attention module """
    def __init__(self, h_dim, x_dim):
        super(BaseAttention, self).__init__()
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.write_head = nn.Linear(h_dim, x_dim)

    def read(self, x, x_hat, h):
        return torch.cat([x, x_hat], dim=1)

    def write(self, x):
        return self.write_head(x)


class DRAW(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, T=10, attention=BaseAttention):
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
        self.attention = attention(h_dim, x_dim)

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







