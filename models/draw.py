import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_filterbanks(A, B, params, N):
    # Unpack params
    gt_X = params[0]
    gt_Y = params[1]
    log_var = params[2]
    log_dt = params[3]

    # retrieve non-log versions
    var = torch.exp(log_var + 1e-8)

    # calculate grid center
    g_X = ((A + 1) * (gt_X + 1) / 2).item()
    g_Y = ((B + 1) * (gt_Y + 1) / 2).item()

    # calculate stride
    d = (torch.exp(log_dt) * (torch.max(torch.tensor([A, B])) - 1) / (N - 1)).item()
    
    # compute filters
    F_X = torch.zeros((N, A)).to(device)
    F_Y = torch.zeros((N, B)).to(device)

    # construct mean vectors
    mu_X = torch.linspace(
        g_X + (- N/2 - 0.5) * d, 
        g_X + (N-1 - N/2 - 0.5) * d,
        N
    ).to(device)
    mu_Y = torch.linspace(
        g_Y + (- N/2 - 0.5) * d, 
        g_Y + (N-1 - N/2 - 0.5) * d,
        N
    ).to(device)

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
    def __init__(self, h_dim, x_dim, x_shape, N):
        super(FilterbankAttention, self).__init__()
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.N = N
        self.A = x_shape[0]
        self.B = x_shape[1]
        self.W_read = nn.Linear(h_dim, 5)
        self.W_write = nn.Linear(h_dim, self.N**2)

    def read(self, x, x_hat, h):
        """ Performs the read operation with attention """
        params = self.W_read(h)[0]
        gamma = torch.exp(params[4])

        # reshape x
        x = torch.reshape(x, (-1, self.A, self.B))
        x_hat = torch.reshape(x_hat, (-1, self.A, self.B))

        # compute filterbank matrices
        F_X, F_Y = compute_filterbanks(
            self.A, self.B, params, self.N
        )

        # filter x and x_hat
        x_filt = gamma * torch.matmul(torch.matmul(F_Y, x), F_X.T)
        x_hat_filt = gamma * torch.matmul(torch.matmul(F_Y, x_hat), F_X.T)

        # reshape back again
        x_filt = torch.reshape(x_filt, (-1, self.N**2))
        x_hat_filt = torch.reshape(x_hat_filt, (-1, self.N**2))
        return torch.cat([x_filt, x_hat_filt], dim=1)

    def write(self, h_dec):
        params = self.W_read(h_dec)[0]
        F_X, F_Y = compute_filterbanks(
            self.A, self.B, params, self.N
        )
        w_t = torch.reshape(self.W_write(h_dec), (-1, self.N, self.N))
        c_new = torch.matmul(torch.matmul(F_Y.T, w_t), F_X) / torch.exp(params[4])
        return torch.reshape(c_new, (-1, self.A*self.B))


class DRAW(nn.Module):
    def __init__(self, config, x_shape):
        super(DRAW, self).__init__()
        self.h_dim = config['h_dim']
        self.x_dim = x_shape[0] * x_shape[1]
        self.z_dim = config['z_dim']
        self.T = config['T']
        self.N = config['N']

        # instantiate distribution layers
        self.variational = nn.Linear(self.h_dim, 2*self.z_dim)
        self.observation = nn.Linear(self.x_dim, self.x_dim)

        # define attention module
        if config['attention'] == 'base':
            self.attention = BaseAttention(self.h_dim, self.x_dim)
            enc_dim = 2*self.x_dim + self.h_dim
        elif config['attention'] == 'filterbank':
            self.attention = FilterbankAttention(self.h_dim, self.x_dim, x_shape, self.N)
            enc_dim = 2*self.N**2 + self.h_dim
        else:
            raise Exception(f"Error: Attention module '{config['attention']}' not implemented.")

        # Recurrent encoder/decoder using LSTM
        self.encoder = nn.LSTMCell(enc_dim, self.h_dim)
        self.decoder = nn.LSTMCell(self.z_dim, self.h_dim)

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
            x_hat = x - torch.sigmoid(canvas)

            # use attention to read image: N x N
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

            # write on canvas: A x B
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