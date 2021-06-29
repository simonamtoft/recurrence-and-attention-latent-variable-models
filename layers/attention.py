import torch
import torch.nn as nn


def compute_filterbanks(A, B, log_var, gt_X, gt_Y, log_dt, N):
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
            self.A, self.B, params[2], params[0], params[1], params[3], self.N
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
            self.A, self.B, params[2], params[0], params[1]), params[3], self.N
        )
        w_t = torch.reshape(self.W_write(h_dec), (-1, self.N, self.N))
        c_new = torch.matmul(torch.matmul(F_Y.T, w_t), F_X) / torch.exp(params[4])
        return torch.reshape(c_new, (-1, self.A*self.B))
