import torch
import torch.nn as nn


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


def compute_filterbank(A, B, log_var, gt_X, gt_Y, log_dt, N):
    """ Compute the two filterbank matrices

    Input
        log_var :   The log of the variance.
        gt_X    :   Used to calculate center X coordinate.
        gt_Y    :   Used to calculate center Y coordinate.
        log_dt  :   Used to calculate the stride.
        N       :   Number of points in the patch in each dimension.

    Return
        F_X     :   Horizontal filterbank matrix of size (N, A)
        F_Y     :   Vertical filterbank matrix of size (N, B)
    """ 

    # retrieve non-log versions
    var = np.exp(log_var + 1e-8)

    # calculate grid center
    g_X = (A + 1) * (gt_X + 1) / 2
    g_Y = (B + 1) * (gt_Y + 1) / 2

    # calculate stride
    d = np.exp(log_dt) * (np.max([A, B]) - 1) / (N - 1)
    
    # compute filters
    F_X = np.zeros((N, A))
    F_Y = np.zeros((N, B))

    # construct mean vectors
    mu_X = np.linspace(
        g_X + (- N/2 - 0.5) * d, 
        g_X + (N-1 - N/2 - 0.5) * d,
        N
    )
    mu_Y = np.linspace(
        g_Y + (- N/2 - 0.5) * d, 
        g_Y + (N-1 - N/2 - 0.5) * d,
        N
    )

    for a in range(A):
        F_X[:, a] = np.exp( -(a - mu_X)**2 / (2 * var))

    for b in range(B):
        F_Y[:, b] = np.exp( -(b - mu_Y)**2 / (2 * var))

    # normalize filters (should each sum to 1)
    F_X = F_X / np.sum(F_X)
    F_Y = F_Y / np.sum(F_Y)
    
    # get range 
    x_range = [
        int(np.max([0, mu_X[0]])), 
        int(np.min([A, mu_X[N-1]]))
    ]
    y_range = [
        int(np.max([0, mu_Y[0]])),
        int(np.min([B, mu_Y[N-1]]))
    ]

    # crop fitlers to fit image
    F_X = F_X[:, x_range[0]:x_range[1]]
    F_Y = F_Y[:, y_range[0]:y_range[1]]

    return F_X, F_Y, x_range, y_range
