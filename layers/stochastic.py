import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# Define reparameterization trick
def reparametrize(mu, log_var):
    # draw epsilon from N(0, 1)
    eps = Variable(torch.randn(mu.size()), requires_grad=False)

    # Ensure it is on correct device
    if mu.is_cuda:
        eps = eps.cuda()

    # std = exp(log_std) <= log_std = 0.5 * log_var
    std = log_var.mul(0.5).exp_()

    # trick: z = std*eps + mu
    z = mu.addcmul(std, eps)
    return z


class GaussianSample(nn.Module):
    """ 
    Layer that enables sampling from a Gaussian distribution
    By calling GaussianSample(x), it returns [z, mu, log_var]
    """
    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Create linear layers for the mean and log variance
        self.mu = nn.Linear(self.in_features, self.out_features)
        self.log_var = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return reparametrize(mu, log_var), mu, log_var

