import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

from layers import GaussianSample
from inference import log_gaussian, log_standard_gaussian


class Encoder(nn.Module):
    """
        Inference Network.

        Infer the probability distribtuion p(z|x) from the data by
        fitting a variational distribtuion q(z|x).
    
    Inputs:
        dims (array) :  Dimensions of the networks on the form 
                        [input_dim, [hidden_dims], latent_dim] 
    Returns:
        Tuple of (z, mu, log(sigma^2))
    """
    def __init__(self, dims, sample_layer=GaussianSample):
        super(Encoder, self).__init__()

        # Setup network dimensions
        [x_dim, h_dim, z_dim] = dims
        neuron_dims = [x_dim, *h_dim]

        # Define the hidden layer as a stack of linear layers
        linear_layers = []
        for i in range(1, len(neuron_dims)):
            linear_layers.append(
                nn.Linear(neuron_dims[i - 1], neuron_dims[i])
            )
        self.hidden = nn.ModuleList(linear_layers)

        # Define sampling function
        self.sample = sample_layer(h_dim[-1], z_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.sample(x)


class Decoder(nn.Module):
    """
        Generative network.
        Generate samples from the original distribution p(x).
    
    Inputs:
        dims (array) :  Dimensions of the networks on the form
                        [latent_dim, [hidden_dims], input_dim] 
    
    Returns:
        decoded x
    """
    def __init__(self, dims):
        super(Decoder, self).__init__()

        # Setup network dimensions
        [z_dim, h_dim, x_dim] = dims
        neuron_dims = [z_dim, *h_dim]

        # Define the hidden layer as a stack of linear layers
        linear_layers = []
        for i in range(1, len(neuron_dims)):
            linear_layers.append(
                nn.Linear(neuron_dims[i - 1], neuron_dims[i])
            )
        self.hidden = nn.ModuleList(linear_layers)

        # Define reconstruction layer and activation function
        self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.activation(self.reconstruction(x))


class VariationalAutoencoder(nn.Module):
    """
        Variational Autoencoder model consisting of the encoder + decoder.

    Inputs:
        dims (array) :  Dimensions of the networks on the form
                        [input_dim, latent_dim, [hidden_dims]]
         
    """

    def __init__(self, dims, as_beta=False):
        super(VariationalAutoencoder, self).__init__()
        # setup network dimensions
        [x_dim, z_dim, h_dim] = dims

        # Define encoder/decoder pair 
        self.encoder = Encoder([x_dim, h_dim, z_dim])
        self.decoder = Decoder([z_dim, list(reversed(h_dim)), x_dim])
        self.kld = 0

        # zero out the biases
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

        # change to beta VAE
        if as_beta:
            self.decoder = nn.Sequential(
                nn.Linear(z_dim, h_dim[0]),
                nn.Tanh(),
                nn.Linear(h_dim[0], h_dim[1]),
                nn.Tanh(),
                nn.Linear(h_dim[1], h_dim[1]),
                nn.Tanh(),
                nn.Linear(h_dim[1], x_dim),
                nn.Sigmoid(),
            )


    def _kld(self, z, q_param, p_param=None):
        """
            Compute KL-divergence of some element z.
        
        Inputs:
            z           : sample from the q distribution
            q_param     : (mu, log_var) of the q distribution.
            p_param     : (mu, log_var) of the p distribution.
        
        Returns:
            KL-divergence of q||p 
        """

        # Define q distribution
        (mu, log_var) = q_param
        qz = log_gaussian(z, mu, log_var)

        # Define p distribution
        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)
        
        kl = qz - pz
        return kl

    def forward(self, x):
        """
            Run datapoint through model to reconstruct input
        Inputs:
            x       : input data

        Returns:
            x_mu    : reconstructed input
        """
        # fit q(z|x) to x
        z, z_mu, z_log_var = self.encoder(x)

        # compute KL-divergence
        self.kld = self._kld(z, (z_mu, z_log_var))

        # reconstruct input via. decoder
        x_mu = self.decoder(z)
        return x_mu

    def sample(self, z):
        """
            Given a z ~ N(0, I) it generates a sample from the 
            learned distribution based on p_theta(x|z)
        """
        return self.decoder(z)
