import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

from layers import GaussianSample, GaussianMerge
from inference import log_gaussian, log_standard_gaussian

from vae import VariationalAutoencoder


class ReconstructionDecoder(nn.Module):
    """ Identical to Decoder in vae.py """
    def __init__(self, dims):
        super(ReconstructionDecoder, self).__init__()

        [z_dim, h_dim, x_dim] = dims

        neurons = [z_dim, *h_dim]
        linear_layers = [nn.Linear(neurons[i-1], neurons[i]) for i in range(1, len(neurons))]
        self.hidden = nn.ModuleList(linear_layers)

        self.reconstruction = nn.Linear(h_dim[-1], x_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.output_activation(self.reconstruction(x))
 

class Encoder(nn.Module):
    """ Ladder encoder
    Differs from standard encoder by using batch-norm and LReLU activation.
    """
    def __init__(self, dims):
        super(Encoder, self).__init__()
        [x_dim, h_dim , z_dim] = dims
        
        self.linear = nn.Linear(x_dim, h_dim)
        self.batch_norm = nn.BatchNorm1d(h_dim)
        self.sample = GaussianSample(h_dim, z_dim)

    def forward(self, x):
        x = F.leaky_relu(
            self.batch_norm(
                self.linear(x)
            ),
            0.1
        )
        return x, self.sample(x)


class Decoder(nn.Module):
    """ Ladder decoder
    Differs from standard decoder by using batch-norm and LReLU activation.
    """

    def __init__(self, dims):
        [z_dim, h_dim, x_dim] = dims

        # part 1
        self.linear_1 = nn.Linear(x_dim, h_dim)
        self.batch_norm_1 = nn.BatchNorm1d(h_dim)
        self.merge = GaussianMerge(h_dim, z_dim)

        # part 2
        self.linear_2 = nn.Linear(x_dim, h_dim)
        self.batch_norm_2 = nn.BatchNorm1d(h_dim)
        self.sample = GaussianSample(h_dim, z_dim)

    def forward(self, x, l_mu=None, l_log_var=None):
        if l_mu is None:
            z = F.leaky_relu(
                self.batch_norm_1(
                    self.linear_1(z)
                ),
                0.1
            )
            q_z, q_mu, q_log_var = self.merge(z, l_mu, l_log_var)
        
        # sample from decoder
        z = F.leaky_relu(
            self.batch_norm_2(
                self.linear_2(x)
            ),
            0.1
        )
        z, p_mu, p_log_var = self.sample(z)

        if l_mu is None:
            return z
        
        return z, (q_z, (q_mu, q_log_var), (p_mu, p_log_var))


class LadderVAE(VariationalAutoencoder):
    """Ladder Variational Autoencoder"""
    def __init__(self, dims):
        super(LadderVAE, self).__init__([x_dim, z_dim[0], h_dim])
        [x_dim, z_dim, h_dim] = dims


        # define encoder and decoder layers
        neuron_dims = [x_dim, *h_dim]
        encoder_layers = [
            Encoder([neuron_dims[i - 1], neuron_dims[i], z_dim[i - 1]])
            for i in range(1, len(neuron_dims))
        ]
        decoder_layers = [
            Decoder([z_dim[i - 1], h_dim[i - 1], z_dim[i]])
            for i in range(1, len(h_dim))
        ][::-1]

        # creade encoder, decoder and reconstruction
        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.reconstruction = ReconstructionDecoder([z_dim[0], h_dim, x_dim])

        # zero out bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        # get latent representation
        latents = []
        for encoder in self.encoder:
            x, (z, mu, log_var) = encoder(x)
            latents.append((mu, log_var))
        latents = list(reversed(latents))

        self.kld = 0
        for i, decoder in enumerate([-1, *self.decoder]):
            l_mu, l_log_var = latents[i]
            if i == 0:
                self.kld += self._kld(z, (l_mu, l_log_var))
            else:
                z, kl = decoder(z, l_mu, l_log_var)
                self.kld += self._kld(*kl)

        return self.reconstruction(z)

    def sample(self, z):
        for decoder in self.decoder:
            z = decoder(z)
        return self.reconstruction(z)

