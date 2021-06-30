import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from models import VariationalAutoencoder
from training import train_vae

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download binarized MNIST data
def tmp_lambda(x):
    return torch.bernoulli(x)

mnist_data = MNIST(
    './', 
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(tmp_lambda)
    ])
)

# Define config
config = {
    'batch_size': 64,
    'epochs': 250,
    'lr': 3e-4,
    'h_dim': [512, 256, 256, 256],
    'z_dim': 128, 
    'as_beta': False
}

# split into training and validation sets
train_set, val_set = torch.utils.data.random_split(mnist_data, [50000, 10000])

# Setup data loader
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = DataLoader(
    train_set,
    batch_size=config['batch_size'],
    shuffle=True,
    **kwargs
)
val_loader = DataLoader(
    val_set,
    batch_size=config['batch_size'],
    shuffle=True,
    **kwargs
)

# Instantiate model
h_dims = config['h_dim']
z_dim = config['z_dim']
x_dim = 784
model = VariationalAutoencoder(config, x_dim).to(device)

# Train model
train_vae(model, config, train_loader, val_loader, 'vae')
