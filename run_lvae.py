import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, \
    Lambda, RandomAffine

from models import LadderVAE
from training import train_vae

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define config
config = {
    'model': 'Ladder VAE',
    'batch_size': 128,
    'epochs': 2000,         # 2000
    'lr': 1e-3,
    # 'h_dim': [512, 256, 128, 64],
    # 'z_dim': [64, 32, 16, 8], 
    'h_dim': [512, 256, 256],
    'z_dim': [64, 32, 32], 
    'lr_decay': {
        # 'n_epochs': 500,
        # 'delay': 50,
        'n_epochs': 4000,
        'delay': 200,
        'offset': 0,
    },
    'kl_warmup': True,
    'as_beta': True,
}

# Define transformation
def tmp_lambda(x):
    return torch.bernoulli(x)

data_transform = Compose([
    ToTensor(),
    Lambda(tmp_lambda)
])

# Download binarized MNIST data
train_data = MNIST('./', train=True, download=True, transform=data_transform)

# split into training and validation sets
train_set, val_set = torch.utils.data.random_split(train_data, [50000, 10000])

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
model = LadderVAE(config, x_dim=784).to(device)

# Train model
train_vae(model, config, train_loader, val_loader, 'generative-project')
