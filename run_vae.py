import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, \
    Lambda, RandomAffine

from models import VariationalAutoencoder
from training import train_vae

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Define config
config = {
    'batch_size': 64,
    'epochs': 250,
    'lr': 3e-4,
    'h_dim': [512, 256, 256, 256],
    'z_dim': 128, 
    'as_beta': True,
    'lr_decay': {
        'n_epochs': 500,
        'offset': 0,
        'delay': 25,
    },
    'affine': True,
    'lr_warmup': True,
}

# Define transformation
def tmp_lambda(x):
    return torch.bernoulli(x)

_train_transform = [
    ToTensor(),
    Lambda(tmp_lambda)
]
test_transform = Compose([ToTensor(), Lambda(tmp_lambda)])

if config['affine']:
    _train_transform.append(
        RandomAffine(degrees=7, translate=(0.1, 0.1), scale=(1, 1.1)),
    )
train_transform = Compose(_train_transform)

# Download binarized MNIST data
train_data = MNIST('./', train=True, download=True,transform=train_transform)
test_data = MNIST('./', train=False, download=True,transform=test_transform)

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
model = VariationalAutoencoder(config, x_dim=784).to(device)

# Train model
train_vae(model, config, train_loader, val_loader, 'vae')
