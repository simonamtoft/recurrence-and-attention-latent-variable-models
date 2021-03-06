import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda

from models import DRAW
from training import train_draw

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define config
config = {
    'batch_size': 128,
    'epochs': 2000,
    'lr': 1e-3,
    'lr_decay': {
        'n_epochs': 4000,
        'delay': 200,
        'offset': 0,
    },
    'h_dim': 256,
    'z_dim': 32, 
    'T': 10,
    'N': 12,
    'attention': 'base', # filterbank, base
    'kl_warmup': True,
}

# Define transformation
def tmp_lambda(x):
    return torch.bernoulli(x)

data_transform = Compose([
    ToTensor(),
    Lambda(tmp_lambda)
])

# Download binarized MNIST data
train_data = MNIST('./', download=True, transform=data_transform)

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

# get shape of input
data_iter = iter(train_loader)
images, labels = data_iter.next()
x_shape = images.shape[2:4]

# Instantiate model
model = DRAW(config, x_shape).to(device)

# Train model
train_draw(model, config, train_loader, val_loader, "generative-project")
