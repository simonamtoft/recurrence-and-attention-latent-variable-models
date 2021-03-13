import torch
import numpy as np
import sys
from urllib import request
from torch.utils.data import Dataset

from operator import __or__
from functools import reduce
from torch.utils.data.sampler import SubsetRandomSampler
from six.moves import urllib    # error fixining 


cuda = torch.cuda.is_available()


# Define a sampler
def get_mnist_sampler(labels, n=None, n_labels=10):
    # Only choose classes in n_labels
    classes = np.arange(n_labels)
    (indices,) = np.where(reduce(__or__, [labels == i for i in classes]))

    # Ensure uniform distribution of labels
    np.random.shuffle(indices)
    indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])

    indices = torch.from_numpy(indices)
    sampler = SubsetRandomSampler(indices)
    return sampler


# Define flatten transform as a 'lambda' func
def tmp_lambda_func(x):
    return torch.flatten(x)


def get_mnist(location="./MNIST", batch_size=64):
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
  
    flatten_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(tmp_lambda_func),
    ])

    # <<< HTTP Error 403 >>> FIXED 
    # https://stackoverflow.com/questions/60548000/getting-http-error-403-forbidden-error-when-download-mnist-dataset 
    opener = urllib.request.build_opener()
    opener.addheader = ('User-agent', 'Mozilla/5.0 Chrome/35.0.1916.47')
    urllib.request.install_opener(opener)

    # Download the train and validation data
    train = MNIST(
        location, 
        train=True, 
        download=True,
        transform=flatten_transform, 
    )
    valid = MNIST(
        location, 
        train=False, 
        download=False,
        transform=flatten_transform, 
    )

    # loaders, which perform the actual work
    train_loader = DataLoader(
        train, 
        batch_size=batch_size,
        num_workers=2,
        pin_memory=cuda,
        sampler=get_mnist_sampler(train.targets)
    )
    test_loader = DataLoader(
        valid, 
        batch_size=batch_size, 
        num_workers=2,
        pin_memory=cuda,
        sampler=get_mnist_sampler(valid.targets)
    )
    return train_loader, test_loader

