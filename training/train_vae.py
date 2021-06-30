import wandb
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable

from .train_utils import log_images
from .losses import bce_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_vae(model, config, train_loader, val_loader, project_name='vae'):
    print(f"\nTraining will run on device: {device}")
    print(f"\nStarting training with config:")
    print(json.dumps(config, sort_keys=False, indent=4))

    # Initialize a new wandb run
    wandb.init(project=project_name, config=config)
    wandb.watch(model)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))

    for epoch in range(config['epochs']):
        prog_str = f"{epoch+1}/{config['epochs']}"
        print(f'Epoch {prog_str}')
        
        # Train Epoch
        model.train()
        elbo_train = []
        kld_train = []
        recon_train = []
        for x, _ in iter(train_loader):
            batch_size = x.size(0)

            # Pass batch through model
            x = x.view(batch_size, -1)
            x = Variable(x).to(device)
            x_hat = model(x)
            kld = model.kld

            # Compute losses
            recon = -bce_loss(x_hat, x)
            elbo = recon - kld
            L = -torch.mean(elbo)

            # Update gradients
            L.backward()
            optimizer.step()
            optimizer.zero_grad()

            # save losses
            elbo_train.append(torch.mean(elbo).item())
            kld_train.append(torch.mean(kld).item())
            recon_train.append(torch.mean(recon).item())
        
        # Log train stuff
        wandb.log({
            'recon_train': torch.tensor(recon_train).mean(),
            'kl_train': torch.tensor(kld_train).mean(),
            'elbo_train': torch.tensor(elbo_train).mean()
        }, commit=False)

        # Validation epoch
        model.eval()
        elbo_val = []
        kld_val = []
        recon_val = []
        with torch.no_grad():
            for x, _ in iter(val_loader):
                batch_size = x.size(0)

                 # Pass batch through model
                x = x.view(batch_size, -1)
                x = Variable(x).to(device)
                x_hat = model(x)
                kld = model.kld

                # Compute losses
                recon = -bce_loss(x_hat, x)
                elbo = recon - kld
                L = -torch.mean(elbo)

                # save losses
                elbo_val.append(torch.mean(elbo).item())
                kld_val.append(torch.mean(kld).item())
                recon_val.append(torch.mean(recon).item())
        
        # Log validation stuff
        wandb.log({
            'recon_val': torch.tensor(recon_val).mean(),
            'kl_val': torch.tensor(kld_val).mean(),
            'elbo_val': torch.tensor(elbo_val).mean()
        }, commit=False)

        # Sample from model
        x_mu = Variable(torch.randn(16, config['z_dim'])).to(device)
        x_sample = model.sample(x_mu)

        # Log images to wandb
        log_images(x_hat, x_sample)
    
    # Finalize training
    wandb.finish()