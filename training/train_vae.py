import os
import wandb
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable

from .train_utils import log_images, lambda_lr, DeterministicWarmup
from .losses import bce_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_NAME = 'vae_model.pt'


def train_vae(model, config, train_loader, val_loader, project_name='vae'):
    print(f"\nTraining will run on device: {device}")
    print(f"\nStarting training with config:")
    print(json.dumps(config, sort_keys=False, indent=4))

    # Initialize a new wandb run
    wandb.init(project=project_name, config=config)
    wandb.watch(model)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))

    # Set learning rate scheduler
    if "lr_decay" in config:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_lr(**config["lr_decay"])
        )
    
    # linear deterministic warmup
    if config["kl_warmup"]:
        gamma = DeterministicWarmup(n=50, t_max=1)
    else:
        gamma = DeterministicWarmup(n=1, t_max=1)

    # Run training
    for epoch in range(config['epochs']):
        prog_str = f"{epoch+1}/{config['epochs']}"
        print(f'Epoch {prog_str}')
        
        # Train Epoch
        model.train()
        alpha = next(gamma)
        elbo_train = []
        kld_train = []
        recon_train = []
        for x, _ in iter(train_loader):
            batch_size = x.size(0)

            # Pass batch through model
            x = x.view(batch_size, -1)
            x = Variable(x).to(device)
            x_hat, kld = model(x)

            # Compute losses
            recon = torch.mean(bce_loss(x_hat, x))
            kl = torch.mean(kld)
            loss = recon + alpha * kl

            # Update gradients
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # save losses
            elbo_train.append(torch.mean(-loss).item())
            kld_train.append(torch.mean(kl).item())
            recon_train.append(torch.mean(recon).item())
        
        # Log train stuff
        wandb.log({
            'recon_train': torch.tensor(recon_train).mean(),
            'kl_train': torch.tensor(kld_train).mean(),
            'elbo_train': torch.tensor(elbo_train).mean()
        }, commit=False)

        # Update scheduler
        if "lr_decay" in config:
            scheduler.step()

        # Validation epoch
        model.eval()
        with torch.no_grad():
            elbo_val = []
            kld_val = []
            recon_val = []
            for x, _ in iter(val_loader):
                batch_size = x.size(0)

                # Pass batch through model
                x = x.view(batch_size, -1)
                x = Variable(x).to(device)
                x_hat, kld = model(x)

                # Compute losses
                recon = torch.mean(bce_loss(x_hat, x))
                kl = torch.mean(kld)
                loss = recon + alpha * kl

                # save losses
                elbo_val.append(torch.mean(-loss).item())
                kld_val.append(torch.mean(kld).item())
                recon_val.append(torch.mean(recon).item())
        
        # Log validation stuff
        wandb.log({
            'recon_val': torch.tensor(recon_val).mean(),
            'kl_val': torch.tensor(kld_val).mean(),
            'elbo_val': torch.tensor(elbo_val).mean()
        }, commit=False)

        # Sample from model
        if isinstance(config['z_dim'], list):
            x_mu = Variable(torch.randn(16, config['z_dim'][0])).to(device)
        else:
            x_mu = Variable(torch.randn(16, config['z_dim'])).to(device)
        x_sample = model.sample(x_mu)

        # Log images to wandb
        log_images(x_hat, x_sample, epoch)
    
    # Save final model
    torch.save(model, SAVE_NAME)
    wandb.save(SAVE_NAME)

    # Finalize training
    wandb.finish()