import wandb
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable

from .train_utils import log_images #lambda_lr, 
from .losses import bce_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_vae(model, config, train_loader, val_loader, project_name='vae'):
    print(f"\nTraining will run on device: {device}")
    print(f"\nStarting training with config:")
    print(json.dumps(config, sort_keys=False, indent=4))

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))

    for epoch in range(config['epochs']):
        prog_str = f"{epoch+1}/{config['epochs']}"
        print(f'Epoch {prog_str}')

        model.train()
        for x, _ in iter(train_loader):

            # Pass batch through model
            x = Variable(x).to(device)
            x_hat = model(x)

            # Compute losses
            recon = -bce_loss(x_hat, x)
            elbo = recon - model.kld
            L = -torch.mean(elbo)

            L.backward()
            optimizer.step()
            optimizer.zero_grad()

# def train_vae(model, config, train_loader, val_loader, project_name='vae'):
#     print(f"\nTraining will run on device: {device}")
#     print(f"\nStarting training with config:")
#     print(json.dumps(config, sort_keys=False, indent=4))

#     # Initialize a new wandb run
#     wandb.init(project=project_name, config=config)
#     wandb.watch(model)

#     # define optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))

#     # Training Loop
#     for epoch in range(config['epochs']):

#         # Train part
#         model.train()
#         elbo_train = []
#         kld_train = []
#         recon_train = []
#         for x, i in tqdm(train_loader, disable=True, desc=f"train ({prog_str})"):
#             batch_size = x.size(0)

#             # Pass through model
#             x = Variable(x).to(device)
#             x_hat, kld = model(x)

#             # Compute loss
#             recon = -bce_loss(x_hat, x)
#             elbo = recon - kld
#             L = -torch.mean(elbo)
#             # recon = -bce_loss(x_hat, u)
#             # elbo = likelihood - kld
#             # L = -torch.mean(elbo)
#             # recon = torch.mean(bce_loss(x_hat, x).sum(1))
#             # kld = torch.mean(kld.sum(1))
#             # elbo = recon + kld

#             # Update gradients
#             L.backward()
#             optimizer.step()
#             optimizer.zero_grad()
            
#             # save losses
#             elbo_train.append(torch.mean(elbo).item())
#             kld_train.append(torch.mean(kld).item())
#             recon_train.append(torch.mean(recon).item())

#         # Log train stuff
#         wandb.log({
#             'recon_train': torch.tensor(recon_train).mean(),
#             'kl_train': torch.tensor(kld_train).mean(),
#             'elbo_train': torch.tensor(elbo_train).mean()
#         }, commit=False)

#         # Evaluate on validation set
#         model.eval()
#         with torch.no_grad():
#             loss_recon = []
#             loss_kl = []
#             loss_elbo = []

#             for x, i in tqdm(val_loader, disable=True, desc=f"val ({prog_str})"):
#                 batch_size = x.size(0)

#                # Pass through model
#                 x = Variable(x).to(device)
#                 x_hat, kld = model(x)

#                 # Compute loss
#                 recon = -bce_loss(x_hat, u)
#                 elbo = likelihood - kld
#                 L = -torch.mean(elbo)

#                 # Update gradients
#                 L.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
                
#                 # save losses
#                 elbo_train.append(torch.mean(elbo).item())
#                 kld_train.append(torch.mean(kld).item())
#                 recon_train.append(torch.mean(recon).item())
            
#             # Log validation stuff
#             wandb.log({
#                 'recon_val': torch.tensor(loss_recon).mean(),
#                 'kl_val': torch.tensor(loss_kl).mean(),
#                 'elbo_val': torch.tensor(loss_elbo).mean()
#             }, commit=False)

#             # Sample from model
#             x_sample = model.sample()

#             # Log images to wandb
#             log_images(x_hat, x_sample)
    
#     # Finalize training
#     wandb.finish()