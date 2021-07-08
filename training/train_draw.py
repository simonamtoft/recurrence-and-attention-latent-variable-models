import wandb
import json
from tqdm import tqdm
import torch
import torch.nn as nn

from .train_utils import lambda_lr, log_images, DeterministicWarmup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_draw(model, config, train_loader, val_loader, project_name='DRAW'):
    print(f"\nTraining will run on device: {device}")
    print(f"\nStarting training with config:")
    print(json.dumps(config, sort_keys=False, indent=4))

    # Initialize a new wandb run
    wandb.init(project=project_name, config=config)
    wandb.watch(model)

    # Define loss function
    bce_loss = nn.BCELoss(reduction='none').to(device)

    # Set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['lr'], betas=(0.5, 0.999)
    )

    # Set learning rate scheduler
    if "lr_decay" in config:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_lr(**config["lr_decay"])
        )

    # linear deterministic warmup
    if config["kl_warmup"]:
        N_t = int(config['epochs'] / 10)
    else:
        N_t = 1
    gamma = DeterministicWarmup(n=N_t, t_max=1)

    for epoch in range(config['epochs']):
        prog_str = f"{epoch+1}/{config['epochs']}"
        print(f"Epoch {prog_str}")

        # Prepare epoch
        loss_recon = []
        loss_kl = []
        loss_elbo = []
        alpha = next(gamma)

        # Go through all training batches
        model.train()
        for x, i in tqdm(train_loader, disable=True, desc=f"train ({prog_str})"):
            batch_size = x.size(0)

            # Pass through model
            x = x.view(batch_size, -1).to(device)
            x_hat, kld = model(x)
            x_hat = torch.sigmoid(x_hat)

            # compute losses
            reconstruction = torch.mean(bce_loss(x_hat, x).sum(1))
            kl = torch.mean(kld.sum(1))
            loss = reconstruction + alpha * kl

            # Update gradients
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # save losses
            loss_recon.append(reconstruction.item())
            loss_kl.append(kl.item())
            loss_elbo.append(-loss.item())
        
        # Log train stuff
        wandb.log({
            'recon_train': torch.tensor(loss_recon).mean(),
            'kl_train': torch.tensor(loss_kl).mean(),
            'elbo_train': torch.tensor(loss_elbo).mean()
        }, commit=False)
        
        # Update scheduler
        if "lr_decay" in config:
            scheduler.step()
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            loss_recon = []
            loss_kl = []
            loss_elbo = []
            for x, i in tqdm(val_loader, disable=True, desc=f"val ({prog_str})"):
                batch_size = x.size(0)

                # Pass through model
                x = x.view(batch_size, -1).to(device)
                x_hat, kld = model(x)
                x_hat = torch.sigmoid(x_hat)
                
                # Compute losses
                reconstruction = torch.mean(bce_loss(x_hat, x).sum(1))
                kl = torch.mean(kld.sum(1))
                loss = reconstruction + alpha * kl 

                # save losses
                loss_recon.append(reconstruction.item())
                loss_kl.append(kl.item())
                loss_elbo.append(-loss.item())
            
            # Log validation stuff
            wandb.log({
                'recon_val': torch.tensor(loss_recon).mean(),
                'kl_val': torch.tensor(loss_kl).mean(),
                'elbo_val': torch.tensor(loss_elbo).mean()
            }, commit=False)

            # Sample from model
            x_sample = model.sample()

            # Log images to wandb
            log_images(x_hat, x_sample, epoch)
    
    # Finalize training
    wandb.finish()
