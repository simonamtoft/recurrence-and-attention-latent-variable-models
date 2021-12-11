from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformation
def tmp_lambda(x):
    return torch.bernoulli(x)

data_transform = Compose([
    ToTensor(),
    Lambda(tmp_lambda)
])

# Download test MNIST data
test_data = MNIST('./', train=False, download=True, transform=data_transform)

# Setup data loader
kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
test_loader = DataLoader(
    test_data,
    batch_size=1,
    shuffle=True,
    **kwargs
)

# Load saved models
model_path = './results/saved_models/'
vae = torch.load(model_path + 'vae_model.pt').to(device)
lvae = torch.load(model_path + 'lvae_model.pt').to(device)

# Define loss function
bce_loss = nn.BCELoss(reduction='none').to(device)

# Go through all test batches
vae.eval()
lvae.eval()
with torch.no_grad():

    vae_recon = []
    vae_kl = []
    vae_elbo = []

    lvae_recon = []
    lvae_kl = []
    lvae_elbo = []

    for x, i in tqdm(test_loader, disable=True):
        batch_size = x.size(0)
        x = x.view(batch_size, -1).to(device)

        # VAE: Pass through model
        x_hat, kld = vae(Variable(x))
        recon = torch.mean(bce_loss(x_hat, x).sum(1))
        kl = torch.mean(kld)
        loss = recon + kl

        vae_recon.append(recon.item())
        vae_kl.append(kl.item())
        vae_elbo.append(-loss.item())

        # LVAE: Pass through model
        x_hat, kld = lvae(Variable(x))
        recon = torch.mean(bce_loss(x_hat, x).sum(1))
        kl = torch.mean(kld)
        loss = recon + kl

        lvae_recon.append(recon.item())
        lvae_kl.append(kl.item())
        lvae_elbo.append(-loss.item())

    print("\nVAE")
    print(f"Recon: {torch.tensor(vae_recon).mean()}")
    print(f"KL: {torch.tensor(vae_kl).mean()}")
    print(f"ELBO: {torch.tensor(vae_elbo).mean()}")

    print("\nLVAE")
    print(f"Recon: {torch.tensor(lvae_recon).mean()}")
    print(f"KL: {torch.tensor(lvae_kl).mean()}")
    print(f"ELBO: {torch.tensor(lvae_elbo).mean()}")
