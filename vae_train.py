import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from torch.optim import Adam

from models.vae import Encoder, Decoder, Model as VAE
from datasets.mnist import train_loader

from configs.vae_config import x_dim, hidden_dim, latent_dim, lr, epochs

# Model Hyperparameters
cuda = True
_device = torch.device("cuda" if cuda else "cpu")
batch_size = 100
x_dim = 784
hidden_dim = 400
latent_dim = 200
lr = 1e-3
epochs = 30

# Model definition
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
model = VAE(encoder=encoder, decoder=decoder, device=_device).to(_device)


# Loss function
# BCE_loss = nn.BCELoss()
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


optimizer = Adam(model.parameters(), lr=lr)
print("Start training VAE...")
model.train()

# Train loop
for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(batch_size, x_dim)
        x = x.to(_device)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)

        overall_loss += loss.item()

        loss.backward()
        optimizer.step()

    print(
        "\tEpoch",
        epoch + 1,
        "complete!",
        "\tAverage Loss: ",
        overall_loss / (batch_idx * batch_size),
    )

print("Finish!!")
