import torch
import torch.nn as nn
from torch.optim import Adam

from models.vae import Encoder, Decoder, VariationalAutoEncoder as VAE
from datasets.mnist import train_loader

from configs.vae_config import (
    x_dim,
    hidden_dim,
    latent_dim,
    lr,
    epochs,
    weight_path,
    train_batch_size,
)

# Model Hyperparameters
cuda = True
device = torch.device("cuda" if cuda else "cpu")

# Model definition
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
model = VAE(encoder=encoder, decoder=decoder, device=device).to(device)


# Loss function
# https://statproofbook.github.io/P/mvn-dent
def loss_function(x, x_hat, mean, log_var):
    reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reconstruction_loss + KLD


optimizer = Adam(model.parameters(), lr=lr)

print("Start training VAE...")
model.train()

# Train loop
for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.view(train_batch_size, x_dim)
        x = x.to(device)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)

        overall_loss += loss.item() / train_loader.batch_size

        loss.backward()
        optimizer.step()
    # scheduler.step()

    print(
        "\tEpoch",
        epoch + 1,
        "complete!",
        "\t LR=",
        optimizer.param_groups[0]["lr"],
        "\tAverage Loss: ",
        overall_loss / (batch_idx + 1),
    )

print("Finish!!")
torch.save(model.state_dict(), weight_path)
