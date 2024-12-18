import torch
import torch.nn as nn
from torch.optim import Adam

from models.cvae import CVAE2 as CVAE
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
device = "cuda" if torch.cuda.is_available() else "cpu"
weight_path = "tmp/weights/cvae_120_case2_test_b_90.pth"
beta = 0.90

# Model definition
encoder = dict(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, depth=3)
decoder = dict(output_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, depth=3)
model = CVAE(encoder=encoder, decoder=decoder, device=device, num_classes=10).to(device)


def loss_function(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mean: torch.Tensor,
    log_var: torch.Tensor,
    mean_per_labels: torch.Tensor,
) -> torch.Tensor:
    """_summary_

    Args:
        x (torch.Tensor): _description_
        x_hat (torch.Tensor): _description_
        mean (torch.Tensor): _description_
        log_var (torch.Tensor): _description_
        mean_per_labels (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    # https://statproofbook.github.io/P/mvn-dent
    # See my note!
    KLD = -0.5 * torch.sum(
        1 + log_var - log_var.exp() - (mean - mean_per_labels).pow(2)
    )

    return (1 - beta) * reproduction_loss + beta * KLD


optimizer = Adam(model.parameters(), lr=lr)
print("Start training C-VAE...")
model.train()

# Train loop
for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.view(train_batch_size, x_dim)
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        x_hat, mean, log_var = model(x, y)
        mean_per_labels = model.means[y]
        loss = loss_function(x, x_hat, mean, log_var, mean_per_labels)

        overall_loss += loss.item() / train_loader.batch_size

        loss.backward()
        optimizer.step()

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
