import torch
import torch.nn as nn
from torch.optim import Adam

from models.experimental.cvae import CVAE
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

torch.manual_seed(999)
# Model Hyperparameters
sampling = True
latent_sample_num = 128
if sampling:
    weight_path = f"tmp/weights/vae_universal_prior_120_L{latent_sample_num}.pth"
else:
    latent_sample_num = -1
    weight_path = f"tmp/weights/vae_universal_prior_120_L1.pth"

cuda = True
device = torch.device("cuda" if cuda else "cpu")

# Model definition
encoder = dict(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, depth=3)
decoder = dict(output_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, depth=3)
model = CVAE(
    encoder=encoder,
    decoder=decoder,
    latent_dim=latent_dim,
    device=device,
    num_classes=10,
    latent_sample_num=latent_sample_num,
).to(device)

optimizer = Adam(model.parameters(), lr=lr)
print("Start training C-VAE...")
model.train()

# Train loop
for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.view(train_batch_size, x_dim)
        x = x.to(device)
        loss = model(x, y=y, mode="train", sampling=sampling)

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
