import torch
import torch.nn as nn
from torch.optim import Adam, Adamax

from models.experimental.vae import VAE
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


# torch.manual_seed(999)
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
decoder = dict(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim, depth=3)
model = VAE(
    encoder=encoder,
    decoder=decoder,
    device=device,
    latent_dim=latent_dim,
    latent_sample_num=latent_sample_num,
).to(device)

optimizer = Adamax(model.parameters(), lr=lr)
print("Start training VAE...")
model.train()
for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.view(train_batch_size, x_dim)
        x = x.to(device)
        loss = model(x, mode="train", sampling=sampling)
        overall_loss += loss.item() / train_loader.batch_size
        # print(loss.item())

        optimizer.zero_grad()
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
