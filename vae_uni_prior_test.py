import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from configs.vae_config import (
    hidden_dim,
    latent_dim,
    test_batch_size,
    weight_path,
    x_dim,
)
from datasets.mnist import test_loader
from models.experimental.vae import VAE

batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"

# torch.manual_seed(999)
weight_path = "tmp/weights/vae_universal_prior_120.pth"
# Model Hyperparameters
cuda = True
device = torch.device("cuda" if cuda else "cpu")

# Model definition
encoder = dict(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, depth=3)
decoder = dict(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim, depth=3)
model = VAE(encoder=encoder, decoder=decoder, device=device, latent_dim=latent_dim).to(
    device
)

def plot_latent(autoencoder, data_loader, num_batches=100):
    for i, (x, y) in enumerate(data_loader):
        x = x.view(test_batch_size, x_dim)
        z = autoencoder.encoder(x.to(device))
        z = z[0].to("cpu").detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10")
        if i > num_batches:
            break
    plt.colorbar()
    plt.savefig("tmp/uniprior_vae_latent_embeddings.png")


def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28
    img = np.zeros((n * w, n * w))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape(28, 28).to("cpu").detach().numpy()
            img[(n - 1 - i) * w : (n - 1 - i + 1) * w, j * w : (j + 1) * w] = x_hat
    plt.clf()
    plt.imshow(img, extent=[*r0, *r1])
    plt.savefig("tmp/reconstructed.png")
    # save_image(img, "test.png")

# Latent embeddings
plot_latent(model, test_loader)

# Reconstructed images
# plot_reconstructed(model, r0=(-3, 3), r1=(-3, 3))
