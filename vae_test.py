import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from configs.vae_config import (hidden_dim, latent_dim, test_batch_size,
                                weight_path, x_dim)
from datasets.mnist import test_loader
from models.vae import Decoder, Encoder
from models.vae import VariationalAutoEncoder as VAE

batch_size = 1
device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_latent(autoencoder, data_loader, num_batches=100):
    for i, (x, y) in enumerate(data_loader):
        x = x.view(test_batch_size, x_dim)
        z = autoencoder.encoder(x.to(device))
        z = z[0].to("cpu").detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10")
        if i > num_batches:
            break
    plt.colorbar()
    plt.savefig("tmp/latent_embeddings.png")


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


# Model definition
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
model = VAE(encoder=encoder, decoder=decoder, device=device)

model.load_state_dict(torch.load(weight_path, weights_only=True))
model.to(torch.device(device))
model.eval()

with torch.no_grad():
    for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
        x = x.view(test_loader.batch_size, x_dim)
        x = x.to(device)
        x_hat, _, _ = model(x)
        break

with torch.no_grad():
    noise = torch.randn(1, latent_dim).to(device)
    generated_images = model.decoder(noise)


save_image(generated_images.view(batch_size, 1, 28, 28), "tmp/generated_sample_2.png")


# Latent embeddings
plot_latent(model, test_loader)

# Reconstructed images
plot_reconstructed(model, r0=(-3, 3), r1=(-3, 3))
