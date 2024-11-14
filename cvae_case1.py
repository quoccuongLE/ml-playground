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
from torch.utils.data import DataLoader
from models.cvae_001 import CVAECase1 as CVAE

import torch.nn.functional as F

batch_size = 1
weight_path = "tmp/weights/cvae_120_case1.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"


def plot_latent(
    autoencoder: torch.nn.Module,
    data_loader: DataLoader,
    num_batches: int = 100,
    num_classes: int = 10,
):
    for i, (x, y) in enumerate(data_loader):
        x = x.view(test_batch_size, x_dim)
        x = x.to(device)
        # enc_label = F.one_hot(y, num_classes=num_classes)
        z = autoencoder.encoder(x)
        z = z[0].to("cpu").detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10")
        if i > num_batches:
            break
    plt.colorbar()
    plt.savefig("tmp/latent_embeddings_cvae.png")


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
encoder = dict(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, depth=3)
decoder = dict(output_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, depth=3)
model = CVAE(encoder=encoder, decoder=decoder, device=device, num_classes=10).to(device)

model.load_state_dict(torch.load(weight_path, weights_only=True))
model.to(torch.device(device))
model.eval()

# Latent embeddings
plot_latent(model, test_loader)

with torch.no_grad():
    z = torch.randn(1, latent_dim).to(device)
    dec_label = F.one_hot(torch.tensor([4]).to(torch.int64).cuda(), num_classes=model.num_classes)
    generated_images = model.decoder(torch.cat((z, dec_label), dim=1))

save_image(generated_images.view(batch_size, 1, 28, 28), "tmp/generated_sample_cvae_001.png")
