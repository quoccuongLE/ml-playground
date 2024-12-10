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
from models.experimental.cvae import CVAE

batch_size = 1
weight_path = "tmp/weights/uni_cvae_120_b_90.pth"
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
    plt.savefig("latent_embeddings_cvae_case2_b99.png")


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
prior = dict(radius=4.0, sigma_1=2.0, sigma_2=0.1)
model = CVAE(
    encoder=encoder,
    decoder=decoder,
    latent_dim=latent_dim,
    num_classes=10,
    latent_sample_num=256,
    prior=prior,
    device=device,
).to(device)

model.load_state_dict(torch.load(weight_path, weights_only=True))
model.to(torch.device(device))
model.eval()

# Latent embeddings
plot_latent(model, test_loader)
