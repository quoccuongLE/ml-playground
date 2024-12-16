from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def plot_latent(
    autoencoder: nn.Module,
    data_loader: DataLoader,
    x_dim: int,
    num_batches: int = 100,
    num_classes: int = 10,
    test_batch_size: int = 1,
    device: str = "cuda:0",
    save_img_path: str = "latent_embeddings_cvae_case2_b80.png",
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
    plt.savefig(save_img_path)


def plot_reconstructed(
    autoencoder: nn.Module,
    r0: Tuple[float] = (-5, 10),
    r1: Tuple[float] = (-10, 5),
    n: int = 12,
    device: str = "cuda:0",
    save_img_path: str = "tmp/reconstructed.png",
):
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
    plt.savefig(save_img_path)
