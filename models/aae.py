from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions.prior import BasePrior
from .distributions import factory as prior_factory
from .vae import VariationalAutoEncoder


class Discriminator(nn.Module):

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 512,
        depth: int = 3,
    ):
        super().__init__()
        self._layers = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for _ in range(depth - 1):
            self._layers.append(nn.Linear(hidden_dim, hidden_dim))
            self._layers.append(nn.LeakyReLU(0.2))

        self.linear_relu_stack = nn.Sequential(*self._layers)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, with_logits: bool = False) -> torch.Tensor:
        x = self.linear_relu_stack(x)
        if with_logits:
            return torch.sigmoid(self.output_proj(x))
        else:
            return self.output_proj(x)

    def loss(self, x: torch.Tensor, labels: torch.Tensor, reduction: str = "sum"):
        preds = self.forward(x)
        return F.binary_cross_entropy_with_logits(preds.squeeze(), labels, reduction=reduction)


class AdversiaralAutoEncoder:

    def __init__(
        self,
        autoencoder: Union[nn.Module, dict],
        discriminator: Union[nn.Module, dict],
        prior: Union[nn.Module, dict],
        device: str,
        latent_dim: int = 2,
        latent_sample_num: int = 128,
        beta: float = 0.5,
    ):
        self.autoencoder: VariationalAutoEncoder = None
        if isinstance(autoencoder, dict):
            self.autoencoder = VariationalAutoEncoder(
                encoder=autoencoder["encoder"],
                decoder=autoencoder["decoder"],
                device=device,
                latent_dim=latent_dim,
                latent_sample_num=latent_sample_num,
                beta=beta,
            )
        else:
            self.autoencoder = autoencoder

        self.prior: BasePrior = None
        if isinstance(prior, dict):
            self.prior = prior_factory.build(prior, name=prior["type"], latent_dim=latent_dim, device=device)
        else:
            self.prior = prior

        self.discriminator: nn.Module = None
        if isinstance(discriminator, dict):
            self.discriminator = Discriminator(
                input_dim=latent_dim,
                hidden_dim=discriminator["hidden_dim"],
                depth=discriminator["depth"],
            )
        else:
            self.discriminator = discriminator

    def to(self, device: str):
        self.autoencoder.to(device)
        self.discriminator.to(device)

    def train(self):
        self.autoencoder.train()
        self.discriminator.train()

    def eval(self):
        self.autoencoder.eval()
        self.discriminator.eval()

    def loss(self, x: torch.Tensor):
        pass

    def sample(self, labels: torch.Tensor):
        z = self.prior.sample(labels=labels)
        return self.decoder(z)
