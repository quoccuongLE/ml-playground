from typing import Optional, Union
import torch
import torch.nn as nn

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class AdversiaralAutoEncoder(VariationalAutoEncoder):

    def __init__(
        self,
        encoder: Union[nn.Module, dict],
        decoder: Union[nn.Module, dict],
        prior: Union[nn.Module, dict],
        device: str,
        latent_dim: Optional[int] = None,
        latent_sample_num: int = 128,
        beta: float = 0.5,
    ):
        super().__init__(encoder, decoder, device, latent_dim, latent_sample_num, beta)
        self.prior: BasePrior = None
        if isinstance(prior, dict):
            self.prior = prior_factory.build(prior)
        else:
            self.prior = prior

        self.discriminator = None
