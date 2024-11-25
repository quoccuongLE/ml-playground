from typing import Tuple, Union
import torch
import torch.nn as nn
from .vae import VariationalAutoEncoder

import torch.nn.functional as F


class CVAE(VariationalAutoEncoder):
    """Conditional VAE 1: z and y are independent
    - y controls what number will be generated
    - z controls everything else (styles, etc.)
    Graphical model: (z) -> (x) and (y) -> (x)

    Structure:
        * Encoding: z ~ p(z|y) (inference process)
        * Decoding: x ~ p(x|x, y) (generative process)
    """
    def __init__(
        self,
        encoder: Union[nn.Module, dict],
        decoder: Union[nn.Module, dict],
        num_classes: int,
        device: str,
    ):
        if isinstance(decoder, dict):
            decoder["latent_dim"] += num_classes
        super().__init__(encoder, decoder, device=device)
        self.num_classes = num_classes

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z ~ p(z|x)
        mean, log_var = self.encoder(x)
        z = self.reparameterization(
            mean, torch.exp(0.5 * log_var)
        )  # takes exponential function (log var -> var)
        dec_label = F.one_hot(y, num_classes=self.num_classes)
        # x_hat ~ p(x|z,y)
        x_hat = self.decoder(torch.cat((z, dec_label), dim=1))

        return x_hat, mean, log_var


class CVAE2(VariationalAutoEncoder):
    """Conditional VAE 2: z and y are dependent
    - y controls distribution p(z|x, y)
    Graphical model: (y) -> (z) and (z) -> (x)

    Structure:
        * Encoding: z ~ p(z|x, y) (inference process)
        * Decoding: z ~ p_/phi(z|y) x ~ p(x|z) (generative process)
    """

    def __init__(
        self,
        encoder: Union[nn.Module, dict],
        decoder: Union[nn.Module, dict],
        num_classes: int,
        device: str,
    ):
        super().__init__(encoder, decoder, device=device)
        self.num_classes = num_classes
        self.gaussians_means: dict[int, np.ndarray] = dict()
        self.gaussians_covs: dict[int, np.ndarray] = dict()
        self.init_mixture_of_gaussians(radius=4.0)
        self.means = torch.tensor(np.stack(list(self.gaussians_means.values()))).to(
            device
        )

    def init_mixture_of_gaussians(self, radius: float, sigma: float = 1.0):
        angles = np.linspace(0, 2 * np.pi, self.num_classes, endpoint=False)
        for i, angle in enumerate(angles):
            self.gaussians_means[i] = radius * np.array([np.cos(angle), np.sin(angle)])
            self.gaussians_covs[i] = np.array([[sigma, 0], [0, sigma]])

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z ~ p(z|x, y)
        mean, log_var = self.encoder(x)
        z = self.reparameterization(
            mean, torch.exp(0.5 * log_var)
        )  # takes exponential function (log var -> var)
        # x_hat ~ p(x|z)
        x_hat = self.decoder(z)

        return x_hat, mean, log_var
