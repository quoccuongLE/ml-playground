from typing import Tuple, Union
import torch
import torch.nn as nn
from .vae import VariationalAutoEncoder

import torch.nn.functional as F


class CVAE(VariationalAutoEncoder):
    """Conditional VAE: z and y are independent
    - y controls what number will be generated
    - x controls everything else (styles, etc.)
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
