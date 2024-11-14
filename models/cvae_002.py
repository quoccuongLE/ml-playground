from typing import Tuple, Union
import torch
import torch.nn as nn
from .vae import VariationalAutoEncoder

import torch.nn.functional as F


class CVAECase2(VariationalAutoEncoder):

    def __init__(
        self,
        encoder: Union[nn.Module, dict],
        decoder: Union[nn.Module, dict],
        num_classes: int,
        device: str,
    ):
        # if isinstance(encoder, dict):
        #     encoder["input_dim"] += num_classes
        # if isinstance(decoder, dict):
        #     decoder["latent_dim"] += num_classes
        super().__init__(encoder, decoder, device=device)
        self.num_classes = num_classes
        self.label_projector = nn.Sequential(
            nn.Linear(num_classes, decoder["latent_dim"]),
            nn.LeakyReLU(0.2),
        )

    def condition_on_label(self, z, y):
        projected_label = self.label_projector(y.float())
        return z + projected_label

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # enc_label = F.one_hot(y, num_classes=self.num_classes)
        # mean, log_var = self.encoder(torch.cat((x, enc_label), dim=1))
        mean, log_var = self.encoder(x)
        z = self.reparameterization(
            mean, torch.exp(0.5 * log_var)
        )  # takes exponential function (log var -> var)
        # x_hat = self.decoder(torch.cat((z, dec_label), dim=1))
        dec_label = F.one_hot(y, num_classes=self.num_classes)
        _z = self.condition_on_label(z, dec_label)
        x_hat = self.decoder(_z)

        return x_hat, mean, log_var
