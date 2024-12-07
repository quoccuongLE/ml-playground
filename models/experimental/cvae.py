from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from .vae import VAE
from ..distributions.prior import GaussianMultivariateMixture2D

import torch.nn.functional as F


class CVAE(VAE):
    """Conditional VAE 2: z and y are dependent
    - y controls distribution p(z|x, y)
    Graphical model: (y) -> (z) and (z) -> (x)

    Structure:
        * Encoding: z ~ p(z|x, y) (inference process)
        * Decoding: z ~ p_\phi(z|y) x ~ p(x|z) (generative process)
    """

    def __init__(
        self,
        encoder: Union[nn.Module, dict],
        decoder: Union[nn.Module, dict],
        num_classes: int,
        device: str,
        latent_dim: int,
        latent_sample_num: int = 128,
        beta: float = 0.5,
    ):
        super().__init__(
            encoder,
            decoder,
            device=device,
            latent_dim=latent_dim,
            latent_sample_num=latent_sample_num,
            beta=beta
        )
        self.prior = GaussianMultivariateMixture2D(latent_dim=latent_dim, num_classes=num_classes, device=device)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, reduction: str = "sum", mode: str = "train", sampling: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z ~ p(z|x, y)
        mean, log_var = self.encoder(x)
        if sampling:
            # Unlike non-sampling case, in reparameterization step samples
            # latent_sample_num times, leading a output shape of [latent_sample_num, batch_size, latent_dim]
            z = self.encoder.reparameterization(
                mean=mean, log_var=log_var, sample_num=self.latent_sample_num
            )
            # Only take the first samples among latent_sample_num samples
            x_hat = self.decoder(z[0])
        else:
            z = self.encoder.reparameterization(mean=mean, log_var=log_var)
            x_hat = self.decoder(z)

        if mode == "train":
            if not sampling:
                raise NotImplementedError
            else:
                RE = - self.decoder.log_prob(x, x_hat)
                # log_posterior_prob = self.encoder.log_prob(mean=mean, log_var=log_var, z=z)
                # log_prior_prob = self.prior.log_prob(x=z, label=y)
                # KL = (log_posterior_prob - log_prior_prob)
                # log_posterior_prob = self.encoder.log_prob(mean=mean, log_var=log_var, z=z)
                # log_prior_prob = self.prior.log_prob(x=z, label=y)
                KL = (
                    (
                        self.encoder.log_prob(mean=mean, log_var=log_var, z=z)
                        - self.prior.log_prob(x=z, label=y)
                    )
                    .mean(dim=0)
                    .sum()
                )
                return (1 - self.beta) * RE + self.beta * KL
        else:
            return x_hat, mean, log_var
