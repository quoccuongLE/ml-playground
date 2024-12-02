import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn

from ..vae import Encoder as _Encoder, Decoder as _Decoder
from ..distributions.prior import Prior
from ..distributions.utils import log_normal_diag, log_bernoulli, log_categorical


""" Code adapted from Ref.
https://jmtomczak.github.io/blog/4/4_VAE.html
"""


class Encoder(_Encoder):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 512,
        latent_dim: int = 2,
        depth: int = 3,
        head_depth: int = 3,
    ):
        super().__init__(input_dim, hidden_dim, latent_dim, depth, head_depth)

    @staticmethod
    def reparameterization(mean: torch.Tensor, log_var: torch.Tensor, sample_num: int = -1) -> torch.Tensor:
        """The reparameterization trick for Gaussians.

        Args:
            mean (torch.Tensor): Mean
            log_var (torch.Tensor): log-variance

        Returns:
            torch.Tensor: Sampled latent variable
        """
        # z = mu + std * epsilon, in which epsilon ~ Normal(0,1)
        # First, we need to get std from log-variance.
        # std = torch.exp(0.5 * log_var)
        # # Second, we sample epsilon from Normal(0,1).
        # eps = torch.randn_like(std)
        if sample_num == -1:
            eps = torch.randn_like(log_var)
            # The final output
            return mean + log_var * eps
        else:
            shape = log_var.shape
            eps = torch.randn([sample_num] + list(shape)).to(log_var.device)
            return mean + log_var * eps

    # This function implements the output of the encoder network (i.e., parameters of a Gaussian).
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return super(Encoder, self).forward(x)

    # Sampling procedure.
    def sample(
        self,
        x: Optional[torch.Tensor] = None,
        mean: Optional[torch.Tensor] = None,
        log_var: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Samples a latent variable `z` from the Gaussian distribution
        parameterized by the mean `mean` and log-variance `log_var`.
        If `mean` and `log_var` are not provided, they are calculated by
        calling the `encode` method on the input `x`.

        Args:
            x (torch.Tensor, optional): Input tensor.
            mean (torch.Tensor, optional): Mean of the Gaussian distribution.
            log_var (torch.Tensor, optional): Log-variance of the Gaussian distribution.
        Returns:
            torch.Tensor: A sample from the Gaussian distribution.
        """
        assert (x is not None) or (
            mean is None and log_var is None
        ), "mu and log-var cannot be None, while x is None!"
        # If we don't provide a mean and a log-variance, we must first calcuate it:
        if (mean is None) and (log_var is None):
            mean, log_var = super(Encoder, self).forward(x)
        # Otherwise, we can simply apply the reparameterization trick!
        z = self.reparameterization(mean, log_var)
        return z

    # This function calculates the log-probability that is later used for calculating the ELBO.
    @staticmethod
    def log_prob(
        mean: Optional[torch.Tensor] = None,
        log_var: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        reduction: Optional[str] = None
    ) -> torch.Tensor:
        return log_normal_diag(x=z, mu=mean, log_var=log_var, reduction=reduction)


class Decoder(_Decoder):

    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dim: int = 512,
        output_dim: int = 784,
        depth: int = 3,
        distribution: str = "bernoulli",
        num_vals: int = 256,
    ):
        assert distribution in [
            "categorical",
            "bernoulli",
        ], "Distribution should be either `categorical` or `bernoulli`"
        if distribution == "categorical":
            output_dim *= num_vals
        super().__init__(latent_dim, hidden_dim, output_dim, depth)
        # The distribution used for the decoder (it is categorical by default, as discussed above).
        self.distribution = distribution
        # The number of possible values. This is important for the categorical distribution.
        self.num_vals = num_vals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_relu_stack(x)
        raw_output = self.output_proj(x)

        # In this example, we use only the categorical distribution...
        if self.distribution == "categorical":
            # We save the shapes: batch size
            batch_size = raw_output.shape[0]
            # and the dimensionality of x.
            dim = raw_output.shape[1] // self.num_vals
            # Then we reshape to (Batch size, Dimensionality, Number of Values).
            raw_output = raw_output.view(batch_size, dim, self.num_vals)
            # To get probabilities, we apply softmax.
            mu_d = torch.softmax(raw_output, 2)
            return mu_d
        else:
            # However, we also present the Bernoulli distribution. We are nice, aren't we?
            # In the Bernoulli case, we have x_d \in {0,1}. Therefore, it is enough to output a single probability,
            # because p(x_d=1|z) = \theta and p(x_d=0|z) = 1 - \theta
            mu_d = torch.sigmoid(raw_output)
            return mu_d

    # This function calculates parameters of the likelihood function p(x|z)
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward(z)

    # This function implements sampling from the decoder.
    def sample(self, z: torch.Tensor) -> torch.Tensor:
        outs = self.forward(z)

        if self.distribution == "categorical":
            # We take the output of the decoder
            mu_d = outs[0]
            # and save shapes (we will need that for reshaping).
            b = mu_d.shape[0]
            m = mu_d.shape[1]
            # Here we use reshaping
            mu_d = mu_d.view(mu_d.shape[0], -1, self.num_vals)
            p = mu_d.view(-1, self.num_vals)
            # Eventually, we sample from the categorical (the built-in PyTorch function).
            x_new = torch.multinomial(p, num_samples=1).view(b, m)

        else:
            # In the case of Bernoulli, we don't need any reshaping
            mu_d = outs[0]
            # and we can use the built-in PyTorch sampler!
            x_new = torch.bernoulli(mu_d)

        return x_new

    # This function calculates the conditional log-likelihood function.
    def log_prob(self, x: torch.Tensor, x_hat: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        if self.distribution == "categorical":
            log_p = log_categorical(
                x, x_hat, num_classes=self.num_vals, reduction=reduction, dim=-1
            ).sum(-1)

        else:
            log_p = log_bernoulli(x, x_hat, reduction=reduction)

        return log_p


class VAE(nn.Module):

    def __init__(
        self,
        encoder: Union[nn.Module, dict],
        decoder: Union[nn.Module, dict],
        device: str,
        latent_dim: Optional[int] = None,
        latent_sample_num: int = 128,
        beta: float = 0.5
    ):
        super(VAE, self).__init__()
        self.device = device
        self.latent_sample_num = latent_sample_num
        self.beta = beta
        self.prior = Prior(latent_dim=latent_dim)
        if isinstance(encoder, nn.Module):
            self.encoder = encoder
        elif isinstance(encoder, dict):
            self.encoder = Encoder(
                input_dim=encoder["input_dim"],
                hidden_dim=encoder["hidden_dim"],
                latent_dim=encoder["latent_dim"],
                depth=encoder["depth"],
            )
        else:
            raise TypeError(f"Unsupported type {type(encoder)}!")
        if isinstance(decoder, nn.Module):
            self.decoder = decoder
        elif isinstance(decoder, dict):
            self.decoder = Decoder(
                latent_dim=decoder["latent_dim"],
                hidden_dim=decoder["hidden_dim"],
                output_dim=decoder["output_dim"],
                depth=decoder["depth"],
                distribution=decoder.get("distribution", "bernoulli"),
                num_vals=decoder.get("num_vals", 256),
            )
        else:
            raise TypeError(f"Unsupported type {type(decoder)}!")

    def forward(
        self, x: torch.Tensor, reduction: str = "sum", mode: str = "train", sampling: bool = True
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
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
            # ELBO
            if reduction == "sum":
                if not sampling:
                    # In the case where the chosen prior and chosen posterior are
                    # gaussian distributions (in the original paper of VAE
                    # Kingma & Welling 2013), we can directly obtain KL-Divergence
                    # value (KL) and reconstruction loss (RE) via their
                    # analytical forms (see. Appendix of the paper)
                    KL = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                    RE = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
                else:
                    # In the case where analytical forms are not available, we
                    # apply Monte Carlo method by sampling over a distribution
                    # (see reparameterization code in Encoder)
                    RE = - self.decoder.log_prob(x, x_hat)
                    # Mean(dim=0) here is to calculate the expectation value of
                    # the integral by Monte Carlo method
                    KL = (self.encoder.log_prob(mean=mean, log_var=log_var, z=z) - self.prior.log_prob(z)).mean(dim=0).sum()
                return (1 - self.beta) * RE + self.beta * KL
            else:
                raise NotImplementedError
        else:
            return x_hat, mean, log_var

    def sample(self, batch_size: int = 64):
        z = self.prior.sample(batch_size=batch_size)
        return self.decoder.sample(z)
