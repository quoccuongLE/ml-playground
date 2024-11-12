from typing import Tuple, Union
import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 512,
        latent_dim: int = 2,
        depth: int = 3,
    ):
        super(Encoder, self).__init__()
        self._layers = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for _ in range(depth - 1):
            self._layers.append(nn.Linear(hidden_dim, hidden_dim))
            self._layers.append(nn.LeakyReLU(0.2))

        self.linear_relu_stack = nn.Sequential(*self._layers)
        self.mean_proj = nn.Linear(hidden_dim, latent_dim)
        self.var_proj = nn.Linear(hidden_dim, latent_dim)
        self.training = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.linear_relu_stack(x)
        mean = self.mean_proj(x)
        log_var = self.var_proj(x)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"

        return mean, log_var


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dim: int = 512,
        output_dim: int = 784,
        depth: int = 3,
    ):
        super(Decoder, self).__init__()
        self._layers = [nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for _ in range(depth - 1):
            self._layers.append(nn.Linear(hidden_dim, hidden_dim))
            self._layers.append(nn.LeakyReLU(0.2))
        self.linear_relu_stack = nn.Sequential(*self._layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_relu_stack(x)
        x_hat = torch.sigmoid(self.output_proj(x))
        return x_hat


class VariationalAutoEncoder(nn.Module):

    def __init__(
        self,
        encoder: Union[nn.Module, dict],
        decoder: Union[nn.Module, dict],
        device: str,
    ):
        super(VariationalAutoEncoder, self).__init__()
        self.device = device
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
            )
        else:
            raise TypeError(f"Unsupported type {type(decoder)}!")

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_var = self.encoder(x)
        z = self.reparameterization(
            mean, torch.exp(0.5 * log_var)
        )  # takes exponential function (log var -> var)
        x_hat = self.decoder(z)

        return x_hat, mean, log_var
