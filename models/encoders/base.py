from typing import Tuple
import torch
import torch.nn as nn

from models.residual import ResidualStack


class Encoder(nn.Module):

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 512,
        latent_dim: int = 2,
        depth: int = 3,
        head_depth: int = 3,
    ):
        super(Encoder, self).__init__()
        self._layers = [nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for _ in range(depth - 1):
            self._layers.append(nn.Linear(hidden_dim, hidden_dim))
            self._layers.append(nn.LeakyReLU(0.2))

        self.linear_relu_stack = nn.Sequential(*self._layers)
        # self.mean_proj = nn.Linear(hidden_dim, latent_dim)
        # self.var_proj = nn.Linear(hidden_dim, latent_dim)

        self._mean_layers, self._var_layers = [], []
        for _ in range(head_depth - 1):
            self._mean_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self._mean_layers.append(nn.LeakyReLU(0.2))
            self._var_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self._var_layers.append(nn.LeakyReLU(0.2))
        self._mean_layers.append(nn.Linear(hidden_dim, latent_dim))
        self._var_layers.append(nn.Linear(hidden_dim, latent_dim))
        self.mean_proj = nn.Sequential(*self._mean_layers)
        self.var_proj = nn.Sequential(*self._var_layers)
        self.training = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.linear_relu_stack(x)
        mean = self.mean_proj(x)
        log_var = self.var_proj(x)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"

        return mean, log_var


class CNNEncoder(nn.Module):

    _kernel: int = 4
    _stride: int = 2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        nb_res_layers: int,
    ):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                kernel_size=self._kernel,
                stride=self._stride,
                padding=1,
            ),
            nn.ReLU(),  # inplace=False
            nn.Conv2d(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=self._kernel,
                stride=self._stride,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=self._kernel - 1,
                stride=self._stride - 1,
                padding=1,
            ),
            ResidualStack(
                in_channels=out_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                res_layers=nb_res_layers,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_stack(x)
