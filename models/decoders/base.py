import torch
import torch.nn as nn

from models.residual import ResidualStack


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


class CNNDecoder(nn.Module):
    _kernel: int = 4
    _stride: int = 2

    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        res_hidden_channels: int,
        nb_res_layers: int,
        out_channels: int = 3,
    ):
        super().__init__()
        self.inv_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=middle_channels,
                kernel_size=self._kernel - 1,
                stride=self._stride,
                padding=1,
            ),
            ResidualStack(
                in_channels=middle_channels,
                out_channels=middle_channels,
                hidden_channels=res_hidden_channels,
                res_layers=nb_res_layers,
            ),
            nn.ConvTranspose2d(
                in_channels=middle_channels // 2,
                out_channels=out_channels,
                kernel_size=self._kernel,
                stride=self._stride,
                padding=1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inv_conv_stack(x)
