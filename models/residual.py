import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, hidden_channels: int
    ):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self.res_block(x)


class ResidualStack(nn.Module):

    def __init__(
        self, in_channels: int, out_channels: int, hidden_channels: int, res_layers: int
    ):
        super().__init__()
        _layers = [
            ResidualLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
            )
        ] * res_layers
        self.stack = nn.Sequential(*_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        return F.relu(x)


if __name__ == "__main__":
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()
    res = ResidualLayer(40, 40, 20)
    res_out = res(x)
    print("Res Layer out shape:", res_out.shape)
