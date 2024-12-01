from typing import Optional
import torch
import torch.nn as nn

from .utils import log_standard_normal


class Prior(nn.Module):
    def __init__(self, latent_dim: int, sample_num: int = 256):
        super(Prior, self).__init__()
        self.latent_dim = latent_dim
        self.sample_num = sample_num

    def sample(self, batch_size: int):
        z = torch.randn((batch_size, self.latent_dim))
        return z

    def sample_n(self, batch_size: int, sample_num: Optional[int] = None) -> torch.Tensor:
        if sample_num is None:
            sample_num = self.sample_num
        z = torch.randn((sample_num, batch_size, self.latent_dim))
        return z

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return log_standard_normal(z)
