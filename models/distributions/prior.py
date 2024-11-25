import torch
import torch.nn as nn

from .utils import log_standard_normal


class Prior(nn.Module):
    def __init__(self, latent_dim: int):
        super(Prior, self).__init__()
        self.latent_dim = latent_dim

    def sample(self, batch_size):
        z = torch.randn((batch_size, self.latent_dim))
        return z

    def log_prob(self, z):
        return log_standard_normal(z)
