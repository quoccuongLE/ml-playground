from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.distribution import Distribution
from torch.distributions.multivariate_normal import MultivariateNormal

from .utils import log_standard_normal


def cov_rotation(cov: np.ndarray, angle_rad: float) -> np.ndarray:
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    return rotation_matrix @ cov @ rotation_matrix.T


class Prior:
    def __init__(self, latent_dim: int, sample_num: int = 256):
        super().__init__()
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


class GaussianMultivariateMixture:

    def __init__(self, latent_dim: int, num_classes: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.gaussians_means: dict[int, np.ndarray] = dict()
        self.gaussians_covs: dict[int, np.ndarray] = dict()
        self.gaussians: dict[int, np.ndarray] = dict()
        self._init_mixture_of_gaussians(radius=4.0, sigma_1=2.0, sigma_2=0.1)

    def _init_mixture_of_gaussians(
        self, radius: float, sigma_1: float = 1.0, sigma_2: float = 0.2
    ):
        angles = np.linspace(0, 2 * np.pi, self.num_classes, endpoint=False)
        for i, angle in enumerate(angles):
            mean = radius * np.array([np.cos(angle), np.sin(angle)])
            cov = np.array([[sigma_1, 0], [0, sigma_2]])
            self.gaussians_means[i] = mean
            self.gaussians_covs[i] = cov_rotation(cov, angle)
            self.gaussians[i] = MultivariateNormal(mean, cov)

    def sample(self, batch_size: int, label: int):
        pass

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return super().log_prob(value)
