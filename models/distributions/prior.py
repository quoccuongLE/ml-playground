import math
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

    def sample_n(
        self, batch_size: int, sample_num: Optional[int] = None
    ) -> torch.Tensor:
        if sample_num is None:
            sample_num = self.sample_num
        z = torch.randn((sample_num, batch_size, self.latent_dim))
        return z

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return log_standard_normal(z)


class GaussianMultivariateMixture2D:

    def __init__(self, latent_dim: int, num_classes: int, device: str = "cpu"):
        super().__init__()
        assert (
            latent_dim == 2
        ), "This is a 2D Multivariate Gaussian Mixture, the latent dimension must be equal to 2"
        self.latent_dim = latent_dim
        self._device = device
        self.num_classes = num_classes
        self.gaussians_means: dict[int, np.ndarray] = dict()
        self.gaussians_covs: dict[int, np.ndarray] = dict()
        self.gaussians: dict[int, Distribution] = dict()
        self._mean: torch.Tensor = None
        self._cov: torch.Tensor = None
        self._det: torch.Tensor = None
        self._inv: torch.Tensor = None
        self._init_mixture_of_gaussians(radius=4.0, sigma_1=2.0, sigma_2=0.1)

    def _init_mixture_of_gaussians(
        self, radius: float, sigma_1: float = 1.0, sigma_2: float = 0.2
    ):
        angles = np.linspace(0, 2 * np.pi, self.num_classes, endpoint=False)
        _mean = []
        _cov = []
        _det = []
        _inv = []
        for i, angle in enumerate(angles):
            mean = torch.from_numpy(
                radius * np.array([np.cos(angle), np.sin(angle)])
            ).to(torch.float)
            cov = torch.from_numpy(
                cov_rotation(np.array([[sigma_1, 0], [0, sigma_2]]), angle)
            ).to(torch.float)
            determinant = torch.det(cov)
            inv_cov = torch.inverse(cov)
            self.gaussians_means[i] = mean
            self.gaussians_covs[i] = cov
            self.gaussians[i] = MultivariateNormal(mean, cov)
            _mean.append(self.gaussians_means[i])
            _cov.append(self.gaussians_covs[i])
            _det.append(determinant)
            _inv.append(inv_cov)
        self._mean = torch.stack(_mean).to(device=self._device)
        self._cov = torch.stack(_cov).to(device=self._device)
        self._det = torch.stack(_det).to(device=self._device)
        self._inv = torch.stack(_inv).to(device=self._device)

    def sample(self, labels: torch.Tensor):
        res = []
        for label in labels:
            res.append(self.gaussians[int(label)].sample(torch.Size([1])))
        return torch.stack(res)

    def log_prob(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        num_samples = x.shape[0]
        batch_size = x.shape[1]
        assert x.shape[-2] == label.shape[0]
        mean = self._mean[label].repeat(num_samples, 1)
        inv = self._inv[label].repeat(num_samples, 1, 1)

        # [num_samples*batch_size, latent_dim]
        x_m_mean = x.reshape(-1, 1, self.latent_dim) - mean[:, None, :]

        # log_prob = -0.5 * torch.matmul(
        #     torch.matmul(x_m_mean, inv), torch.transpose(x_m_mean, 1, 2)
        # ).reshape(num_samples, batch_size, self.latent_dim)
        log_prob = -0.5 * x_m_mean * torch.matmul(inv, x_m_mean.transpose(1,2)).transpose(1,2)
        # [num_samples*batch_size, 1, latent_dim] -> [num_samples, batch_size, latent_dim]
        log_prob = log_prob.reshape(num_samples, batch_size, self.latent_dim)
        log_prob += -0.5 * self.latent_dim * torch.log(torch.tensor(2.0 * torch.pi))
        log_prob += -0.5 * torch.log(self._det[label])[None, :, None]

        return log_prob
