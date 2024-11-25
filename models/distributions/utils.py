from typing import Optional
import numpy as np

import torch.nn.functional as F

import torch
import torch.utils.data

PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.0e-5


def log_categorical(
    x: torch.Tensor,
    p: torch.Tensor,
    num_classes: int = 256,
    reduction: Optional[str] = None,
    dim: Optional[int] = None,
):
    x_one_hot = F.one_hot(x.long(), num_classes=num_classes)
    log_p = x_one_hot * torch.log(torch.clamp(p, EPS, 1.0 - EPS))
    if reduction == "avg":
        return torch.mean(log_p, dim)
    elif reduction == "sum":
        return torch.sum(log_p, dim)
    else:
        return log_p


# TODO: Review this function
# def log_normal_diag(
#     x: torch.Tensor,
#     mean: torch.Tensor,
#     log_var: torch.Tensor,
#     average: bool = False,
#     dim: int | None = None,
# ) -> torch.Tensor:
#     log_normal = -0.5 * (
#         log_var + torch.pow(x - mean, 2) * torch.pow(torch.exp(log_var), -1)
#     )
#     if average:
#         return torch.mean(log_normal, dim)
#     else:
#         return torch.sum(log_normal, dim)


def log_normal_diag(
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    reduction: Optional[str] = None,
    dim: Optional[int] = None,
) -> torch.Tensor:
    D = x.shape[1]
    log_p = -0.5 * D * torch.log(2. * PI) - 0.5 * log_var - 0.5 * (x - mu).pow(2) / log_var.exp()
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p


# def log_normal_standard(
#     x: torch.Tensor, average: bool = False, dim: int | None = None
# ) -> torch.Tensor:
#     log_normal = -0.5 * torch.pow(x, 2)
#     if average:
#         return torch.mean(log_normal, dim)
#     else:
#         return torch.sum(log_normal, dim)


def log_standard_normal(
    x: torch.Tensor,
    reduction: Optional[str] = None,
    dim: Optional[int] = None,
) -> torch.Tensor:
    D = x.shape[1]
    log_p = -0.5 * D * torch.log(2.0 * PI) - 0.5 * x.pow(2)
    if reduction == "avg":
        return torch.mean(log_p, dim)
    elif reduction == "sum":
        return torch.sum(log_p, dim)
    else:
        return log_p


def log_bernoulli(
    x: torch.Tensor, p: torch.Tensor, reduction: Optional[str] = None, dim: Optional[int] = None
) -> torch.Tensor:
    pp = torch.clamp(p, EPS, 1.0 - EPS)
    log_p = x * torch.log(pp) + (1.0 - x) * torch.log(1.0 - pp)
    if reduction == "avg":
        return torch.mean(log_p, dim)
    elif reduction == "sum":
        return torch.sum(log_p, dim)
    else:
        return log_p
