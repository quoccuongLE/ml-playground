from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):

    _embedding_dim: int
    _nb_embeddings: int
    beta: float

    def __init__(self, embedding_dim: int, num_embeddings: int, beta: float):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self.beta = beta
        self._embedding = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )

    def _encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        # Distance matrix
        dist_mat = (
            (x**2).sum(dim=1, keepdim=True)
            + (self._embedding.weight**2).sum(dim=1)
            - 2 * torch.matmul(x, self._embedding.weight.t())
        )
        # dist_mat = torch.cdist(z_flattened, self._embedding.weight)

        # Get top-k
        min_encoding_indices = torch.argmin(dist_mat, dim=1).unsqueeze(1)
        encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to("cuda:0")
        encodings.scatter_(1, min_encoding_indices, 1)
        # Get quantized latent vectors
        quantized = torch.matmul(encodings, self._embedding.weight).view(x.shape)
        return quantized, encodings

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        z = z.permute(0, 2, 3, 1).contiguous()  # [BS, C, H, W] -> [BS, H, W, C]
        z_flattened = z.view(-1, self._embedding_dim)
        quantized, encodings = self._encode(z_flattened)

        # Compute loss
        # loss =  ((z.detach() - quantized)**2).mean() + self.beta * ((z - quantized.detach())**2).mean()
        loss = F.mse(z.detach(), quantized) + self.beta * F.mse(z, quantized.detach())

        quantized = z + (quantized - z).detach()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return loss, quantized, perplexity, encodings


class VectorQuantizerEMA(VectorQuantizer):

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        beta: float,
        decay: float,
        epsilon: float = 1e-5,
    ):
        super().__init__(embedding_dim, num_embeddings, beta)
        self._embedding.weight.data.normal_()
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_weight = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_weight.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()  # [BS, C, H, W] -> [BS, H, W, C]
        z_flattened = z.view(-1, self._embedding_dim)
        quantized, encodings = self._encode(z_flattened)

        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), z_flattened)
            self._ema_weight = nn.Parameter(self._ema_weight * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_weight / self._ema_cluster_size.unsqueeze(1))

        e_latent_loss = F.mse_loss(quantized.detach(), z)
        loss = self.beta * e_latent_loss

        # Straight Through Estimator
        quantized = z + (quantized - z).detach()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return loss, quantized, perplexity, encodings
