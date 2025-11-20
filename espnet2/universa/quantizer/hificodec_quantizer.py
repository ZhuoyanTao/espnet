import torch
import torch.nn as nn

class QuantizerModule(nn.Module):
    def __init__(self, n_e: int, e_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, x: torch.Tensor):
        # x: (N, e_dim)
        # distances: (N, n_e)
        d = torch.sum(x**2, dim=1, keepdim=True) \
            + torch.sum(self.embedding.weight**2, dim=1) \
            - 2 * torch.matmul(x, self.embedding.weight.T)
        min_indices = torch.argmin(d, dim=1)              # (N,)
        z_q = self.embedding(min_indices)                 # (N, e_dim)
        return z_q, min_indices


class HiFiCodecQuantizer(nn.Module):
    def __init__(self,
                 dim: int = 512,
                 n_groups: int = 8,
                 n_codes: int = 1024,
                 codebook_loss_lambda: float = 1.0,
                 commitment_loss_lambda: float = 0.25,
                 residual_layers: int = 2):
        super().__init__()
        assert dim % n_groups == 0, f"dim {dim} must be divisible by n_groups {n_groups}"
        self.dim = dim
        self.n_groups = n_groups
        self.n_codes = n_codes
        self.sub_dim = dim // n_groups
        self.codebook_loss_lambda = codebook_loss_lambda
        self.commitment_loss_lambda = commitment_loss_lambda
        self.residual_layers = residual_layers

        # two residual stages (as in HiFi-Codec)
        self.quantizer_modules = nn.ModuleList(
            [QuantizerModule(n_codes, self.sub_dim) for _ in range(n_groups)]
        )
        self.quantizer_modules2 = nn.ModuleList(
            [QuantizerModule(n_codes, self.sub_dim) for _ in range(n_groups)]
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, T) where C == self.dim
        returns:
            quantized_out: (B, C, T)
            total_loss: scalar tensor
            all_indices: list of length n_groups * residual_layers, each (B*T,) ints
        """
        B, C, T = x.shape
        assert C == self.dim, f"Expected channel dim {self.dim}, got {C}"

        quantized_out = torch.zeros_like(x)
        residual = x
        all_losses = []
        all_indices = []

        for layer_idx in range(self.residual_layers):
            z_q, loss, indices = self._quantize_step(residual, layer_idx)
            residual = residual - z_q                      # all in (B, C, T)
            quantized_out = quantized_out + z_q
            all_indices.extend(indices)
            all_losses.append(loss)

        total_loss = torch.mean(torch.stack(all_losses))
        return quantized_out, total_loss, all_indices

    def _quantize_step(self, x: torch.Tensor, layer_idx: int):
        """
        x: (B, C, T)  →  work in (B, T, C) for grouping
        returns z_q in (B, C, T)
        """
        B, C, T = x.shape
        x_btC = x.transpose(1, 2)                 # (B, T, C)
        flat = x_btC.reshape(-1, C)               # (B*T, C)
        parts = torch.split(flat, self.sub_dim, dim=-1)  # list of (B*T, sub_dim)

        modules = self.quantizer_modules if layer_idx == 0 else self.quantizer_modules2

        zs = []
        idxs = []
        for sub, mod in zip(parts, modules):
            z_q_sub, ind = mod(sub)               # (B*T, sub_dim), (B*T,)
            zs.append(z_q_sub)
            idxs.append(ind)

        z_q_btC = torch.cat(zs, dim=-1)           # (B*T, C)
        z_q_btC = z_q_btC.reshape(B, T, C)        # (B, T, C)

        # Losses in (B, T, C) space
        codebook_loss = torch.mean((z_q_btC - x_btC.detach()) ** 2)
        commit_loss   = torch.mean((z_q_btC.detach() - x_btC) ** 2)
        loss = self.codebook_loss_lambda * codebook_loss + self.commitment_loss_lambda * commit_loss

        # STE: preserve gradients w.r.t. x_btC
        z_q_btC = x_btC + (z_q_btC - x_btC).detach()

        # back to (B, C, T)
        z_q = z_q_btC.transpose(1, 2)
        return z_q, loss, idxs
