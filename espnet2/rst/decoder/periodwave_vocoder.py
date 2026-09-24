"""PeriodWave: a published flow-matching vocoder, conditioned on SSL features.

Derived from PeriodWave (Lee, Kim, Lee, ICLR 2025),
https://github.com/sh-lee-prml/PeriodWave (MIT licence, Copyright (c) 2024
Sang-Hoon Lee). The multi-period vector-field
estimator, its ConvNeXt-V2 conditioner and the flow-matching wrapper are taken
from that repository:

  model/periodwave_encodec.py       -> ``VectorFieldEstimator``, the solvers and
                                       the flow-matching loss
  model/periodwave_encodec_utils.py -> ``GeneratorP``, ``MultiPeriodGenerator``,
                                       ``DownConv2D``, ``MelCondConv2D``,
                                       ``UpConv2D``, ``Resblock``, ``FinalBlock``
                                       and the conditioner
  model/convnext.py                 -> ``ConvNeXtV2Block``, ``GRN``, ``DropPath``
  model/diffusion_module.py         -> ``SinusoidalPosEmb``

The EnCodec variant is vendored rather than the mel one because its
conditioning is already a latent sequence rather than a mel spectrogram, and it
drops PriorGrad's energy prior (``target_std``), which needs a per-frame energy
statistic we cannot compute from predicted SSL features at inference time. Its
loss is therefore a plain MSE on the velocity, exactly as in
``model/periodwave_encodec.py``.

Three adaptations, all of them resolution bookkeeping:

1. The U-Net folds the waveform to ``T / (64 * period)`` before the conditioning
   is added, so conditioning must arrive at ``T / 64``. Upstream reaches it with
   one transposed convolution: x4 from mel frames at hop 256 (mel variant), x5
   from EnCodec latents at hop 320 (this variant). Our features are w2v-BERT 2.0
   at 50 Hz against a 48 kHz waveform, a hop of 960, so the conditioner has to
   upsample by 960 / 64 = 15. That is done in two stages (x5 then x3) with the
   upstream ConvNeXt blocks split between them, rather than one x15 transposed
   convolution, to avoid the checkerboard artefacts a single wide stage invites.
2. ``embed`` takes ``input_dim`` SSL channels instead of 128 EnCodec channels.
3. ``period_token`` is a registered buffer, not a ``.cuda()`` tensor, so the
   module works on CPU and moves with the model under DistributedDataParallel.

The public interface is the one ``ESPnetRestorationFlowVocoderModel`` expects
from a flow vocoder (``condition``, ``velocity``, ``generate``, ``upsample_factor``,
``data_scale``) plus ``flow_loss``, which the model calls in place of its own
generic objective so PeriodWave keeps its own noise scale.

MIT License

Copyright (c) 2024 Sang-Hoon Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Conv2d, ConvTranspose1d, ConvTranspose2d
from torch.nn.utils import remove_weight_norm, weight_norm


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Stochastic depth, per sample, on the residual branch."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class GRN(nn.Module):
    """Global response normalisation (ConvNeXt V2)."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt-V2 block over a 1D sequence (depthwise conv, LN, GRN, MLP)."""

    def __init__(self, dim: int, intermediate_dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)
        return residual + self.drop_path(x)


class SinusoidalPosEmb(nn.Module):
    """Timestep embedding, as in the upstream diffusion_module."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = 1000.0 * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Resblock1D(nn.Module):
    def __init__(self, output_dim: int, kernel_size: int, dilation: int):
        super().__init__()
        self.block1 = weight_norm(
            Conv1d(
                output_dim,
                output_dim,
                kernel_size,
                1,
                dilation=dilation,
                padding=get_padding(kernel_size, dilation),
            )
        )
        self.block2 = weight_norm(
            Conv1d(
                output_dim,
                output_dim,
                kernel_size,
                1,
                dilation=1,
                padding=get_padding(kernel_size, 1),
            )
        )

    def forward(self, x):
        h = self.block2(F.silu(self.block1(F.silu(x))))
        return x + h

    def remove_weight_norm(self):
        remove_weight_norm(self.block1)
        remove_weight_norm(self.block2)


class Resblock(nn.Module):
    """Resblock1D on the folded (time, period) plane: kernels act on time."""

    def __init__(self, output_dim: int, kernel_size: int, dilation: int):
        super().__init__()
        self.block1 = weight_norm(
            Conv2d(
                output_dim,
                output_dim,
                (kernel_size, 1),
                (1, 1),
                dilation=(dilation, 1),
                padding=(get_padding(kernel_size, dilation), 0),
            )
        )
        self.block2 = weight_norm(
            Conv2d(
                output_dim,
                output_dim,
                (kernel_size, 1),
                (1, 1),
                dilation=(1, 1),
                padding=(get_padding(kernel_size, 1), 0),
            )
        )

    def forward(self, x):
        h = self.block2(F.silu(self.block1(F.silu(x))))
        return x + h

    def remove_weight_norm(self):
        remove_weight_norm(self.block1)
        remove_weight_norm(self.block2)


class FinalBlock(nn.Module):
    """Sum over periods -> waveform-rate velocity."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.ResBlocks = nn.Sequential(
            Resblock1D(output_dim, 3, 1),
            Resblock1D(output_dim, 3, 2),
            Resblock1D(output_dim, 3, 4),
        )
        self.final_layer = weight_norm(Conv1d(output_dim, 1, 7, 1, 3, bias=False))

    def forward(self, x):
        return self.final_layer(F.silu(self.ResBlocks(x)))

    def remove_weight_norm(self):
        for layer in self.ResBlocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.final_layer)


class DownConv2D(nn.Module):
    """Encoder stage: stride on the time axis, timestep embedding added."""

    def __init__(
        self, input_dim, output_dim, kernel_size, stride, hidden_dim, act=False
    ):
        super().__init__()
        self.act = nn.SiLU() if act else nn.Identity()
        self.down = weight_norm(
            Conv2d(
                input_dim,
                output_dim,
                (kernel_size, 1),
                (stride, 1),
                padding=(get_padding(kernel_size, 1), 0),
                bias=False,
            )
        )
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, output_dim))
        self.ResBlocks = nn.Sequential(
            Resblock(output_dim, 3, 1), Resblock(output_dim, 3, 2)
        )

    def forward(self, x, time_emb):
        x = self.down(self.act(x))
        x = x + self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        return self.ResBlocks(x)

    def remove_weight_norm(self):
        for layer in self.ResBlocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.down)


class MelCondConv2D(nn.Module):
    """Bottleneck stage: the conditioning enters here, at T / (64 * period)."""

    def __init__(self, input_dim, output_dim, kernel_size, stride):
        super().__init__()
        self.down = weight_norm(
            Conv2d(
                input_dim,
                output_dim,
                (kernel_size, 1),
                (stride, 1),
                padding=(get_padding(kernel_size, 1), 0),
                bias=False,
            )
        )
        self.ResBlocks = nn.Sequential(
            Resblock(output_dim, 3, 1),
            Resblock(output_dim, 3, 2),
            Resblock(output_dim, 3, 4),
        )

    def forward(self, x, cond):
        x = self.down(F.silu(x))
        # The conditioner is built to be at least as long as the folded signal;
        # replicate-pad rather than fail if a segment length makes it shorter.
        if cond.size(2) < x.size(2):
            cond = F.pad(cond, (0, 0, 0, x.size(2) - cond.size(2)), mode="replicate")
        x = x + cond[:, :, : x.size(2), :]
        return self.ResBlocks(x)

    def remove_weight_norm(self):
        remove_weight_norm(self.down)
        for layer in self.ResBlocks:
            layer.remove_weight_norm()


class UpConv2D(nn.Module):
    """Decoder stage: transposed convolution on time, skip and timestep added."""

    def __init__(self, input_dim, output_dim, kernel_size, stride, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, output_dim))
        self.ResBlocks = nn.Sequential(
            Resblock(output_dim, 3, 1), Resblock(output_dim, 3, 2)
        )
        self.up = weight_norm(
            ConvTranspose2d(
                input_dim,
                output_dim,
                (stride * 2, 1),
                (stride, 1),
                padding=(stride // 2, 0),
                bias=False,
            )
        )

    def forward(self, x, skip, time_emb):
        x = self.up(F.silu(x))
        x = x + skip + self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        return self.ResBlocks(x)

    def remove_weight_norm(self):
        for layer in self.ResBlocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.up)


class GeneratorP(nn.Module):
    """One U-Net, run once per period on the waveform folded to (T/p, p)."""

    def __init__(self, period, kernel_size=9, stride=4, final_dim=32, hidden_dim=512):
        super().__init__()
        self.period = period
        self.final_dim = final_dim
        self.downs = nn.ModuleList(
            [
                DownConv2D(1, hidden_dim // 16, 1, 1, hidden_dim, act=False),
                DownConv2D(
                    hidden_dim // 16,
                    hidden_dim // 8,
                    kernel_size,
                    stride,
                    hidden_dim,
                    act=True,
                ),
                DownConv2D(
                    hidden_dim // 8,
                    hidden_dim // 4,
                    kernel_size,
                    stride,
                    hidden_dim,
                    act=True,
                ),
            ]
        )
        self.mids = nn.ModuleList(
            [MelCondConv2D(hidden_dim // 4, hidden_dim, kernel_size, stride)]
        )
        self.ups = nn.ModuleList(
            [
                UpConv2D(hidden_dim, hidden_dim // 4, kernel_size, stride, hidden_dim),
                UpConv2D(
                    hidden_dim // 4, hidden_dim // 8, kernel_size, stride, hidden_dim
                ),
                UpConv2D(hidden_dim // 8, final_dim, kernel_size, stride, hidden_dim),
            ]
        )

    def forward(self, x, time_emb, cond, i):
        b, c, t = x.shape
        ori_t = t
        period = self.period[i]
        if t % (64 * period) != 0:
            n_pad = period * 64 - (t % (period * 64))
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad

        x = x.view(b, c, t // period, period)

        skips = []
        for layer in self.downs:
            x = layer(x, time_emb)
            skips.append(x)
        for layer in self.mids:
            x = layer(x, cond)
        for layer in self.ups:
            x = layer(x, skips.pop(), time_emb)

        x = x.reshape(b, self.final_dim, -1)
        return x[:, :, :ori_t]

    def remove_weight_norm(self):
        for group in (self.downs, self.mids, self.ups):
            for layer in group:
                layer.remove_weight_norm()


class MultiPeriodGenerator(nn.Module):
    """The same U-Net weights applied at every period in ``periods``."""

    def __init__(self, periods=(1, 2, 3, 5, 7), final_dim=32, hidden_dim=512):
        super().__init__()
        self.periods_len = len(periods)
        self.mpg = GeneratorP(list(periods), final_dim=final_dim, hidden_dim=hidden_dim)

    def forward(self, x, time_emb, cond) -> List[torch.Tensor]:
        return [
            self.mpg(x, time_emb[:, i, :], cond[i], i) for i in range(self.periods_len)
        ]

    def remove_weight_norm(self):
        self.mpg.remove_weight_norm()


class SSLConditioner(nn.Module):
    """SSL features at 50 Hz -> one conditioning tensor per period at T/(64 p).

    Upstream's conditioner (``MelSpectrogramUpsampler``) upsamples once, by the
    factor that takes its frames to T / 64. Ours needs 15, done as x5 then x3
    with the four ConvNeXt blocks split between the stages.
    """

    def __init__(
        self,
        input_dim: int,
        periods: Sequence[int] = (1, 2, 3, 5, 7),
        hidden_dim: int = 512,
        cond_rates: Sequence[int] = (5, 3),
    ):
        super().__init__()
        self.embed = Conv1d(input_dim, hidden_dim, 7, stride=1, padding=3)

        self.block_num = 8
        self.convnext_block = nn.ModuleList(
            [
                ConvNeXtV2Block(hidden_dim, int(hidden_dim * 3), drop_path=0.1)
                for _ in range(self.block_num)
            ]
        )

        # One transposed convolution per rate; channels drop to hidden//2 at the
        # first stage, as upstream does, and stay there.
        self.ups = nn.ModuleList()
        self.cond = nn.ModuleList()
        dim = hidden_dim
        for rate in cond_rates:
            out_dim = hidden_dim // 2
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(dim, out_dim, rate * 2, rate, padding=rate // 2)
                )
            )
            self.cond.append(
                nn.ModuleList(
                    [
                        ConvNeXtV2Block(out_dim, int(hidden_dim * 2), drop_path=0.1)
                        for _ in range(2)
                    ]
                )
            )
            dim = out_dim

        self.periods = list(periods)
        self.len_periods = len(self.periods)
        self.cond_block = nn.ModuleList(
            [
                weight_norm(
                    Conv2d(
                        dim,
                        hidden_dim,
                        (p * 2, 1),
                        (p, 1),
                        padding=(get_padding(p * 2, 1), 0),
                    )
                )
                for p in self.periods
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """(B, D, T_frames) -> list of (B, hidden, T/(64 p), 1)."""
        x = self.embed(x)
        residual = x
        for block in self.convnext_block:
            x = block(x)
        x = x + residual

        for up, blocks in zip(self.ups, self.cond):
            x = up(F.silu(x))
            for block in blocks:
                x = block(x)

        x = F.silu(x).unsqueeze(-1)

        conds = []
        for i in range(self.len_periods):
            p = self.periods[i]
            n_pad = p - (x.shape[-2] % p)
            # Upstream writes this as a six-value pad, which torch reads as 3D
            # padding on an unbatched tensor; the four-value form is the same
            # reflection on the time axis and does not depend on that reading.
            padded = F.pad(x, (0, 0, 0, n_pad), "reflect")
            conds.append(self.cond_block[i](padded))
        return conds

    def remove_weight_norm(self):
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.cond_block:
            remove_weight_norm(block)


class VectorFieldEstimator(nn.Module):
    """v(x_t, t | c): multi-period U-Nets summed over periods."""

    def __init__(
        self,
        input_dim: int,
        periods: Sequence[int] = (1, 2, 3, 5, 7),
        final_dim: int = 32,
        hidden_dim: int = 512,
        cond_rates: Sequence[int] = (5, 3),
    ):
        super().__init__()
        self.len_periods = len(periods)
        self.hidden_dim = hidden_dim
        self.MelCond = SSLConditioner(input_dim, periods, hidden_dim, cond_rates)
        # Upstream's EnCodec model, which its authors recommend over the mel
        # one, embeds t and the period at full width and halves in the MLP.
        self.time_pos_emb = SinusoidalPosEmb(hidden_dim)
        self.period_emb = nn.Embedding(self.len_periods, hidden_dim)
        nn.init.normal_(self.period_emb.weight, 0.0, hidden_dim**-0.5)
        self.register_buffer(
            "period_token",
            torch.arange(self.len_periods, dtype=torch.long),
            persistent=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.mpg = MultiPeriodGenerator(
            periods=periods, final_dim=final_dim, hidden_dim=hidden_dim
        )
        self.proj_layer = FinalBlock(final_dim)

    def cond_encoder(self, feat: torch.Tensor) -> List[torch.Tensor]:
        return self.MelCond(feat)

    def _time_period_emb(self, t: torch.Tensor) -> torch.Tensor:
        t = self.time_pos_emb(t)
        p = self.period_emb(self.period_token) * math.sqrt(self.hidden_dim)
        p = p.unsqueeze(0).expand(t.shape[0], self.len_periods, -1)
        t = torch.cat([t.unsqueeze(1).expand(-1, p.shape[1], -1), p], dim=2)
        return self.mlp(t)

    def decoder(self, x: torch.Tensor, cond: List[torch.Tensor], t: torch.Tensor):
        xs = self.mpg(x, self._time_period_emb(t), cond)
        x = torch.sum(torch.stack(xs), dim=0) / math.sqrt(self.len_periods)
        return self.proj_layer(x)

    def forward(self, x: torch.Tensor, feat: torch.Tensor, t: torch.Tensor):
        return self.decoder(x, self.cond_encoder(feat), t)

    def remove_weight_norm(self):
        self.MelCond.remove_weight_norm()
        self.mpg.remove_weight_norm()
        self.proj_layer.remove_weight_norm()


class PeriodWaveVocoder(nn.Module):
    """PeriodWave as a restoration vocoder: 50 Hz SSL features -> 48 kHz waveform.

    Args:
        input_dim: SSL feature dimension.
        periods: the folding periods; one U-Net pass each.
        hidden_dim: bottleneck width (upstream uses 512).
        final_dim: channels summed over periods before the output projection.
        cond_rates: conditioner upsampling stages; ``prod(cond_rates) * 64``
            must equal ``output_sr / 50``, so (5, 3) for 48 kHz.
        noise_scale: standard deviation of the flow's source noise.
        sigma_min: flow-matching path width at t = 1.
        num_steps, solver: ODE steps and integrator used by ``generate``.
        data_scale: gain applied to the waveform before it is matched against
            the source noise; ``generate`` divides it back out.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        periods: Sequence[int] = (1, 2, 3, 5, 7),
        hidden_dim: int = 512,
        final_dim: int = 32,
        cond_rates: Sequence[int] = (5, 3),
        noise_scale: float = 0.25,
        sigma_min: float = 1e-4,
        num_steps: int = 16,
        solver: str = "midpoint",
        data_scale: float = 10.0,
    ):
        super().__init__()
        if solver not in ("euler", "midpoint"):
            raise ValueError(f"solver must be euler or midpoint, got {solver!r}")
        # The U-Net's last decoder stage is added to the first encoder skip,
        # which carries hidden_dim // 16 channels; upstream ties the two by
        # using 512 and 32. Check it rather than fail deep inside the forward.
        if final_dim != hidden_dim // 16:
            raise ValueError(
                f"final_dim must equal hidden_dim // 16 ({hidden_dim // 16}), "
                f"got {final_dim}"
            )
        self.input_dim = input_dim
        self.noise_scale = noise_scale
        self.sigma_min = sigma_min
        self.num_steps = num_steps
        self.solver = solver
        self.data_scale = data_scale
        self.fold_factor = 64
        self.upsample_factor = int(math.prod(cond_rates)) * self.fold_factor
        self.estimator = VectorFieldEstimator(
            input_dim, periods, final_dim, hidden_dim, cond_rates
        )

    def condition(self, ssl_feat: torch.Tensor) -> List[torch.Tensor]:
        """(B, T, D) features -> per-period conditioning."""
        return self.estimator.cond_encoder(ssl_feat.transpose(1, 2))

    def velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: List[torch.Tensor]
    ) -> torch.Tensor:
        """x_t (B, 1, T), t (B,), conditioning -> v (B, 1, T)."""
        return self.estimator.decoder(x_t, cond, t)

    def flow_loss(self, x1: torch.Tensor, ssl_feat: torch.Tensor) -> torch.Tensor:
        """Conditional flow-matching MSE, as in model/periodwave_encodec.py.

        ``x1`` is (B, 1, T), already multiplied by ``data_scale`` by the caller.
        """
        cond = self.condition(ssl_feat)
        t = torch.rand([x1.size(0), 1, 1], device=x1.device, dtype=x1.dtype)
        x0 = torch.randn_like(x1) * self.noise_scale
        x_t = (1 - (1 - self.sigma_min) * t) * x0 + t * x1
        u = x1 - (1 - self.sigma_min) * x0
        pred = self.velocity(x_t, t.view(-1), cond)
        return F.mse_loss(pred, u)

    @torch.no_grad()
    def generate(
        self,
        ssl_feat: torch.Tensor,
        num_steps: Optional[int] = None,
        solver: Optional[str] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """(B, T, D) features -> (B, T * upsample_factor) waveform."""
        steps = num_steps or self.num_steps
        solver = solver or self.solver
        cond = self.condition(ssl_feat)
        n = ssl_feat.size(1) * self.upsample_factor
        x = (
            torch.randn(
                cond[0].size(0), 1, n, device=ssl_feat.device, dtype=cond[0].dtype
            )
            * self.noise_scale
            * temperature
        )
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x.size(0),), i * dt, device=x.device, dtype=x.dtype)
            if solver == "euler":
                x = x + dt * self.velocity(x, t, cond)
            else:
                k1 = self.velocity(x, t, cond)
                k2 = self.velocity(x + 0.5 * dt * k1, t + 0.5 * dt, cond)
                x = x + dt * k2
        return (x.squeeze(1) / self.data_scale).clamp(-1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D, T_frames) -> (B, 1, T): the GAN vocoders' convention."""
        return self.generate(x.transpose(1, 2)).unsqueeze(1)

    def remove_weight_norm(self) -> None:
        self.estimator.remove_weight_norm()
