"""Flow-matching training of the restoration vocoder (recipe stages 7 and 8).

Same data, collate and frozen-encoder feature extraction as the GAN vocoder
model: the velocity network trains on 1 s excerpts whose features were
computed with 8 s of surrounding speech in view. One optimizer, ESPnet's
plain Trainer. The loss is the conditional flow-matching objective
E ||v(x_t, t | c) - (x_1 - (1 - sigma_min) x_0)||^2 with t ~ U(0, 1).
"""

from typing import Optional

import torch
from torch import nn

from espnet2.rst.rst_vocoder_model import SSL_FRAME_RATE, RestorationVocoderFeatures
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class ESPnetRestorationFlowVocoderModel(RestorationVocoderFeatures, AbsESPnetModel):
    """Conditional flow matching on the GAN model's excerpts.

    Args:
        ssl_encoder: frozen feature extractor (teacher, or the stage-5 predictor).
        vocoder: a flow vocoder (FlowVocoder or PeriodWaveVocoder): anything with
            ``condition``, ``velocity``, ``generate``, ``upsample_factor`` and
            ``data_scale``; one with ``flow_loss`` owns its objective.
        use_predicted_feat: False for stage 7 (teacher on clean speech), True
            for stage 8 (predictor on degraded speech).
        input_sr, output_sr: encoder and waveform sample rates.
        segment_duration: seconds of aligned features and audio per example.
        sigma_min: residual noise scale at t = 1 of the flow-matching path.
    """

    def __init__(
        self,
        ssl_encoder: nn.Module,
        vocoder: nn.Module,
        use_predicted_feat: bool = False,
        input_sr: int = 16000,
        output_sr: int = 48000,
        segment_duration: float = 1.0,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        if (
            output_sr % SSL_FRAME_RATE != 0
            or vocoder.upsample_factor != output_sr // SSL_FRAME_RATE
        ):
            raise ValueError(
                f"vocoder upsamples by {vocoder.upsample_factor}, but "
                f"output_sr / {SSL_FRAME_RATE} Hz = {output_sr // SSL_FRAME_RATE}"
            )
        self.ssl_encoder = ssl_encoder
        self.ssl_encoder.requires_grad_(False)
        self.vocoder = vocoder
        self.use_predicted_feat = use_predicted_feat
        self.input_sr, self.output_sr = input_sr, output_sr
        self.hop = output_sr // SSL_FRAME_RATE
        self.segment_frames = max(1, int(round(segment_duration * SSL_FRAME_RATE)))
        self.segment_samples = self.segment_frames * self.hop
        self.sigma_min = sigma_min

    def forward(
        self,
        speech_ref1: torch.Tensor,
        speech_ref1_lengths: torch.Tensor,
        noisy_speech: Optional[torch.Tensor] = None,
        noisy_speech_lengths: Optional[torch.Tensor] = None,
        vocoder_crop_start: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if vocoder_crop_start is None:
            vocoder_crop_start = speech_ref1.new_zeros(
                speech_ref1.size(0), dtype=torch.long
            )
        feat, feat_lengths = self._ssl_features(
            speech_ref1, speech_ref1_lengths, noisy_speech, noisy_speech_lengths
        )
        feat, x1 = self._crop(
            feat, feat_lengths, speech_ref1, speech_ref1_lengths, vocoder_crop_start
        )
        x1 = x1.unsqueeze(1) * self.vocoder.data_scale
        if hasattr(self.vocoder, "flow_loss"):
            # PeriodWave keeps its own source-noise scale and builds the
            # conditioning per period, so it owns the objective; the path and
            # the velocity target are the same as below.
            loss = self.vocoder.flow_loss(x1, feat)
        else:
            cond = self.vocoder.condition(feat)
            x1, cond = self._trim(x1, cond)
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.size(0), device=x1.device)
            tt = t.view(-1, 1, 1)
            x_t = (1 - (1 - self.sigma_min) * tt) * x0 + tt * x1
            target = x1 - (1 - self.sigma_min) * x0
            v = self.vocoder.velocity(x_t, t, cond)
            loss = ((v - target) ** 2).mean()
        stats = dict(loss=loss.detach())
        loss, stats, weight = force_gatherable((loss, stats, x1.size(0)), loss.device)
        return loss, stats, weight

    def collect_feats(self, **batch):
        return {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.ssl_encoder.eval()
        return self
