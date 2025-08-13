# ein notation:
# b - batch, n - sequence (frames), nt - text sequence, d - mel bins, nw - raw wave len

from __future__ import annotations
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from importlib.resources import files

from espnet2.tts.abs_tts import AbsTTS

# F5-TTS imports
from f5_tts.model.modules import MelSpec
from f5_tts.infer.utils_infer import load_model
from f5_tts.model.utils import lens_to_mask

def _dev(module: nn.Module) -> torch.device:
    return next(module.parameters()).device

class F5TTS(AbsTTS):
    """Thin ESPnet wrapper around the upstream F5-TTS CFM model."""

    def __init__(
        self,
        idim: int,
        odim: int,
        *,
        # Pretrained model to fine-tune (recommended)
        pretrained_ckpt: str,
        model_cfg: str = "F5TTS_v1_Base",     # name inside f5_tts/configs/*.yaml
        use_ema: bool = True,                 # use EMA weights from ckpt
        # Mel frontend must match the upstream model defaults
        mel_spec_kwargs: dict = dict(
            n_fft=1024, hop_length=240, win_length=1024,
            n_mel_channels=100, target_sample_rate=24000,
            mel_spec_type="vocos",
        ),
        # Text vocab: if you’re using ESPnet tokenization, pass a token->id map
        vocab_char_map: dict[str, int] | None = None,
        ode_method: str = "euler",
        device: Optional[str] = None,
    ):
        super().__init__()
        self.idim = idim
        self.odim = odim

        # ---- mel frontend
        self.mel_spec = MelSpec(**mel_spec_kwargs)
        self.num_channels = self.mel_spec.n_mel_channels
        assert self.num_channels == odim, f"odim ({odim}) must equal n_mel_channels ({self.num_channels})"

        # ---- load upstream model (CFM + DiT backbone) from checkpoint
        # - this builds the transformer/backbone per config and loads weights
        # - returns an EMA-wrapped model when use_ema=True
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # load_model(model_cls, model_cfg, ckpt_path, ...)
        # model_cls is resolved inside load_model from model_cfg yaml; we just pass cfg name here
        from hydra.utils import get_class
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model_cfg}.yaml")))
        model_cls = get_class(f"f5_tts.model.{cfg.model.backbone}")

        mel_type = mel_spec_kwargs.get("mel_spec_type", "vocos")
        ema = load_model(
            model_cls, cfg.model.arch, pretrained_ckpt,
            mel_spec_type=mel_type,
            vocab_file="", ode_method=ode_method, use_ema=use_ema, device=device
        )
        # unwrap to the actual torch.nn.Module
        self.inner = ema.model if hasattr(ema, "model") else ema
        self.inner = self.inner.to(device)
        # keep training in float32 to avoid fp16 dtype mismatches
        self.inner = self.inner.float()

        # ---- vocab / “null” id used for classifier-free guidance (not critical here)
        self.vocab_char_map = vocab_char_map
        self.null_token_id = 0 if vocab_char_map is None else vocab_char_map.get("<pad>", 0)

    @property
    def device(self):
        return _dev(self)

    # ========== Training ==========
    def forward(
        self,
        text: torch.LongTensor,            # (B, Ttxt)
        text_lengths: torch.LongTensor,    # (B,) (unused by upstream, just keep for ESPnet API)
        feats: torch.FloatTensor,          # (B, T, D) mel frames
        feats_lengths: torch.LongTensor,   # (B,)
        sids: torch.LongTensor | None = None,
        spembs: torch.FloatTensor | None = None,
        **kwargs,
    ):
        # Upstream expects mel bins==100 (default). Ensure shape (B,T,D)
        if feats.dim() == 3 and feats.shape[1] == self.num_channels:
            feats = feats.transpose(1, 2)  # (B,D,T)->(B,T,D) if recipe gives channel-first
        assert feats.size(-1) == self.num_channels, f"expected mel bins D={self.num_channels}, got {feats.size(-1)}"

        # dtype/device hygiene
        feats = feats.to(self.device, dtype=torch.float32)
        text  = text.to(self.device, dtype=torch.long)
        lens  = feats_lengths.to(self.device, dtype=torch.long)

        # Call upstream CFM.forward; it returns (loss, gt_mel, pred_mel)
        loss, gt, pred = self.inner.forward(inp=feats, text=text, lens=lens)
        # Reduce to python float for logging
        stats = {"loss": float(loss.detach().cpu())}
        # Weight for dataset-average
        weight = float(feats_lengths.sum().item())
        return loss, stats, weight

    # ========== Inference ==========
    @torch.no_grad()
    def inference(
        self,
        text: torch.LongTensor,                     # (Ttxt,)
        sids: torch.LongTensor | None = None,
        spembs: torch.FloatTensor | None = None,
        duration: int | None = None,               # frames to generate
        use_ref_audio: bool = False,
        ref_audio: torch.FloatTensor | None = None,# (D,T) mel or (T,D) mel or raw wav
        steps: int = 32,
        cfg_strength: float = 1.0,
        vocoder: Callable[[torch.FloatTensor], torch.FloatTensor] | None = None,
        **kwargs,
    ):
        self.eval()
        B = 1

        # Build conditioning mel
        if use_ref_audio and ref_audio is not None:
            x = ref_audio
            if x.dim() == 1:  # raw waveform
                mel = self.mel_spec(x[None, ...])            # (1, D, T)
                cond = mel.permute(0, 2, 1)                   # (1, T, D)
            elif x.dim() == 2 and x.shape[0] == self.num_channels:
                cond = x.transpose(0, 1)[None, ...]           # (1, T, D)
            elif x.dim() == 2:
                cond = x[None, ...]                           # (1, T, D)
            else:
                raise ValueError("Bad ref_audio shape")
        else:
            T = int(duration or 400)
            cond = torch.zeros((B, T, self.num_channels), device=self.device, dtype=torch.float32)

        cond = cond.to(self.device, dtype=torch.float32)

        # Upstream sample() returns mel (B,T,D) unless you pass a vocoder to it.
        out, _ = self.inner.sample(
            cond=cond,
            text=text[None, :].to(self.device, dtype=torch.long),
            duration=cond.shape[1],
            steps=steps,
            cfg_strength=cfg_strength,
            vocoder=None,
            use_epss=True,
            no_ref_audio=not use_ref_audio,
        )

        if out.dim() == 3:  # mel
            mel = out.squeeze(0)  # (T,D)
            return {"feat_gen": mel} if vocoder is None else {"wav": vocoder(mel.transpose(0,1)[None, ...]).squeeze(0)}
        else:                # waveform
            return {"wav": out.squeeze(0)}

    # ESPnet will call this if your recipe provides feats
    def collect_feats(self, batch):
        feats = batch["feats"]
        if feats.dim() == 3 and feats.shape[1] == self.num_channels:
            feats = feats.transpose(1, 2)  # (B,D,T)->(B,T,D)
        return {"feats": feats}
