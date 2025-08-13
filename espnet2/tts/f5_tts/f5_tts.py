"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

"""F5 TTS related modules for ESPnet2."""

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)

from espnet2.tts.abs_tts import AbsTTS
import importlib

def import_string(path): m, c = path.rsplit(".", 1); return getattr(importlib.import_module(m), c)
def build_f5_transformer(target: str, **conf): return import_string(target)(**conf)

class F5TTS(AbsTTS):
    def __init__(
        self,
        idim: int,      
        odim: int,  
        transformer: nn.Module | None = None,        # ← allow None
        transformer_target: str | None = None,       # ← class path, e.g. "f5_tts.model.transformer.FlowTransformer"
        transformer_conf: dict | None = None,        # ← kwargs
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()
        self.idim = idim
        self.odim = odim
        self.frac_lengths_mask = frac_lengths_mask
        self.null_token_id = 0     # must be >= 0; use your <pad> or <unk> id
        self.pad_id = -1           # only for length heuristics, not fed to the model
        
        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        if transformer is None:
            assert transformer_target is not None, "Provide transformer or transformer_target"
            transformer = build_f5_transformer(transformer_target, **(transformer_conf or {}))
        self.transformer = transformer
        self.dim = getattr(self.transformer, "dim", None)   # avoid AttributeError

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map


    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(
        self,
        text: torch.LongTensor,            # (B, Ttxt)
        text_lengths: torch.LongTensor,    # (B,)
        feats: torch.FloatTensor,          # (B, Tmel, D)  OR (B, D, T) depending on your choice
        feats_lengths: torch.LongTensor,   # (B,)
        sids: torch.LongTensor | None = None,
        spembs: torch.FloatTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, float], float]:
        
        if feats.dim() == 3 and feats.shape[1] == self.num_channels:
            feats = feats.transpose(1, 2)  # (B, D, T) -> (B, T, D)

        assert feats.dim() == 3, f"feats must be 3D, got {feats.shape}"
        B, T, D = feats.shape
        assert D == self.num_channels, f"mel dim {D} != num_channels {self.num_channels}"
        device, dtype = feats.device, feats.dtype
        lens = feats_lengths

        mask = lens_to_mask(lens, length=T).to(device)  # (B, T)

        # random span mask for infilling
        frac_lengths = torch.zeros((B,), device=device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths) & mask  # (B, T)

        # x1 = ground truth mel; x0 = gaussian
        x1 = feats
        x0 = torch.randn_like(x1)

        # time scalar per sample
        time = torch.rand((B,), dtype=dtype, device=device)
        t = time[:, None, None]

        phi = (1 - t) * x0 + t * x1          # φ_t(x)
        flow = x1 - x0                       # target flow

        # conditional audio (masked x1)
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)

        # classifier-free guidance drops
        drop_audio_cond = random() < self.audio_drop_prob
        drop_text = (random() < self.cond_drop_prob)

        # text is already int ids from ESPnet
        txt_in = text if not drop_text else torch.full_like(text, self.null_token_id)
        pred = self.transformer(
            x=phi,
            cond=cond if not drop_audio_cond else torch.zeros_like(cond),
            text=txt_in,
            time=time,
            mask=mask,   # if your transformer wants (B,T,1), use mask.unsqueeze(-1)
        )

        loss = F.mse_loss(pred.float(), flow.float(), reduction="none")
        loss = loss[rand_span_mask]                  # only train on masked span
        loss = loss.mean()

        stats = {"loss": float(loss.detach().cpu())}
        weight = float(feats_lengths.sum().item())   # frame-weighted averaging
        return loss, stats, weight



    @torch.no_grad()
    def inference(
        self,
        text: torch.LongTensor,                     # (Ttxt,)
        sids: torch.LongTensor | None = None,
        spembs: torch.FloatTensor | None = None,
        duration: int | None = None,
        use_ref_audio: bool = False,
        ref_audio: torch.FloatTensor | None = None, # (D, T) or (T, D)
        steps: int = 32,
        cfg_strength: float = 1.0,
        vocoder: Callable[[torch.FloatTensor], torch.FloatTensor] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        self.eval()
        B = 1

        # build cond mel
        if use_ref_audio and ref_audio is not None:
            if ref_audio.dim() == 2 and ref_audio.shape[0] == self.num_channels:
                ref = ref_audio.transpose(0, 1)[None, ...]   # (1, T, D)
            elif ref_audio.dim() == 2:
                ref = ref_audio[None, ...]                   # (1, T, D)
            else:
                # raw waveform -> mel
                ref = self.mel_spec(ref_audio)[None, ...].transpose(1, 2)
        else:
            # no reference -> zeros, length ~ duration
            T = int(duration or 400)  # pick a sane default
            ref = torch.zeros((B, T, self.num_channels), device=self.device, dtype=next(self.parameters()).dtype)


        out, _ = self.sample(
            cond=ref,
            text=text.unsqueeze(0),
            duration=duration or ref.shape[1],
            steps=steps,
            cfg_strength=cfg_strength,
            vocoder=None,                   # do mel->wav here if you pass one below
            use_epss=True,
            no_ref_audio=not use_ref_audio,
        )

        # out is mel (B, T, D) if no vocoder was used inside sample()
        if out.dim() == 3:  # mel
            return {"feat_gen": out.squeeze(0)} if vocoder is None else {"wav": vocoder(out.transpose(1,2)).squeeze(0)}
        else:               # waveform
            return {"wav": out.squeeze(0)}


    def collect_feats(self, batch):
        feats = batch["feats"]
        if feats.dim() == 3 and feats.shape[1] == self.num_channels:
            feats = feats.transpose(1, 2)
        return {"feats": feats}

    
    
    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        text_lengths: torch.LongTensor | None = None,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        use_epss=True,
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # raw wave
        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode
        

        def fn(t, x):
            # at each step, conditioning is fixed
            # step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

            # predict flow (cond)
            if cfg_strength < 1e-5:
                pred = self.transformer(
                    x=x,
                    cond=step_cond,
                    text=text,
                    time=t,
                    mask=mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
                return pred

            # predict flow (cond and uncond), for classifier-free guidance
            pred_cfg = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                cfg_infer=True,
                cache=True,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory



    
