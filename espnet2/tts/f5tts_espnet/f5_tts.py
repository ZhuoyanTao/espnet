# espnet2/tts/f5tts_espnet.py
# -----------------------------------------------------------------------------
# ESPnet2 <-> F5-TTS thin wrapper
#
# Usage in your YAML:
#   tts: f5tts_espnet
#   tts_conf:
#     pretrained_ckpt: /path/to/model_xxx.safetensors  # or a dir that contains it
#     model_cfg: F5TTS_v1_Base
#     use_ema: true
#     mel_spec_kwargs:
#       n_fft: 1024
#       hop_length: 240
#       win_length: 1024
#       n_mel_channels: 100
#       target_sample_rate: 24000
#       mel_spec_type: vocos
#
# This expects F5-TTS installed and importable as `f5_tts`.
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import glob
import warnings
from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from espnet2.tts.abs_tts import AbsTTS

# F5-TTS imports
try:
    from f5_tts.model.modules import MelSpec
    from f5_tts.infer.utils_infer import load_model
except Exception as e:
    raise ImportError(
        "Could not import F5-TTS. Please `pip install -e .` inside the F5-TTS repo "
        "or ensure it is on PYTHONPATH. Underlying error:\n" + repr(e)
    )

# Optional helpers
def _dev(module: nn.Module) -> torch.device:
    return next(module.parameters()).device

def _resolve_ckpt(path_or_dir: str) -> str:
    """
    Accepts:
      - exact file to .safetensors / .pt
      - a directory containing one of those files
    Returns:
      absolute path to checkpoint file.
    """
    path_or_dir = os.path.expanduser(path_or_dir)
    if os.path.isfile(path_or_dir):
        return os.path.abspath(path_or_dir)

    if os.path.isdir(path_or_dir):
        # prefer *.safetensors, then *.pt, then model_last.*
        pats = [
            os.path.join(path_or_dir, "**", "*.safetensors"),
            os.path.join(path_or_dir, "**", "model_*.pt"),
            os.path.join(path_or_dir, "**", "model_last.pt"),
            os.path.join(path_or_dir, "**", "*.pt"),
        ]
        for pat in pats:
            hits = sorted(glob.glob(pat, recursive=True))
            if hits:
                return os.path.abspath(hits[0])

    raise FileNotFoundError(
        f"pretrained_ckpt not found. Gave: {path_or_dir!r}. "
        "Point to a .safetensors/.pt file or a directory that contains one."
    )

def _load_f5_config(model_cfg: str):
    """
    model_cfg may be a bare name like 'F5TTS_v1_Base' (searched under f5_tts/configs/)
    or a direct path to a YAML file.
    """
    try:
        from importlib.resources import files as pkg_files
        from omegaconf import OmegaConf
        if os.path.isfile(model_cfg):
            return OmegaConf.load(model_cfg)
        cfg_path = pkg_files("f5_tts").joinpath(f"configs/{model_cfg}.yaml")
        return OmegaConf.load(str(cfg_path))
    except Exception as e:
        raise RuntimeError(
            f"Failed to load F5-TTS config for model_cfg={model_cfg!r}. "
            "Pass a valid cfg name (existing under f5_tts/configs/) or an absolute YAML path."
        ) from e

def _get_f5_model_cls(cfg) -> type:
    # F5-TTS configs typically expose model.backbone (module) and model.arch (kwargs)
    try:
        from hydra.utils import get_class
        backbone = cfg.model.backbone  # e.g. "DiT"
        return get_class(f"f5_tts.model.{backbone}")
    except Exception as e:
        raise RuntimeError(
            "Could not resolve F5-TTS model class from config "
            "(expected cfg.model.backbone)."
        ) from e


class F5TTSEspnet(AbsTTS):
    """ESPnet2 wrapper around the upstream F5-TTS CFM model."""

    def __init__(
        self,
        idim: int,
        odim: int,
        *,
        pretrained_ckpt: str,
        model_cfg: str = "F5TTS_v1_Base",     # name in f5_tts/configs/ or YAML path
        use_ema: bool = True,                 # use EMA weights
        # Mel frontend must match F5-TTS defaults used during pretraining
        mel_spec_kwargs: Dict = dict(
            n_fft=1024, hop_length=240, win_length=1024,
            n_mel_channels=100, target_sample_rate=24000,
            mel_spec_type="vocos",
        ),
        # Optional: if you're using ESPnet tokenization for phonemes/chars, pass a map
        vocab_char_map: Optional[Dict[str, int]] = None,
        ode_method: str = "euler",
        device: Optional[str] = None,
        freeze_backbone: bool = False,
        dtype: Union[str, torch.dtype] = "float32",
        force_text_num_embeds: Optional[int] = None,
    ):
        super().__init__()
        self.idim = idim
        self.odim = odim
        text_num_embeds = int(force_text_num_embeds) if force_text_num_embeds else int(self.idim)

        # ---- device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
       # Map dtype to torch dtype (accept str or torch.dtype)
        torch_dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, str(dtype), None)
        if torch_dtype is None:
            torch_dtype = torch.float32

        # ---- mel frontend (time-major mels: (B, T, D))
        self.mel_spec = MelSpec(**mel_spec_kwargs)
        self.num_channels = int(self.mel_spec.n_mel_channels)
        if self.num_channels != odim:
            raise ValueError(
                f"odim ({odim}) must match n_mel_channels ({self.num_channels}). "
                "Adjust `n_mel_channels` in tts_conf.mel_spec_kwargs and ESPnet feature dims."
            )

        # ---- load upstream model (CFM + DiT backbone) from checkpoint
        # ---- load upstream model (CFM + DiT backbone) from checkpoint (manual path)
        from hydra.utils import get_class
        from f5_tts.model.cfm import CFM
        from safetensors.torch import load_file as safe_load_file

        # --- always build the network from cfg ---
        cfg = _load_f5_config(model_cfg)

        backbone  = cfg.model.backbone
        model_cls = get_class(f"f5_tts.model.{backbone}")
        arch_dict = dict(cfg.model.arch) if hasattr(cfg.model, "arch") else {}

        transformer = model_cls(
            **arch_dict,
            text_num_embeds=text_num_embeds,
            mel_dim=int(self.num_channels),
        )
        model = CFM(transformer=transformer).to(self._device)

        # --- only load external weights if a checkpoint path is given ---
        if pretrained_ckpt and str(pretrained_ckpt).strip():
            ckpt_path = _resolve_ckpt(pretrained_ckpt)
            if ckpt_path.endswith(".safetensors"):
                from safetensors.torch import load_file as safe_load_file
                state = safe_load_file(ckpt_path, device=str(self._device))
            else:
                ckpt = torch.load(ckpt_path, map_location=self._device)
                state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

            def _strip_prefixes(sd):
                def strip_one(k):
                    for p in ("ema_model.", "model.", "module."):
                        if k.startswith(p):
                            return k[len(p):]
                    return k
                return {strip_one(k): v for k, v in sd.items() if isinstance(v, torch.Tensor)}

            normalized = _strip_prefixes(state)

            # (optional) drop size-mismatch keys, e.g., text embedding if vocab differs
            model_sd = model.state_dict()
            drop = [k for k,v in normalized.items() if k in model_sd and v.shape != model_sd[k].shape]
            for k in drop: normalized.pop(k, None)

            model.load_state_dict(normalized, strict=False)

        self.inner = model


        if freeze_backbone:
            for p in self.inner.parameters():
                p.requires_grad = False
            warnings.warn("F5-TTS backbone is frozen (requires_grad=False).")

        # ---- token map / null id for classifier-free guidance (not critical)
        self.vocab_char_map = vocab_char_map
        self.null_token_id = 0 if vocab_char_map is None else vocab_char_map.get("<pad>", 0)
        self._ref_db: Dict[Union[int, str], Union[str, torch.Tensor]] = {}
        self.fallback_ref_root: Optional[str] = None  # settable later via attribute or YAML hook
        self.ref_roots: Dict[str, Optional[str]] = {"train": None, "dev": None, "test": None}
        # cache target sample rate robustly
        self.target_sr = int(getattr(self.mel_spec, "target_sample_rate",
                          mel_spec_kwargs.get("target_sample_rate", 24000)))
        
    def set_ref_roots(self, *, train: Optional[str] = None,
                            dev: Optional[str] = None,
                            test: Optional[str] = None) -> None:
        """Set directories that contain per-speaker subfolders for each split."""
        if train is not None: self.ref_roots["train"] = train
        if dev   is not None: self.ref_roots["dev"]   = dev
        if test  is not None: self.ref_roots["test"]  = test
        
    def _ref_to_cond(self, ref: Union[str, torch.Tensor]) -> torch.FloatTensor:
        """Turn a file path or tensor into (1, T, D) mel cond on the right device."""
        if isinstance(ref, str):
            # lazy-load wav (mono) -> mel
            import soundfile as sf
            wav, sr = sf.read(ref)  # numpy array [T]
            x = torch.from_numpy(wav).float().to(self.device)
            if sr != self.target_sr:
                import torchaudio.functional as AF
                x = AF.resample(x, sr, self.target_sr)
            mel = self.mel_spec(x[None, ...])          # (1, D, T)
            cond = mel.permute(0, 2, 1).contiguous()   # (1, T, D)
            return cond

        x = ref
        if x.dim() == 1:
            mel = self.mel_spec(x[None, ...].to(self.device))   # (1, D, T)
            return mel.permute(0, 2, 1).contiguous()            # (1, T, D)
        if x.dim() == 2 and x.shape[0] == self.num_channels:
            return x.transpose(0, 1)[None, ...].to(self.device) # (1, T, D)
        if x.dim() == 2 and x.shape[1] == self.num_channels:
            return x[None, ...].to(self.device)                 # (1, T, D)
        raise ValueError("Reference must be path, raw wav (T,), or mel (D,T)/(T,D).")

    def register_reference(self, key: Union[int, str], ref: Union[str, torch.Tensor]) -> None:
        """Register a reference for a speaker/key. ref can be a file path or tensor."""
        self._ref_db[key] = ref

    def clear_references(self) -> None:
        self._ref_db.clear()

    # Public helper
    @property
    def device(self) -> torch.device:
        return self._device

    # ================= Training =================
    def forward(
        self,
        text: torch.LongTensor,            # (B, Ttxt)
        text_lengths: torch.LongTensor,    # (B,) (not used by upstream; keep for API)
        feats: torch.FloatTensor,          # (B, T, D) mel frames (time-major)
        feats_lengths: torch.LongTensor,   # (B,)
        sids: Optional[torch.LongTensor] = None,
        spembs: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        # Normalize mel layout to (B, T, D)
        if feats.dim() == 3 and feats.shape[1] == self.num_channels:
            # (B, D, T) -> (B, T, D)
            feats = feats.transpose(1, 2)

        if feats.size(-1) != self.num_channels:
            raise ValueError(
                f"Expected mel last-dim D={self.num_channels}, got {feats.size(-1)}"
            )

        # dtype / device
        feats = feats.to(self.device, dtype=torch.float32)
        text = text.to(self.device, dtype=torch.long)
        lens = feats_lengths.to(self.device, dtype=torch.long)

        # stats = {"loss": float(loss.detach().cpu())}
        # weight = feats_lengths.to(loss.device, dtype=loss.dtype).sum()  # Tensor, not float
        # return loss, stats, weight
        #  - stats values must be torch.Tensors (not Python floats)
        #  - weight must be a 1-D vector of per-item weights (shape: [B,])
        # Upstream returns (loss, gt_mel, pred_mel)
        loss, gt, pred = self.inner.forward(inp=feats, text=text, lens=lens)

        # Batch size
        B = feats.shape[0]

        # Make stats per-utterance: expand scalar loss to (B,)
        per_utt_loss = loss.detach().expand(B)             # tensor, shape [B]

        # Weights must be 1-D (B,) and same dtype/device as loss
        weight = feats_lengths.to(loss.device, dtype=loss.dtype)  # shape [B]

        stats = {"loss": per_utt_loss}
        return loss, stats, weight

    # ================= Inference =================
    @torch.no_grad()
    def inference(
        self,
        text: torch.LongTensor,
        sids: Optional[torch.LongTensor] = None,
        spembs: Optional[torch.FloatTensor] = None,
        duration: Optional[int] = None,
        use_ref_audio: bool = False,
        ref_audio: Optional[torch.FloatTensor] = None,
        steps: int = 32,
        cfg_strength: float = 1.0,
        vocoder: Optional[Callable[[torch.FloatTensor], torch.FloatTensor]] = None,
        seed: Optional[int] = None,
        ref_key: Optional[Union[int, str]] = None,
        # NEW:
        ref_split: str = "train",
        ref_glob: str = "*.wav",
        utt_id: Optional[str] = None,
        ref_log_path: Optional[str] = None,   # if set, append CSV lines
        **kwargs,
    ):
        self.eval()
        if seed is not None:
            g = torch.Generator(device=str(self.device))
            g.manual_seed(int(seed))
        else:
            g = None

        B = 1
        cond = None  # <-- initialize
        ref_info = "none"
        B = 1
        cond: Optional[torch.FloatTensor] = None
        ref_info = "none"
        ref_src: str = "none"  # human-friendly tag for where the ref came from
        ref_path: Optional[str] = None
        used_zero = False

        # ---- Try to build conditioning from reference(s)
        if use_ref_audio:
            if ref_audio is not None:
                cond = self._ref_to_cond(ref_audio)
                ref_info = "explicit ref_audio tensor"
                ref_src  = "tensor"
            else:
                key = ref_key
                if key is None and sids is not None and sids.numel() > 0:
                    key = int(sids.view(-1)[0].item())

                # 1) registered ref
                if cond is None and key is not None and key in self._ref_db:
                    src = self._ref_db[key]
                    cond = self._ref_to_cond(self._ref_db[key])
                    ref_info = f"registered reference for key={key}"
                    cond = self._ref_to_cond(src)
                    ref_info = f"registered reference for key={key}"
                    ref_src  = "registered"
                    ref_path = src if isinstance(src, str) else None

                # 2) split-aware fallback search (prefer ref_split, then others)
                if cond is None and key is not None:
                    tried_info = []

                    def _try_split(split_name: str) -> Optional[torch.FloatTensor]:
                        root = self.ref_roots.get(split_name) if hasattr(self, "ref_roots") else None
                        if not root:
                            tried_info.append(f"{split_name}:<no-root>")
                            return None
                        spk_dir = os.path.join(root, str(key))
                        if os.path.isdir(spk_dir):
                            import glob as _glob
                            cands = sorted(_glob.glob(os.path.join(spk_dir, ref_glob)))
                            if cands:
                                # deterministic: pick first (unless user seeds and randomizes elsewhere)
                                chosen = cands[0]
                                tried_info.append(f"{split_name}:{chosen}")
                                nonlocal ref_path, ref_src
                                ref_path = chosen
                                ref_src  = f"split:{split_name}"
                                return self._ref_to_cond(chosen)
                        tried_info.append(f"{split_name}:<none>")
                        return None

                    # try preferred split
                    cond = _try_split(ref_split)

                    # fall back to the other splits
                    if cond is None:
                        for other in ("train", "dev", "test"):
                            if other == ref_split:
                                continue
                            cond = _try_split(other)
                            if cond is not None:
                                break

                    # legacy single-root fallback
                    if cond is None and self.fallback_ref_root:
                        spk_dir = os.path.join(self.fallback_ref_root, str(key))
                        if os.path.isdir(spk_dir):
                            import random, glob as _glob
                            cands = _glob.glob(os.path.join(spk_dir, ref_glob))
                            if cands:
                                chosen = random.choice(cands)
                                cond = self._ref_to_cond(chosen)
                                tried_info.append(f"legacy:{chosen}")

                    if cond is not None:
                        ref_info = " | ".join(tried_info)

        # ---- If still no cond, use zeros of requested duration
        if cond is None:
            T = int(duration or 400)
            cond = torch.zeros((1, T, self.num_channels), device=self.device, dtype=torch.float32)
            no_ref = True
            ref_info = f"zeros (no reference found, duration={T})"
            ref_src  = "zeros"
            used_zero = True
        else:
            no_ref = False

        print(f"[F5TTS inference] Using reference: {ref_info}")
        
        text = text.to(self.device, dtype=torch.long)[None, :]  # (1, Ttxt)

        # Upstream sample() returns mel (B,T,D) unless a vocoder is provided
        out, _ = self.inner.sample(
            cond=cond,
            text=text,
            duration=cond.shape[1],
            steps=int(steps),
            cfg_strength=float(cfg_strength),
            vocoder=None,               # keep mel; let ESPnet run its own vocoder if desired
            use_epss=True,
            no_ref_audio=no_ref,
        )

        if out.dim() == 3:  # mel: (1, T, D)
            mel = out.squeeze(0)  # (T, D)
            # ---- Append a one-line CSV record if requested
            if ref_log_path is not None:
                try:
                    import csv
                    _dir = os.path.dirname(ref_log_path)
                    if _dir:
                        os.makedirs(_dir, exist_ok=True)
                    # write header once if file doesn't exist or is empty
                    need_header = (not os.path.exists(ref_log_path)) or (os.path.getsize(ref_log_path) == 0)
                    with open(ref_log_path, "a", newline="") as f:
                        w = csv.writer(f)
                        # header suggestion:
                        # utt_id, ref_key, ref_split, ref_src, ref_path, used_zero, duration
                        if need_header:
                           w.writerow(["utt_id","ref_key","ref_split","ref_src","ref_path","used_zero","duration"])
                        w.writerow([
                            str(utt_id or ""),
                            "" if ref_key is None else str(ref_key),
                            str(ref_split),
                            ref_src,
                            "" if ref_path is None else ref_path,
                            int(used_zero),
                            int(cond.shape[1]),
                        ])
                except Exception as _e:
                    warnings.warn(f"Failed to append ref log: {ref_log_path!r}: {_e}")

            if vocoder is None:
                return {"feat_gen": mel}
            wav = vocoder(mel.transpose(0, 1)[None, ...]).squeeze(0)  # (T,)
            return {"wav": wav}
        else:
            # waveform already
            return {"wav": out.squeeze(0)}

    # ================= ESPnet hook (optional) =================
    def collect_feats(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ESPnet may call this to cache precomputed features."""
        feats = batch["feats"]
        if feats.dim() == 3 and feats.shape[1] == self.num_channels:
            feats = feats.transpose(1, 2)  # (B, D, T) -> (B, T, D)
        return {"feats": feats}


# Alias that many ESPnet dynamic loaders will look for (snake -> Camel heuristics vary).
# This makes `tts: f5tts_espnet` work regardless of the exact class name guess.
F5TTS = F5TTSEspnet
__all__ = ["F5TTSEspnet", "F5TTS"]
