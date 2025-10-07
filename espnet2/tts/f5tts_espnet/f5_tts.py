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
import re
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
        ref_root_train: Optional[str] = None,
        ref_root_dev:   Optional[str] = None,
        ref_root_test:  Optional[str] = None,
        fallback_ref_root: Optional[str] = None,
        auto_env_ref_roots: bool = True,  # read env vars as a convenience
        kaldi_ref_root_train: Optional[str] = None,
        kaldi_ref_root_dev:   Optional[str] = None,
        kaldi_ref_root_test:  Optional[str] = None,
        kaldi_allow_cmd: bool = True, 
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
        from hydra.utils import get_class
        from f5_tts.model.cfm import CFM
        # (optional import – only used if checkpoint is safetensors; fine to keep)
        from safetensors.torch import load_file as safe_load_file

        # --- always build the network from cfg ---
        cfg = _load_f5_config(model_cfg)
        # Log a short summary of the cfg
        try:
            backbone = cfg.model.backbone
            arch_dict = dict(cfg.model.arch) if hasattr(cfg.model, "arch") else {}
            arch_keys = list(arch_dict.keys())
            more = "..." if len(arch_keys) > 8 else ""
            print(
                "[F5TTS] Loaded model_cfg="
                f"{model_cfg!r}; backbone={backbone}; "
                f"arch_keys={arch_keys[:8]}{more}; "
                f"mel_dim={self.num_channels}; text_num_embeds={text_num_embeds}",
                flush=True,
            )
        except Exception as e:
            print(f"[F5TTS] Loaded model_cfg={model_cfg!r} (summary unavailable: {e})", flush=True)
            # Fallbacks if the above failed
            backbone = cfg.model.backbone
            arch_dict = dict(cfg.model.arch) if hasattr(cfg.model, "arch") else {}

        model_cls = get_class(f"f5_tts.model.{backbone}")

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
                print(f"[F5TTS] Loading F5-TTS checkpoint from {ckpt_path} (safetensors)", flush=True)
                state = safe_load_file(ckpt_path, device=str(self._device))
            else:
                print(f"[F5TTS] Loading F5-TTS checkpoint from {ckpt_path} (torch)", flush=True)
                ckpt = torch.load(ckpt_path, map_location=self._device)
                state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

            def _strip_prefixes(sd):
                def strip_one(k):
                    for p in ("ema_model.", "model.", "module."):
                        if k.startswith(p):
                            return k[len(p):]
                    return k
                # keep only tensor entries
                return {strip_one(k): v for k, v in sd.items() if isinstance(v, torch.Tensor)}

            normalized = _strip_prefixes(state)

            # (optional) drop size-mismatch keys, e.g., text embedding if vocab differs
            model_sd = model.state_dict()
            normalized = {k: v for k, v in normalized.items()
                        if (k in model_sd) and (v.shape == model_sd[k].shape)}

            incompat = model.load_state_dict(normalized, strict=False)
            # be robust to different torch versions
            missing = getattr(incompat, "missing_keys", [])
            unexpected = getattr(incompat, "unexpected_keys", [])
            print(f"[F5TTS] Loaded F5-TTS checkpoint: {ckpt_path} "
                f"(loaded={len(normalized)}, missing={len(missing)}, unexpected={len(unexpected)})",
                flush=True)
        else:
            print("[F5TTS] No external pretrained_ckpt provided; expecting ESPnet to load weights via --model_file.",
                flush=True)

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
        self.kaldi_ref_roots: Dict[str, Optional[str]] = {"train": None, "dev": None, "test": None}
        def _norm(p: Optional[str]) -> Optional[str]:
            if not p: return None
            return os.path.abspath(os.path.expanduser(p))

        self.kaldi_ref_roots["train"] = _norm(kaldi_ref_root_train)
        self.kaldi_ref_roots["dev"]   = _norm(kaldi_ref_root_dev)
        self.kaldi_ref_roots["test"]  = _norm(kaldi_ref_root_test)

        # tiny caches populated on first use
        self._kaldi_wavscp: Dict[str, Dict[str, str]] = {}   # split -> {utt_id: wav_entry}
        self._kaldi_u2s:    Dict[str, Dict[str, str]] = {}   # split -> {utt_id: spk_id}
        self.kaldi_allow_cmd = bool(kaldi_allow_cmd)
        print(
    "[F5TTS] kaldi_ref_roots: "
    f"train={self.kaldi_ref_roots['train']}, "
    f"dev={self.kaldi_ref_roots['dev']}, "
    f"test={self.kaldi_ref_roots['test']}",
    flush=True,
)

        # Allow passing roots from YAML
        roots_from_args = {
            "train": _norm(ref_root_train),
            "dev":   _norm(ref_root_dev),
            "test":  _norm(ref_root_test),
        }

        # Optional: allow environment overrides if user prefers (e.g., for quick tests)
        if auto_env_ref_roots:
            roots_from_env = {
                "train": _norm(os.getenv("F5TTS_REF_ROOT_TRAIN")),
                "dev":   _norm(os.getenv("F5TTS_REF_ROOT_DEV")),
                "test":  _norm(os.getenv("F5TTS_REF_ROOT_TEST")),
            }
            for k, v in roots_from_env.items():
                if v: roots_from_args[k] = v

            fb_env = _norm(os.getenv("F5TTS_REF_ROOT"))
            if fb_env:
                fallback_ref_root = fb_env

        # Apply any provided roots
        for split in ("train", "dev", "test"):
            if roots_from_args[split]:
                self.ref_roots[split] = roots_from_args[split]

        # Optional fallback (legacy single root with per-speaker subdirs)
        if fallback_ref_root:
            self.fallback_ref_root = _norm(fallback_ref_root)

        # Friendly logging + existence hints
        def _ok(p): 
            return (p is not None) and os.path.isdir(p)
        print(
            "[F5TTS] ref_roots: "
            f"train={self.ref_roots['train']} ({'ok' if _ok(self.ref_roots['train']) else 'missing'}), "
            f"dev={self.ref_roots['dev']} ({'ok' if _ok(self.ref_roots['dev']) else 'missing'}), "
            f"test={self.ref_roots['test']} ({'ok' if _ok(self.ref_roots['test']) else 'missing'}); "
            f"fallback={self.fallback_ref_root or 'None'} "
            f"({ 'ok' if _ok(self.fallback_ref_root) else ('missing' if self.fallback_ref_root else 'n/a') })",
            flush=True,
        )

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
        print(f"[F5TTS] ref_roots: "
      f"train={self.ref_roots['train']}, "
      f"dev={self.ref_roots['dev']}, "
      f"test={self.ref_roots['test']}", flush=True)
    
    def _load_kaldi_tables(self, split: str):
        """Lazy-load wav.scp + utt2spk for a split into caches."""
        root = self.kaldi_ref_roots.get(split)
        if not root or split in self._kaldi_wavscp:
            return
        wavscp_path = os.path.join(root, "wav.scp")
        u2s_path    = os.path.join(root, "utt2spk")
        if not (os.path.isfile(wavscp_path) and os.path.isfile(u2s_path)):
            return
        wavscp = {}
        with open(wavscp_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                key, rest = line.split(maxsplit=1)
                wavscp[key] = rest
        u2s = {}
        with open(u2s_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                utt, spk = line.split()
                u2s[utt] = spk
        self._kaldi_wavscp[split] = wavscp
        self._kaldi_u2s[split] = u2s

    def _kaldi_pick_utt_for_spk(self, split: str, key_int: int, choose: str = "first") -> Optional[str]:
        """Return a utt id in this split whose speaker matches key_int (supports idXXXX and plain XXXX)."""
        self._load_kaldi_tables(split)
        if split not in self._kaldi_u2s: return None
        u2s = self._kaldi_u2s[split]
        target_a = f"id{key_int}"
        target_b = str(key_int)
        cands = [u for u, spk in u2s.items() if spk == target_a or spk == target_b]
        if not cands: return None
        cands.sort()
        if choose == "random":
            import random
            return random.choice(cands)
        return cands[0]

    def _load_wav_from_kaldi_entry(self, entry: str) -> torch.FloatTensor:
        """
        entry is from wav.scp: either a filepath or a shell command ending with '|'.
        Returns 1-D float32 tensor at self.target_sr.
        """
        import io, subprocess
        import soundfile as sf
        # Case 1: plain path
        if "|" not in entry and os.path.isfile(entry):
            wav, sr = sf.read(entry, dtype="float32", always_2d=False)
            x = torch.from_numpy(wav).float().to(self.device)
        else:
            if not self.kaldi_allow_cmd:
                raise RuntimeError("wav.scp contains a command, but kaldi_allow_cmd=False")
            # Ensure the command outputs a WAV stream (most Kaldi recipes already do).
            # Run the command and capture stdout bytes.
            proc = subprocess.run(entry, shell=True, stdout=subprocess.PIPE, check=True)
            buf = io.BytesIO(proc.stdout)
            wav, sr = sf.read(buf, dtype="float32", always_2d=False)
            x = torch.from_numpy(wav).float().to(self.device)
        # Resample if needed
        if sr != self.target_sr:
            import torchaudio.functional as AF
            x = AF.resample(x, sr, self.target_sr)
        return x

        
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
        # NEW controls:
        prefer_split_ref: bool = False,       # use split files even if ref_audio is provided
        ref_choose: str = "first",            # "first" | "random" for multiple matches
        **kwargs,
    ):
        self.eval()
        if seed is not None:
            g = torch.Generator(device=str(self.device))
            if seed is not None:
                import random
                random.seed(int(seed))
            g.manual_seed(int(seed))
        else:
            g = None

        B = 1
        cond: Optional[torch.FloatTensor] = None
        ref_info = "none"
        ref_src: str = "none"  # human-friendly tag for where the ref came from
        ref_path: Optional[str] = None
        used_zero = False
        used_split = False

        # --- infer speaker key from utt_id like "id10270-xxx"
        def _infer_key_from_utt(uid: Optional[str]) -> Optional[int]:
            if not uid:
                return None
            import re
            m = re.match(r"^id(\d+)\b", uid) or re.match(r"^(\d+)\b", uid)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
            return None

        # ---- Try to build conditioning from reference(s)
        if use_ref_audio:
            # decide key: explicit > sids[0] > infer from utt_id
            key = ref_key
            if key is None and sids is not None and sids.numel() > 0:
                key = int(sids.view(-1)[0].item())
            if key is None:
                key = _infer_key_from_utt(utt_id)

            # source priority: tensor first unless prefer_split_ref=True
            try_tensor_first = (not prefer_split_ref)
            # ---------- (A) explicit tensor ----------
            if try_tensor_first and (ref_audio is not None):
                cond = self._ref_to_cond(ref_audio)
                ref_info = "explicit ref_audio tensor"
                ref_src  = "tensor"
            # ---------- (B) registered reference ----------
            if cond is None and key is not None and key in self._ref_db:
                src = self._ref_db[key]
                cond = self._ref_to_cond(src)
                ref_info = f"registered reference for key={key}"
                ref_src  = "registered"
                ref_path = src if isinstance(src, str) else None

            # ---------- (C) split search ----------
            if cond is None and key is not None:
                tried_info = []

                def _pick(cands):
                    if not cands: return None
                    cands = sorted(cands)
                    if ref_choose == "random":
                        import random
                        return random.choice(cands)
                    return cands[0]

                def _try_split(split_name: str) -> Optional[torch.FloatTensor]:
                    root = self.ref_roots.get(split_name) if hasattr(self, "ref_roots") else None
                    if not root:
                        tried_info.append(f"{split_name}:<no-root>")
                        return None
                    spk_dir = os.path.join(root, str(key))
                    if os.path.isdir(spk_dir):
                        import glob as _glob, os as _os
                        cands = _glob.glob(os.path.join(spk_dir, ref_glob))
                        chosen = _pick(cands)
                        if chosen:
                            abs_chosen = _os.path.abspath(chosen)
                            tried_info.append(f"{split_name}:{abs_chosen}")
                            nonlocal ref_path, ref_src, used_split
                            ref_path = abs_chosen
                            ref_src  = f"split:{split_name}"
                            used_split = True
                            return self._ref_to_cond(chosen)
                    tried_info.append(f"{split_name}:<none>")
                    return None

                cond = _try_split(ref_split)
                if cond is None:
                    for other in ("train", "dev", "test"):
                        if other == ref_split:
                            continue
                        cond = _try_split(other)
                        if cond is not None:
                            break

                if cond is None and self.fallback_ref_root and key is not None:
                    import glob as _glob, os as _os, random
                    spk_dir = os.path.join(self.fallback_ref_root, str(key))
                    if os.path.isdir(spk_dir):
                        cands = _glob.glob(os.path.join(spk_dir, ref_glob))
                        chosen = _pick(cands)
                        if chosen:
                            abs_chosen = _os.path.abspath(chosen)
                            cond = self._ref_to_cond(chosen)
                            tried_info.append(f"legacy:{abs_chosen}")
                            ref_path = abs_chosen
                            ref_src  = "legacy"
                            used_split = True

                if cond is not None and tried_info:
                    ref_info = " | ".join(tried_info)
                # ---------- (C2) Kaldi-style split (wav.scp + utt2spk) ----------
                if cond is None and key is not None:
                    # first: try the requested split
                    utt = self._kaldi_pick_utt_for_spk(ref_split, key, choose=ref_choose)
                    tried_info = []
                    if utt is None:
                        # fallback: other splits
                        for other in ("train", "dev", "test"):
                            if other == ref_split: continue
                            utt = self._kaldi_pick_utt_for_spk(other, key, choose=ref_choose)
                            if utt:
                                ref_split = other
                                break
                    if utt is not None:
                        wav_entry = self._kaldi_wavscp[ref_split][utt]
                        try:
                            x = self._load_wav_from_kaldi_entry(wav_entry)
                            mel = self.mel_spec(x[None, ...])            # (1, D, T)
                            cond = mel.permute(0, 2, 1).contiguous()     # (1, T, D)
                            ref_src = f"kaldi:{ref_split}"
                            # If wav.scp was a simple path, keep absolute path for logging; else leave empty
                            ref_path = wav_entry if (("|" not in wav_entry) and os.path.isabs(wav_entry)) else None
                            tried_info.append(f"{ref_split}:{utt}")
                            ref_info = " | ".join(tried_info) if tried_info else f"{ref_split}:{utt}"
                        except Exception as e:
                            warnings.warn(f"Kaldi ref load failed for {utt}: {e}")

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
                            "" if (ref_key is None and not (sids is not None and sids.numel() > 0) and _infer_key_from_utt(utt_id) is None)
                              else str(ref_key if ref_key is not None else (int(sids.view(-1)[0].item()) if (sids is not None and sids.numel() > 0) else (_infer_key_from_utt(utt_id) or ""))),
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