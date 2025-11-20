#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Vocos wrapper (HF) with the same ergonomics as Spectrogram2Waveform."""

from __future__ import annotations
from typing import Optional, Union

import numpy as np
import torch

def _lazy_import_vocos():
    from vocos import Vocos  # type: ignore
    return Vocos

class VocosSpectrogram2Waveform(object):
    """
    Flexible mel -> waveform wrapper.

    Accepts mel in any of:
      (T, n_mels), (n_mels, T), (B, T, n_mels), (B, n_mels, T)  [torch or numpy]
    Returns:
      (T,) if B==1 else (B, T)
    """

    def __init__(
        self,
        repo_or_path: str = "charactr/vocos-mel-24khz",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        assume_input: str = "vocos_mel",   # "vocos_mel" | "log_mel"
        target_sr: int = 24000,
    ):
        self.repo_or_path = repo_or_path
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device: torch.device = device
        self.dtype = dtype
        self.assume_input = assume_input
        self.target_sr = int(target_sr)
        self.fs = self.target_sr          # what ESPnet usually looks for
        self.sample_rate = self.target_sr  # some code uses this
        self.sr = self.target_sr           # and some use this

        Vocos = _lazy_import_vocos()
        self.vocos = Vocos.from_pretrained(repo_or_path).to(self.device)
        self.vocos.eval()
        if self.dtype is not None:
            self.vocos = self.vocos.to(self.dtype)

        # Try to read expected n_mels from the checkpoint
        exp_n_mels = None
        for attr_path in ("config.n_mels", "n_mels"):
            cur = self.vocos
            try:
                for part in attr_path.split("."):
                    cur = getattr(cur, part)
                if isinstance(cur, int):
                    exp_n_mels = cur
                    break
            except Exception:
                pass
        self.expected_n_mels: Optional[int] = exp_n_mels
        self.common_mel_set = {80, 100, 128}

        self.params = dict(repo_or_path=repo_or_path, device=str(self.device),
                           assume_input=assume_input, target_sr=self.target_sr)

    def __repr__(self):
        kv = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({kv})"

    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be numpy array or torch tensor.")
        return x

    def _axis_guess(self, shape: tuple[int, ...]) -> tuple[int, int, Optional[int]]:
        """
        Returns (batch_dim, mel_dim, time_dim)
        For 2D: batch_dim=None
        """
        exp = self.expected_n_mels
        if len(shape) == 2:
            a, b = shape
            # Prefer explicit expected n_mels if available
            if exp is not None:
                if a == exp: return (None, 0, 1)
                if b == exp: return (None, 1, 0)
            # Fallback heuristics
            if a in self.common_mel_set and b not in self.common_mel_set:
                return (None, 0, 1)
            if b in self.common_mel_set and a not in self.common_mel_set:
                return (None, 1, 0)
            # If ambiguous, treat last dim as mel (common for (T, n_mels))
            return (None, 1, 0)
        elif len(shape) == 3:
            B, X, Y = shape
            if exp is not None:
                if X == exp: return (0, 1, 2)    # (B, n_mels, T)
                if Y == exp: return (0, 2, 1)    # (B, T, n_mels)
            if X in self.common_mel_set and Y not in self.common_mel_set:
                return (0, 1, 2)
            if Y in self.common_mel_set and X not in self.common_mel_set:
                return (0, 2, 1)
            # Ambiguous: prefer channel-last input (B, T, n_mels)
            return (0, 2, 1)
        else:
            raise ValueError("Expected 2D or 3D input for mel spectrogram.")

    def _prep_mel(self, spc: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Normalize to (B, n_mels, T) on the correct device/dtype.
        """
        x = self._to_tensor(spc)
        if x.dim() not in (2, 3):
            raise ValueError("Expected 2D or 3D tensor for mel spectrogram.")
        # Convert log-mel to linear mel energy if needed
        if self.assume_input == "log_mel":
            # safe pow on torch
            x = torch.pow(torch.tensor(10.0, dtype=x.dtype, device=x.device), x)

        if x.dim() == 2:
            _, mel_dim, time_dim = self._axis_guess(tuple(x.shape))
            if mel_dim == 0 and time_dim == 1:
                x = x[None, :, :]          # (1, n_mels, T)
            elif mel_dim == 1 and time_dim == 0:
                x = x.transpose(0, 1)[None, :, :]  # (1, n_mels, T)
            else:
                raise RuntimeError("Axis guess failed for 2D input.")
        else:
            batch_dim, mel_dim, time_dim = self._axis_guess(tuple(x.shape))
            if (batch_dim, mel_dim, time_dim) == (0, 1, 2):
                pass                           # (B, n_mels, T)
            elif (batch_dim, mel_dim, time_dim) == (0, 2, 1):
                x = x.transpose(1, 2)          # (B, n_mels, T)
            else:
                raise RuntimeError("Axis guess failed for 3D input.")

        return x.to(device=self.device, dtype=torch.float32 if self.dtype is None else self.dtype)

    @torch.no_grad()
    def __call__(self, spc: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        mel = self._prep_mel(spc)       # (B, n_mels, T)
        wav = self.vocos.decode(mel)    # (B, T_wav)
        return wav[0] if wav.shape[0] == 1 else wav
