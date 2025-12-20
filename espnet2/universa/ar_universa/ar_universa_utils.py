# Copyright 2024
# Apache 2.0
#
# Utilities for AR-UniVERSA chunk/prefix/future audio slicing.
#
# This module is intentionally lightweight and does NOT depend on
# ESPnet internals, so it can be used from both training-time
# preprocessor wrappers and inference scripts.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union, Dict, Any

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class ChunkSpec:
    """Metadata describing a chunk extracted from an utterance."""
    start_s: float
    end_s: float
    start_sample: int
    end_sample: int
    is_padded: bool = False


def _to_int(x: float) -> int:
    # Robust int conversion for seconds->samples
    return int(round(x))


def get_num_samples(audio: ArrayLike) -> int:
    """Return time dimension length assuming last dim is time."""
    return int(audio.shape[-1])


def ensure_1d_or_2d(audio: ArrayLike) -> None:
    """Basic sanity check. We accept:
    - (T,)
    - (C, T)
    - (B, T) or (B, C, T) if you pass batched audio into helpers.
    The slicing functions operate on the last dimension only.
    """
    if audio.ndim < 1:
        raise ValueError(f"audio must have at least 1 dim, got shape={audio.shape}")


def slice_audio(
    audio: ArrayLike,
    fs: int,
    start_s: float,
    end_s: float,
) -> ArrayLike:
    """Slice audio by seconds.

    Args:
        audio: np.ndarray or torch.Tensor, last dim is time.
        fs: sampling rate.
        start_s: start time in seconds (>=0).
        end_s: end time in seconds (> start_s).

    Returns:
        Sliced audio view/copy depending on backend.
    """
    ensure_1d_or_2d(audio)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if end_s <= start_s:
        raise ValueError(f"end_s must be > start_s, got start_s={start_s}, end_s={end_s}")

    T = get_num_samples(audio)
    s = max(0, _to_int(start_s * fs))
    e = min(T, _to_int(end_s * fs))

    if e <= s:
        # Return empty slice consistently
        return audio[..., 0:0]

    return audio[..., s:e]


def pad_to_length(
    audio: ArrayLike,
    target_len: int,
    pad_value: float = 0.0,
) -> Tuple[ArrayLike, bool]:
    """Pad last dimension to target_len."""
    ensure_1d_or_2d(audio)
    cur = get_num_samples(audio)
    if cur >= target_len:
        return audio, False

    pad_amount = target_len - cur
    if isinstance(audio, torch.Tensor):
        # F.pad expects (pad_left, pad_right) on last dim
        padded = torch.nn.functional.pad(audio, (0, pad_amount), value=pad_value)
        return padded, True
    else:
        # numpy
        pad_width = [(0, 0)] * audio.ndim
        pad_width[-1] = (0, pad_amount)
        padded = np.pad(audio, pad_width, mode="constant", constant_values=pad_value)
        return padded, True


def chunk_audio(
    audio: ArrayLike,
    fs: int,
    chunk_s: float = 2.0,
    hop_s: Optional[float] = None,
    keep_tail: bool = True,
):
    ensure_1d_or_2d(audio)
    if hop_s is None:
        hop_s = chunk_s

    T = get_num_samples(audio)
    chunk_len = _to_int(chunk_s * fs)
    hop_len = _to_int(hop_s * fs)

    chunks = []

    # 1. Full non-overlapping chunks from the start
    s = 0
    while s + chunk_len <= T:
        chunks.append(audio[..., s : s + chunk_len])
        s += hop_len

    # 2. Tail handling
    if keep_tail and s < T:
        if T <= chunk_len:
            # Very short utterance: just keep what we have
            if not chunks:
                chunks.append(audio[..., :T])
        else:
            # Long utterance: add a final full chunk aligned to the end
            chunks.append(audio[..., T - chunk_len : T])

    # Safety: always return at least one chunk
    if not chunks:
        chunks.append(audio)

    return chunks


def split_prefix_future(
    audio: ArrayLike,
    ratio: float = 0.5,
) -> Tuple[ArrayLike, ArrayLike]:
    """Split audio into prefix and future by ratio on last dimension.

    Args:
        audio: waveform
        ratio: prefix ratio in (0,1). 0.5 means half/half.

    Returns:
        (prefix, future)
    """
    ensure_1d_or_2d(audio)
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"ratio must be in (0,1), got {ratio}")

    T = get_num_samples(audio)
    if T == 0:
        return audio[..., 0:0], audio[..., 0:0]

    cut = int(T * ratio)
    cut = max(1, min(T - 1, cut))  # avoid empty halves for very short signals
    return audio[..., :cut], audio[..., cut:]


def seconds_to_samples(seconds: float, fs: int) -> int:
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    return _to_int(seconds * fs)


def samples_to_seconds(samples: int, fs: int) -> float:
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    return float(samples) / float(fs)


# ------------- Optional: small helper for your JSON output logic ------------- #

def build_mos_multiview_summary(
    mos_global: float,
    mos_chunks: Sequence[float],
    mos_future: Optional[float] = None,
    mos_prefix: Optional[float] = None,
) -> Dict[str, Any]:
    """Convenience helper to build your optional future/trend style outputs
    without introducing new metrics into metric2id.

    trend_future is computed as mos_future - mos_prefix if both provided.
    """
    out: Dict[str, Any] = {
        "mos_global": float(mos_global),
        "mos_chunk2s": [float(x) for x in mos_chunks],
    }
    if mos_future is not None:
        out["mos_future"] = float(mos_future)
    if mos_future is not None and mos_prefix is not None:
        out["trend_future"] = float(mos_future) - float(mos_prefix)
    return out
