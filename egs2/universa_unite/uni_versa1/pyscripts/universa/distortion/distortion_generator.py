#!/usr/bin/env python3
"""
distortion_generator.py

Reads dump/raw/overall_dev/wav.scp (Kaldi ark format), samples N utterances,
injects artificial distortions at a specified timestamp, and writes:
  - Distorted wav files as plain PCM16 .wav files
  - distortion_manifest.csv  (orig_utt_id ↔ dist_utt_id ↔ dist_wav_path)

Audio is read via kaldiio (standard in every ESPnet environment).
Relative ark paths in wav.scp are resolved against --recipe_dir.

Usage (run from anywhere; use absolute paths):
  python distortion_generator.py \
    --recipe_dir   /work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1 \
    --src_dump_dir dump/raw/overall_dev \
    --audio_out_dir dump/raw/distorted_audio/full \
    --manifest_out  exp/distortion_manifest.csv \
    --n_samples     100 \
    --distortion_time 1.5 \
    --distortions   white_noise packet_drop \
    --seed          42

All path arguments that are not absolute are resolved relative to --recipe_dir.
"""

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import kaldiio
import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SAMPLE_RATE = 16000   # fixed for this recipe (raw_fs16000)

# ---------------------------------------------------------------------------
# Distortion registry
# ---------------------------------------------------------------------------
# To add a new distortion:
#   1. Write a function: fn(audio, sr, **kwargs) -> np.ndarray
#   2. Decorate it with @register_distortion("name")
#   3. Pass --distortions name on the command line
# ---------------------------------------------------------------------------

DISTORTION_REGISTRY: Dict[str, Callable] = {}


def register_distortion(name: str):
    def decorator(fn: Callable) -> Callable:
        if name in DISTORTION_REGISTRY:
            raise ValueError(f"Distortion '{name}' already registered.")
        DISTORTION_REGISTRY[name] = fn
        return fn
    return decorator


@register_distortion("white_noise")
def white_noise_burst(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    noise_duration: float = 0.10,
    snr_db: float = 5.0,
    **_,
) -> np.ndarray:
    """Inject a 100 ms white-noise burst at start_sec (SNR-controlled)."""
    audio = audio.copy()
    s = int(start_sec * sr)
    e = min(s + int(noise_duration * sr), len(audio))
    if s >= len(audio):
        log.warning("white_noise: start_sec=%.2f beyond audio length; skipped.", start_sec)
        return audio
    segment = audio[s:e]
    sig_power = float(np.mean(segment ** 2)) + 1e-9
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    rng = np.random.default_rng()
    noise = rng.standard_normal(e - s).astype(np.float32) * float(np.sqrt(noise_power))
    audio[s:e] = segment + noise
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio /= peak
    return audio


@register_distortion("packet_drop")
def packet_drop(
    audio: np.ndarray,
    sr: int,
    start_sec: float,
    drop_duration: float = 0.20,
    **_,
) -> np.ndarray:
    """Zero out a 200 ms window at start_sec to simulate packet loss."""
    audio = audio.copy()
    s = int(start_sec * sr)
    e = min(s + int(drop_duration * sr), len(audio))
    if s >= len(audio):
        log.warning("packet_drop: start_sec=%.2f beyond audio length; skipped.", start_sec)
        return audio
    audio[s:e] = 0.0
    return audio


@register_distortion("clipping")
def hard_clipping(
    audio: np.ndarray,
    sr: int,
    clip_threshold: float = 0.3,
    **_,
) -> np.ndarray:
    """Hard-clip the entire signal to ±clip_threshold."""
    return np.clip(audio.copy(), -clip_threshold, clip_threshold)


# ---------------------------------------------------------------------------
# Kaldi ark / wav.scp helpers
# ---------------------------------------------------------------------------

def read_scp(path: Path) -> Dict[str, str]:
    """Parse a Kaldi scp file → {utt_id: value}."""
    data: Dict[str, str] = {}
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                log.warning("%s:%d  malformed line (skipped): %r", path, lineno, line)
                continue
            data[parts[0]] = parts[1]
    return data


def resolve_path(entry: str, recipe_dir: Path) -> str:
    """
    Make the path portion of a wav.scp entry absolute.

    Kaldi ark entries look like:  path/to/file.ark:offset
    Plain entries look like:      path/to/file.wav
    Both may be relative to the recipe root.
    """
    entry = entry.strip()
    if entry.endswith("|"):
        raise ValueError(
            f"Pipe commands in wav.scp are not supported: {entry!r}\n"
            "Pre-convert pipe entries to plain files before running this script."
        )
    if ":" in entry:
        # Kaldi ark:  path:offset
        idx = entry.rfind(":")
        ark_path = entry[:idx]
        offset = entry[idx:]          # includes the colon
        if not Path(ark_path).is_absolute():
            ark_path = str(recipe_dir / ark_path)
        return ark_path + offset
    else:
        if not Path(entry).is_absolute():
            return str(recipe_dir / entry)
        return entry


def load_audio(wav_entry: str, recipe_dir: Path) -> np.ndarray:
    """
    Load mono float32 audio from a wav.scp entry.
    Handles Kaldi ark format (path:offset) and plain wav files.
    Returns float32 array normalised to [-1, 1].
    """
    resolved = resolve_path(wav_entry, recipe_dir)

    if ":" in resolved:
        # Kaldi ark — wav arks return (sample_rate, ndarray); feature arks return ndarray
        raw = kaldiio.load_mat(resolved)
        if isinstance(raw, tuple):
            _, raw = raw
    else:
        raw, _ = sf.read(resolved, dtype="float32", always_2d=False)

    if raw is None:
        raise ValueError(f"kaldiio returned None for {resolved}")

    audio = np.array(raw, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)         # stereo → mono

    # ESPnet ark files sometimes store PCM16-range values; normalise.
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / 32768.0
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio /= peak

    return audio


# ---------------------------------------------------------------------------
# Distortion application
# ---------------------------------------------------------------------------

def apply_distortions(
    audio: np.ndarray,
    distortion_names: List[str],
    distortion_time: float,
    kwargs: dict,
) -> np.ndarray:
    """Apply registered distortions sequentially."""
    for name in distortion_names:
        fn = DISTORTION_REGISTRY[name]
        audio = fn(audio, SAMPLE_RATE, start_sec=distortion_time, **kwargs)
    return audio


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_utterances(
    wav_scp: Dict[str, str],
    recipe_dir: Path,
    n_samples: int,
    min_duration_sec: float,
    rng: random.Random,
) -> List[Tuple[str, str, np.ndarray, float]]:
    """
    Scan wav_scp, keep utterances >= min_duration_sec, sample n_samples.
    Returns [(utt_id, wav_entry, audio, duration), ...].
    """
    total = len(wav_scp)
    log.info("Scanning %d utterances (min duration=%.2f s) ...", total, min_duration_sec)
    candidates = []

    for i, (utt_id, wav_entry) in enumerate(wav_scp.items()):
        if (i + 1) % 1000 == 0:
            log.info("  Scanned %d / %d ...", i + 1, total)
        try:
            audio = load_audio(wav_entry, recipe_dir)
        except Exception as exc:
            log.warning("  Cannot load %s: %s", utt_id, exc)
            continue
        duration = len(audio) / SAMPLE_RATE
        if duration >= min_duration_sec:
            candidates.append((utt_id, wav_entry, audio, duration))

    log.info("%d / %d utterances pass duration filter.", len(candidates), total)

    if not candidates:
        log.error("No valid utterances found. Check --src_dump_dir and --distortion_time.")
        sys.exit(1)

    if len(candidates) < n_samples:
        log.warning(
            "Only %d valid utterances; reducing --n_samples to %d.",
            len(candidates), len(candidates),
        )
        n_samples = len(candidates)

    sampled = rng.sample(candidates, n_samples)
    log.info("Sampled %d utterances.", len(sampled))
    return sampled


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def abs_or_rel(value: str, recipe_dir: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return recipe_dir / p


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inject distortions into sampled ESPnet dump utterances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--recipe_dir", required=True,
                   help="Absolute path to the ESPnet recipe root "
                        "(e.g. /work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1)")
    p.add_argument("--src_dump_dir", default="dump/raw/overall_dev",
                   help="Dump dir to sample from (relative to recipe_dir or absolute).")
    p.add_argument("--audio_out_dir", default="dump/raw/distorted_audio/full",
                   help="Where distorted .wav files will be written.")
    p.add_argument("--manifest_out", default="exp/distortion_manifest.csv",
                   help="Output CSV manifest path.")
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--distortion_time", type=float, default=1.5,
                   help="Timestamp (s) at which distortions are injected.")
    p.add_argument("--distortions", nargs="+",
                   default=["white_noise", "packet_drop"],
                   choices=list(DISTORTION_REGISTRY),
                   help="Distortions to apply in order.")
    p.add_argument("--noise_duration", type=float, default=0.10,
                   help="white_noise: burst duration (s).")
    p.add_argument("--snr_db", type=float, default=5.0,
                   help="white_noise: SNR in dB (lower = louder).")
    p.add_argument("--drop_duration", type=float, default=0.20,
                   help="packet_drop: zero-fill duration (s).")
    p.add_argument("--clip_threshold", type=float, default=0.3,
                   help="clipping: amplitude threshold.")
    p.add_argument("--min_duration", type=float, default=None,
                   help="Minimum utterance length (s). Default: distortion_time + 0.5.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    recipe_dir = Path(args.recipe_dir)

    if not recipe_dir.is_dir():
        log.error("recipe_dir does not exist: %s", recipe_dir)
        sys.exit(1)

    src_dump  = abs_or_rel(args.src_dump_dir,  recipe_dir)
    audio_out = abs_or_rel(args.audio_out_dir, recipe_dir)
    manifest  = abs_or_rel(args.manifest_out,  recipe_dir)

    wav_scp_path = src_dump / "wav.scp"
    if not wav_scp_path.exists():
        log.error("wav.scp not found: %s", wav_scp_path)
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---- Load source -------------------------------------------------------
    log.info("Reading wav.scp: %s", wav_scp_path)
    wav_scp = read_scp(wav_scp_path)

    # ---- Sample ------------------------------------------------------------
    min_dur = args.min_duration if args.min_duration is not None \
              else args.distortion_time + 0.5
    rng = random.Random(args.seed)
    sampled = sample_utterances(wav_scp, recipe_dir, args.n_samples, min_dur, rng)

    # ---- Distortion kwargs -------------------------------------------------
    dist_kwargs = {
        "noise_duration":  args.noise_duration,
        "snr_db":          args.snr_db,
        "drop_duration":   args.drop_duration,
        "clip_threshold":  args.clip_threshold,
    }

    # ---- Process -----------------------------------------------------------
    audio_out.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    log.info("Injecting: %s at t=%.2f s", args.distortions, args.distortion_time)

    for utt_id, wav_entry, audio, duration in sampled:
        dist_id   = f"dist_{utt_id}"
        dist_path = audio_out / f"{dist_id}.wav"

        try:
            distorted = apply_distortions(
                audio, args.distortions, args.distortion_time, dist_kwargs
            )
            sf.write(str(dist_path), distorted, SAMPLE_RATE, subtype="PCM_16")
            status = "ok"
            log.info("  %-40s  %.2f s  →  %s", utt_id, duration, dist_path.name)
        except Exception as exc:
            log.error("  FAILED %s: %s", utt_id, exc)
            status = f"error: {exc}"
            dist_path = Path("FAILED")

        rows.append({
            "orig_utt_id":     utt_id,
            "dist_utt_id":     dist_id,
            "orig_wav_entry":  wav_entry,
            "dist_wav_path":   str(dist_path),
            "duration_sec":    f"{duration:.4f}",
            "distortion_time": f"{args.distortion_time:.4f}",
            "distortions":     "+".join(args.distortions),
            "sample_rate":     str(SAMPLE_RATE),
            "status":          status,
        })

    # ---- Write manifest ----------------------------------------------------
    with open(manifest, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    log.info("Done. %d / %d utterances OK.", ok, len(rows))
    log.info("Manifest: %s", manifest)
    log.info(
        "\nNext step:\n"
        "  python dataset_builder.py \\\n"
        "    --recipe_dir  %s \\\n"
        "    --manifest    %s \\\n"
        "    --prefix_lengths 2 4 6 8",
        recipe_dir, manifest,
    )


if __name__ == "__main__":
    main()
