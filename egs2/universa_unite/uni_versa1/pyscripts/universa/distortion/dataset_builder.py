#!/usr/bin/env python3
"""
dataset_builder.py

Builds ESPnet-compatible dump directories for the distortion experiment.
All directories are created under dump/raw/ inside the recipe.

Creates:
  dump/raw/distorted_overall_dev/
      wav.scp          → plain wav paths for full distorted utterances
      utt2spk          → self-referential (dist_fileid_X dist_fileid_X)
      spk2utt          → inverted utt2spk
      metric.scp       → copied from overall_dev, utt_ids remapped
      ref_wav.scp      → copied from overall_dev, utt_ids remapped
      feats_type       → "raw"
      utt2num_samples  → computed from distorted wav length
      text             → copied from overall_dev if present

  dump/raw/prefix_distorted_overall_dev_2s_pred/    ← and 4s, 6s, 8s
      wav.scp          → plain wav paths for truncated distorted audio
      utt2spk, spk2utt → self-referential, utt_id = dist_fileid_X__p002000
      metric.scp       → copied from prefix_overall_dev_all_2s_pred/metric.scp
      ref_wav.scp      → copied from prefix_overall_dev_all_2s_pred/ref_wav.scp
      feats_type, utt2num_samples, text

Utterance ID convention (mirrors existing datasets):
  Overall dev:      fileid_XXXXXX
  Prefix (2s):      fileid_XXXXXX__p002000     (ms zero-padded to 6 digits)
  Distorted full:   dist_fileid_XXXXXX
  Distorted prefix: dist_fileid_XXXXXX__p002000

Usage:
  python dataset_builder.py \
    --recipe_dir    /work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1 \
    --manifest      exp/distortion_manifest.csv \
    --prefix_lengths 2 4 6 8

All paths relative to recipe_dir unless absolute.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def abs_or_rel(value: str, recipe_dir: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else recipe_dir / p


def prefix_suffix(sec: float) -> str:
    """Convert prefix duration in seconds to the __pXXXXXX utt_id suffix."""
    ms = int(round(sec * 1000))
    return f"__p{ms:06d}"


# ---------------------------------------------------------------------------
# Kaldi scp I/O
# ---------------------------------------------------------------------------

def read_scp(path: Path) -> Dict[str, str]:
    """Parse a Kaldi scp file → {utt_id: value}. Missing file → empty dict."""
    if not path.exists():
        return {}
    data: Dict[str, str] = {}
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                log.warning("%s:%d skipped malformed line: %r", path, lineno, line)
                continue
            data[parts[0]] = parts[1]
    return data


def write_scp(path: Path, data: Dict[str, str]) -> None:
    """Write a sorted Kaldi scp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for k in sorted(data):
            fh.write(f"{k} {data[k]}\n")


def write_spk2utt(path: Path, utt2spk: Dict[str, str]) -> None:
    """Write spk2utt by inverting utt2spk."""
    spk2utt: Dict[str, List[str]] = {}
    for utt, spk in utt2spk.items():
        spk2utt.setdefault(spk, []).append(utt)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for spk in sorted(spk2utt):
            fh.write(f"{spk} {' '.join(sorted(spk2utt[spk]))}\n")


def write_feats_type(path: Path, ftype: str = "raw") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write(ftype + "\n")


# ---------------------------------------------------------------------------
# Manifest reader
# ---------------------------------------------------------------------------

def read_manifest(path: Path) -> List[Dict[str, str]]:
    """Read distortion_manifest.csv. Returns only rows with status == 'ok'."""
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status", "").strip() == "ok":
                rows.append(row)
    log.info("Manifest: %d valid rows from %s", len(rows), path)
    return rows


# ---------------------------------------------------------------------------
# Audio loading for distorted wavs (plain PCM16 files written by generator)
# ---------------------------------------------------------------------------

def load_wav(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


# ---------------------------------------------------------------------------
# Remap scp entries
# ---------------------------------------------------------------------------

def remap_scp(
    src: Dict[str, str],
    id_map: Dict[str, str],
) -> Dict[str, str]:
    """
    Given {old_utt_id: value} and {old_utt_id: new_utt_id},
    return {new_utt_id: value} for matched entries.
    """
    out = {}
    for old_id, value in src.items():
        if old_id in id_map:
            out[id_map[old_id]] = value
    return out


# ---------------------------------------------------------------------------
# Dataset writers
# ---------------------------------------------------------------------------

def write_data_dir(
    out_dir: Path,
    wav_scp:        Dict[str, str],
    utt2spk:        Dict[str, str],
    metric_scp:     Dict[str, str],
    ref_wav_scp:    Dict[str, str],
    utt2num_samples: Dict[str, str],
    text:           Optional[Dict[str, str]],
) -> None:
    """Write all required files for an ESPnet dump directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_scp(out_dir / "wav.scp",         wav_scp)
    write_scp(out_dir / "utt2spk",         utt2spk)
    write_spk2utt(out_dir / "spk2utt",     utt2spk)
    write_scp(out_dir / "utt2num_samples", utt2num_samples)
    write_feats_type(out_dir / "feats_type", "raw")
    if metric_scp:
        write_scp(out_dir / "metric.scp",  metric_scp)
    else:
        log.warning("  No metric.scp entries to write for %s", out_dir.name)
    if ref_wav_scp:
        write_scp(out_dir / "ref_wav.scp", ref_wav_scp)
    if text:
        write_scp(out_dir / "text",        text)
    log.info("  → %s  (%d utterances)", out_dir, len(wav_scp))


# ---------------------------------------------------------------------------
# Build full distorted dataset
# ---------------------------------------------------------------------------

def build_distorted_full(
    manifest_rows: List[Dict[str, str]],
    src_dump_dir: Path,
    out_dir: Path,
) -> None:
    """
    dump/raw/distorted_overall_dev/
    Utterance IDs: dist_fileid_X
    Audio: plain wav files already written by distortion_generator.py
    """
    log.info("=== Building: %s ===", out_dir.name)

    src_metric   = read_scp(src_dump_dir / "metric.scp")
    src_ref_wav  = read_scp(src_dump_dir / "ref_wav.scp")
    src_text     = read_scp(src_dump_dir / "text") if (src_dump_dir / "text").exists() else {}

    wav_scp:         Dict[str, str] = {}
    utt2spk:         Dict[str, str] = {}
    utt2num_samples: Dict[str, str] = {}
    metric_scp:      Dict[str, str] = {}
    ref_wav_scp:     Dict[str, str] = {}
    text:            Dict[str, str] = {}

    for row in manifest_rows:
        orig_id  = row["orig_utt_id"]
        dist_id  = row["dist_utt_id"]
        dist_wav = row["dist_wav_path"]

        # Audio
        try:
            audio, _ = load_wav(dist_wav)
        except Exception as exc:
            log.warning("  Cannot read %s: %s  (skipping)", dist_wav, exc)
            continue

        wav_scp[dist_id]         = dist_wav
        utt2spk[dist_id]         = dist_id          # self-referential
        utt2num_samples[dist_id] = str(len(audio))

        # Carry-over metadata with remapped key
        if orig_id in src_metric:
            metric_scp[dist_id] = src_metric[orig_id]
        if orig_id in src_ref_wav:
            ref_wav_scp[dist_id] = src_ref_wav[orig_id]
        if orig_id in src_text:
            text[dist_id] = src_text[orig_id]

    write_data_dir(
        out_dir, wav_scp, utt2spk,
        metric_scp, ref_wav_scp, utt2num_samples,
        text or None,
    )


# ---------------------------------------------------------------------------
# Build prefix distorted datasets
# ---------------------------------------------------------------------------

def build_distorted_prefix(
    manifest_rows: List[Dict[str, str]],
    recipe_dir: Path,
    src_prefix_dump_dir: Path,
    out_dir: Path,
    prefix_sec: float,
    prefix_audio_dir: Path,
) -> None:
    """
    dump/raw/prefix_distorted_overall_dev_{N}s_pred/
    Utterance IDs: dist_fileid_X__p{ms:06d}
    Audio: physically truncated distorted wav files.
    Metadata copied from src_prefix_dump_dir (existing clean prefix dump),
    with utt_id remapping fileid_X__p{ms:06d} → dist_fileid_X__p{ms:06d}.
    """
    log.info("=== Building: %s  (%.0f s prefix) ===", out_dir.name, prefix_sec)

    suffix = prefix_suffix(prefix_sec)
    n_samples_prefix = int(prefix_sec * SAMPLE_RATE)

    # Source metadata from the existing CLEAN prefix dump directory
    if not src_prefix_dump_dir.exists():
        log.warning(
            "  Source prefix dump dir not found: %s\n"
            "  metric.scp and ref_wav.scp will be empty for this prefix.\n"
            "  Run the pipeline on the clean prefix datasets first.", src_prefix_dump_dir
        )
    src_metric  = read_scp(src_prefix_dump_dir / "metric.scp")   if src_prefix_dump_dir.exists() else {}
    src_ref_wav = read_scp(src_prefix_dump_dir / "ref_wav.scp")  if src_prefix_dump_dir.exists() else {}
    src_text    = read_scp(src_prefix_dump_dir / "text")         \
                  if (src_prefix_dump_dir / "text").exists() else {}

    prefix_audio_dir.mkdir(parents=True, exist_ok=True)

    wav_scp:         Dict[str, str] = {}
    utt2spk:         Dict[str, str] = {}
    utt2num_samples: Dict[str, str] = {}
    metric_scp:      Dict[str, str] = {}
    ref_wav_scp:     Dict[str, str] = {}
    text_out:        Dict[str, str] = {}
    skipped = 0

    for row in manifest_rows:
        orig_id   = row["orig_utt_id"]
        dist_id   = row["dist_utt_id"]
        dist_wav  = row["dist_wav_path"]
        duration  = float(row["duration_sec"])

        # Skip utterances shorter than this prefix
        if duration < prefix_sec:
            log.debug("  %s (%.2f s) shorter than %.0f s prefix; skipping.", dist_id, duration, prefix_sec)
            skipped += 1
            continue

        # Load and truncate the distorted audio
        try:
            audio, _ = load_wav(dist_wav)
        except Exception as exc:
            log.warning("  Cannot read %s: %s  (skipping)", dist_wav, exc)
            skipped += 1
            continue

        prefix_audio = audio[:n_samples_prefix]
        prefix_utt_id = f"{dist_id}{suffix}"
        prefix_path   = prefix_audio_dir / f"{prefix_utt_id}.wav"
        sf.write(str(prefix_path), prefix_audio, SAMPLE_RATE, subtype="PCM_16")

        wav_scp[prefix_utt_id]         = str(prefix_path)
        utt2spk[prefix_utt_id]         = prefix_utt_id
        utt2num_samples[prefix_utt_id] = str(len(prefix_audio))

        # Metadata from the CLEAN prefix dump, remapping the utt_id key
        clean_prefix_id = f"{orig_id}{suffix}"
        if clean_prefix_id in src_metric:
            metric_scp[prefix_utt_id] = src_metric[clean_prefix_id]
        if clean_prefix_id in src_ref_wav:
            ref_wav_scp[prefix_utt_id] = src_ref_wav[clean_prefix_id]
        if clean_prefix_id in src_text:
            text_out[prefix_utt_id] = src_text[clean_prefix_id]

    if skipped > 0:
        log.info("  %d utterances skipped (too short) for %.0f s prefix.", skipped, prefix_sec)

    write_data_dir(
        out_dir, wav_scp, utt2spk,
        metric_scp, ref_wav_scp, utt2num_samples,
        text_out or None,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build ESPnet dump directories for distorted prefix datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--recipe_dir", required=True,
                   help="Absolute path to the ESPnet recipe root.")
    p.add_argument("--manifest", default="exp/distortion_manifest.csv",
                   help="CSV manifest written by distortion_generator.py.")
    p.add_argument("--src_overall_dev", default="dump/raw/overall_dev",
                   help="Source overall_dev dump dir (relative to recipe_dir or absolute).")
    p.add_argument("--src_prefix_pattern",
                   default="dump/raw/prefix_overall_dev_all_{sec}s_pred",
                   help="Pattern for existing CLEAN prefix dump dirs. {sec} is replaced "
                        "by the integer prefix length. Used to copy metric.scp / ref_wav.scp.")
    p.add_argument("--prefix_audio_base", default="dump/raw/distorted_audio",
                   help="Base directory for distorted prefix wav files.")
    p.add_argument("--prefix_lengths", nargs="+", type=float, default=[2.0, 4.0, 6.0, 8.0],
                   help="Prefix durations in seconds.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    recipe_dir = Path(args.recipe_dir)

    if not recipe_dir.is_dir():
        log.error("recipe_dir does not exist: %s", recipe_dir)
        sys.exit(1)

    manifest_path   = abs_or_rel(args.manifest,        recipe_dir)
    src_overall_dev = abs_or_rel(args.src_overall_dev, recipe_dir)
    prefix_audio_base = abs_or_rel(args.prefix_audio_base, recipe_dir)

    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)
    if not src_overall_dev.is_dir():
        log.error("src_overall_dev not found: %s", src_overall_dev)
        sys.exit(1)

    manifest_rows = read_manifest(manifest_path)
    if not manifest_rows:
        log.error("No valid rows in manifest. Aborting.")
        sys.exit(1)

    dump_raw = recipe_dir / "dump" / "raw"

    # -----------------------------------------------------------------------
    # 1. Full distorted dataset
    # -----------------------------------------------------------------------
    build_distorted_full(
        manifest_rows,
        src_dump_dir=src_overall_dev,
        out_dir=dump_raw / "distorted_overall_dev",
    )

    # -----------------------------------------------------------------------
    # 2. Prefix distorted datasets
    # -----------------------------------------------------------------------
    for sec in sorted(args.prefix_lengths):
        sec_int = int(sec)
        src_prefix_dump = abs_or_rel(
            args.src_prefix_pattern.format(sec=sec_int), recipe_dir
        )
        out_dir = dump_raw / f"prefix_distorted_overall_dev_{sec_int}s_pred"
        prefix_audio_dir = prefix_audio_base / f"prefix_{sec_int}s"

        build_distorted_prefix(
            manifest_rows,
            recipe_dir=recipe_dir,
            src_prefix_dump_dir=src_prefix_dump,
            out_dir=out_dir,
            prefix_sec=sec,
            prefix_audio_dir=prefix_audio_dir,
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    log.info("\n=== All datasets built. ===")
    log.info("Run inference with uni_versa.sh --test_sets:")
    for sec in sorted(args.prefix_lengths):
        log.info("  prefix_distorted_overall_dev_%ds_pred", int(sec))

    log.info(
        "\nExample:\n"
        "  cd %s\n"
        "  ./uni_versa.sh \\\n"
        "    --stage 9 --stop_stage 10 \\\n"
        "    --train_config conf/train_aruniversa_prefix_full.yaml \\\n"
        "    --universa_exp exp/universa_train_aruniversa_prefix_full_raw_fs16000"
        "_defer_full_metatrue.bak.bak.bak.bak \\\n"
        "    --test_sets 'prefix_distorted_overall_dev_2s_pred "
        "prefix_distorted_overall_dev_4s_pred "
        "prefix_distorted_overall_dev_6s_pred "
        "prefix_distorted_overall_dev_8s_pred' \\\n"
        "    --use_ref_wav true --use_ref_text false \\\n"
        "    --gpu_inference false --inference_nj 128 \\\n"
        "    --train_args '--defer_full_meta true' \\\n"
        "    --inference_args '--defer_full_meta true'",
        recipe_dir,
    )


if __name__ == "__main__":
    main()
