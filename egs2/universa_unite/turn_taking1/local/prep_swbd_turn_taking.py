#!/usr/bin/env python3
"""Prepare chunk-level ESPnet training data for the turn-taking ANCHOR model.

Each training example is a fixed-length audio context window with ONE
turn-taking label (the event at the window's last 40 ms frame).  During
inference the same window size is slid at 40 ms stride to get per-frame
predictions.

Reads
-----
  *  Speaker-invariant mono CSV from egs2/swbd/slu1/local/:
       create_switchboard_data_2channels_mono.py output
     Format: file_id, start_time, end_time, tt_label, prev_speaker
     e.g.  4001,0.20,0.24,C,NA
  *  A Switchboard wav.scp (mono) whose utterance keys are "sw0{file_id}".

Writes (per split: train / valid / test)
-----------------------------------------
  data/{split}/wav.scp      — chunk utterance-id → sox trim command
  data/{split}/metric.scp   — chunk utterance-id → JSON {"turn_taking": label}
  data/{split}/utt2spk      — chunk utterance-id → file-level speaker id

wav.scp format example:
  sw04001_000200_004200 sox /path/sw04001.wav -r 16000 -c 1 -t wav - trim 0.20 4.00 |

metric.scp format example:
  sw04001_000200_004200 {"turn_taking": "C"}

Chunk design
------------
  context_s : audio window fed to the model (default 4.0 s)
  stride_s  : window stride during TRAINING data creation (default 2.0 s)
              Use 0.04 s at inference for 40 ms frame-level predictions.
  chunk_s   : Switchboard label granularity (always 0.04 s)

The turn-taking label for a window is the label of the LAST 40 ms frame
inside the window (i.e., the most recent event the model should predict).

Label mapping (BC_1 / BC_2 edge-cases → BC)
  NA → NA
  BC / BC_1 / BC_2 → BC
  I  → I
  T  → T
  C  → C

Usage
-----
python prep_swbd_turn_taking.py \\
    --csv-dir   /path/to/egs2/swbd/slu1 \\
    --wavscp    /path/to/data/train_nodup/wav.scp \\
    --output-dir egs2/universa_unite/turn_taking1/data \\
    [--splits train valid test] \\
    [--context-s 4.0] [--stride-s 2.0]
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CHUNK_S = 0.04          # Switchboard label granularity (fixed)
MIN_AUDIO_S = 0.20      # VAP / Talking Turns warm-up period (skip first 200 ms)

SPLIT_CSV_CANDIDATES = {
    "train": [
        "Train_Two_Channel_Label_Mono_subsample.csv",
        "Train_Two_Channel_Label_Mono.csv",
    ],
    "valid": [
        "Val_Two_Channel_Label_Mono_subsample.csv",
        "Val_Two_Channel_Label_Mono.csv",
    ],
    "test": [
        "Test_Two_Channel_Label_Mono.csv",
    ],
}

LABEL_MAP = {
    "NA":   "NA",
    "BC":   "BC",
    "BC_1": "BC",
    "BC_2": "BC",
    "I":    "I",
    "T":    "T",
    "C":    "C",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_csv(csv_path: Path) -> Dict[str, List[Tuple[float, float, str]]]:
    """Return {file_id: [(start, end, label), ...]} sorted by start time."""
    data: Dict[str, List] = defaultdict(list)
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            file_id = parts[0].strip()
            try:
                start = float(parts[1])
                end   = float(parts[2])
            except ValueError:
                continue
            raw_label = parts[3].strip()
            label = LABEL_MAP.get(raw_label, "C")
            data[file_id].append((start, end, label))
    for fid in data:
        data[fid].sort(key=lambda x: x[0])
    return dict(data)


def parse_wavscp(wavscp_path: Path) -> Dict[str, str]:
    mapping = {}
    with open(wavscp_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


# ---------------------------------------------------------------------------
# Audio path extraction from wav.scp entry
# ---------------------------------------------------------------------------

def _extract_audio_path(wav_entry: str) -> Optional[str]:
    """Extract the bare file path from a wav.scp entry.

    Handles plain paths and common sox/ffmpeg pipe patterns.
    """
    entry = wav_entry.strip()
    # Plain path
    if not entry.endswith("|") and " " not in entry:
        return entry

    # sox pipe: "sox /path/file.sph -t wav - ..." or similar
    tokens = entry.split()
    for i, tok in enumerate(tokens):
        if tok in ("sox", "ffmpeg", "soxi"):
            if i + 1 < len(tokens):
                return tokens[i + 1]
    # Fallback: return the first token that looks like a path
    for tok in tokens:
        if tok.startswith("/") or tok.startswith("./"):
            return tok
    return None


def _make_sox_trim_cmd(wav_entry: str, t_start: float, duration: float) -> str:
    """Build a sox trim command for a [t_start, t_start+duration) window.

    If wav_entry is already a sox pipe we embed the source; otherwise we
    wrap the bare path in a sox call.
    """
    audio_path = _extract_audio_path(wav_entry)
    if audio_path is None:
        audio_path = wav_entry.split()[0]  # last resort

    return (
        f"sox {audio_path} -r 16000 -c 1 -t wav - "
        f"trim {t_start:.6f} {duration:.6f} |"
    )


# ---------------------------------------------------------------------------
# Chunk-level example creation
# ---------------------------------------------------------------------------

def make_chunks(
    file_id: str,
    labels: List[Tuple[float, float, str]],
    wav_entry: str,
    context_s: float,
    stride_s: float,
) -> List[Tuple[str, str, str]]:
    """Return list of (utt_id, sox_cmd, json_metric) for one conversation.

    Parameters
    ----------
    file_id  : 4-digit Switchboard ID, e.g. "4001"
    labels   : sorted list of (start, end, label) at 40 ms granularity
    wav_entry: raw wav.scp value for this conversation
    context_s: length of audio window (seconds)
    stride_s : hop between consecutive windows (seconds)
    """
    if not labels:
        return []

    conv_start = labels[0][0]   # first label start (≥ MIN_AUDIO_S)
    conv_end   = labels[-1][1]  # last label end

    # Build a lookup: end_time → label for quick access
    end_to_label: Dict[float, str] = {}
    for (s, e, lbl) in labels:
        end_to_label[round(e, 6)] = lbl

    examples = []
    t = conv_start
    while True:
        window_start = max(0.0, t - context_s)
        window_end   = t                       # exclusive end
        duration     = window_end - window_start

        # The "last 40 ms frame" in this window ends at t
        label_end_key = round(t, 6)
        if label_end_key not in end_to_label:
            # No label at exactly t — advance to next stride and retry
            t = round(t + stride_s, 6)
            if t > conv_end + 1e-6:
                break
            continue

        label = end_to_label[label_end_key]

        # Utterance ID: sw0{file_id}_{start_ms:07d}_{end_ms:07d}
        start_ms = int(round(window_start * 1000))
        end_ms   = int(round(window_end   * 1000))
        utt_id   = f"sw0{file_id}_{start_ms:07d}_{end_ms:07d}"

        sox_cmd = _make_sox_trim_cmd(wav_entry, window_start, duration)
        metric_json = json.dumps({"turn_taking": label})

        examples.append((utt_id, sox_cmd, metric_json))

        t = round(t + stride_s, 6)
        if t > conv_end + 1e-6:
            break

    return examples


# ---------------------------------------------------------------------------
# Split writing
# ---------------------------------------------------------------------------

def write_split(
    split: str,
    file_data: Dict[str, List],
    wavscp: Dict[str, str],
    output_dir: Path,
    context_s: float,
    stride_s: float,
):
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    wav_lines    = []
    metric_lines = []
    utt2spk_lines = []

    missing = skipped = 0
    for file_id in sorted(file_data.keys()):
        utt_id = "sw0" + file_id
        if utt_id not in wavscp:
            missing += 1
            continue

        labels   = file_data[file_id]
        wav_entry = wavscp[utt_id]

        examples = make_chunks(file_id, labels, wav_entry, context_s, stride_s)
        if not examples:
            skipped += 1
            continue

        for chunk_id, sox_cmd, metric_json in examples:
            wav_lines.append(f"{chunk_id} {sox_cmd}")
            metric_lines.append(f"{chunk_id} {metric_json}")
            utt2spk_lines.append(f"{chunk_id} {utt_id}")

    if missing:
        print(f"  [warn] {missing} conv IDs in CSV not found in wav.scp")
    if skipped:
        print(f"  [warn] {skipped} conversations skipped (no valid chunks)")

    def _write(fname, lines):
        with open(split_dir / fname, "w") as f:
            f.write("\n".join(lines) + "\n")

    _write("wav.scp", wav_lines)
    _write("metric.scp", metric_lines)
    _write("utt2spk", utt2spk_lines)

    print(
        f"  {split}: {len(wav_lines)} chunks from "
        f"{len(file_data) - missing} conversations "
        f"→ {split_dir}"
    )
    return len(wav_lines)


# ---------------------------------------------------------------------------
# metric2id / metric2type helper
# ---------------------------------------------------------------------------

def write_metric_meta(output_dir: Path, include_quality: bool = False):
    """Write metric2id and metric2type files."""
    metrics = []
    if include_quality:
        metrics += [
            ("nisqa_mos_pred", "categorical"),
            ("utmos",          "categorical"),
            ("utmosv2",        "categorical"),
            ("dns_overall",    "categorical"),
            ("plcmos",         "categorical"),
        ]
    metrics.append(("turn_taking", "categorical"))

    meta_dir = output_dir
    meta_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_dir / "metric2id", "w") as f:
        for name, _ in metrics:
            f.write(f"{name}\n")

    with open(meta_dir / "metric2type", "w") as f:
        for name, mtype in metrics:
            f.write(f"{name} {mtype}\n")

    print(f"Wrote metric2id and metric2type to {meta_dir}")
    print("  Metrics:", [n for n, _ in metrics])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir",    required=True, help="Directory with *_Mono*.csv files")
    parser.add_argument("--wavscp",     required=True, help="Switchboard wav.scp (mono)")
    parser.add_argument("--output-dir", required=True, help="Root data/ directory for output")
    parser.add_argument("--splits",     nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--context-s",  type=float, default=4.0,
                        help="Audio context window length in seconds (default 4.0)")
    parser.add_argument("--stride-s",   type=float, default=2.0,
                        help="Window stride during training data creation (default 2.0)")
    parser.add_argument("--include-quality-meta", action="store_true",
                        help="Also write quality metrics to metric2id/metric2type "
                             "(needed for joint quality+TT training config)")
    args = parser.parse_args()

    csv_dir    = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    wavscp     = parse_wavscp(Path(args.wavscp))
    print(f"Loaded {len(wavscp)} entries from {args.wavscp}")
    print(f"Context window: {args.context_s}s, training stride: {args.stride_s}s")

    total_chunks = 0
    for split in args.splits:
        candidates = SPLIT_CSV_CANDIDATES.get(split, [])
        csv_path = None
        for cand in candidates:
            p = csv_dir / cand
            if p.is_file():
                csv_path = p
                break
        if csv_path is None:
            print(f"[skip] No CSV found for split '{split}' in {csv_dir}")
            continue

        print(f"\nProcessing {split} from {csv_path.name} ...")
        file_data = parse_csv(csv_path)
        print(f"  {len(file_data)} conversations in CSV.")

        n = write_split(
            split, file_data, wavscp, output_dir,
            context_s=args.context_s,
            stride_s=args.stride_s,
        )
        total_chunks += n

    # Write metric meta files for training
    print(f"\nTotal chunks: {total_chunks}")
    write_metric_meta(
        output_dir / "local",
        include_quality=args.include_quality_meta,
    )
    print("Done.")


if __name__ == "__main__":
    main()
