#!/usr/bin/env python3
"""Chunk-level turn-taking inference with the quality-aware AR model.

For each utterance in wav.scp, slides a context window of `context_s` seconds
at a stride of `stride_s` seconds (default 0.04 s = 40 ms) to produce one
5-class probability vector per 40 ms frame.

Output format (one line per utterance, matching compute_turn_likelihoods):
    utt_id  p0,p1,p2,p3,p4  p0,p1,p2,p3,p4  ...

where each entry is a comma-separated 5-class probability vector in
LabelIndex order:
    index 0 → C   (Continuation)   [turn_taking@0]
    index 1 → NA  (Silence)        [turn_taking@1]
    index 2 → I   (Interruption)   [turn_taking@2]
    index 3 → T   (Turn change)    [turn_taking@3]
    index 4 → BC  (Backchannel)    [turn_taking@4]

Note: LabelIndex in compute_turn_take_metrics uses C=0, NA=1, IN=2, BC=3, T=4.
Our TT_CLASS_ORDER below must match that exact ordering.

Usage
-----
python run_turn_taking_inference.py \\
    --model-dir exp/tt_causal_wavlm \\
    --wavscp    data/test/wav.scp \\
    --output    decode_test/text \\
    [--context-s 4.0] [--stride-s 0.04] [--device cuda]
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile
import torch
import torch.nn.functional as F

from espnet2.tasks.universa import UniversaTask

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# LabelIndex order from compute_turn_take_metrics.py:
# C=0, NA=1, IN=2, BC=3, T=4
# Our token order (turn_taking@0…4): C, NA, I, T, BC
# Mapping: token_idx → LabelIndex position
#   turn_taking@0 (C)  → LabelIndex 0
#   turn_taking@1 (NA) → LabelIndex 1
#   turn_taking@2 (I)  → LabelIndex 2   (note: LabelIndex calls it IN)
#   turn_taking@3 (T)  → LabelIndex 4
#   turn_taking@4 (BC) → LabelIndex 3
TOKEN_TO_LABELINDEX = [0, 1, 2, 4, 3]   # length = n_classes = 5
N_CLASSES = 5
MIN_START_S = 0.20   # skip first 200 ms (Talking Turns warm-up convention)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_dir: Path, device: str):
    config_file = model_dir / "config.yaml"
    model_file  = model_dir / "valid.loss.best.pth"
    assert config_file.exists(), f"Missing {config_file}"
    assert model_file.exists(),  f"Missing {model_file}"

    model, _ = UniversaTask.build_model_from_file(
        config_file=str(config_file),
        model_file=str(model_file),
        device=device,
    )
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Turn-taking token indices
# ---------------------------------------------------------------------------

def get_tt_token_indices(tokenizer) -> Tuple[int, int]:
    """Return (meta_label_idx, first_value_idx) for the turn_taking metric."""
    meta_key = "turn_taking@meta_label"
    val0_key  = "turn_taking@0"
    if meta_key not in tokenizer.vocab_indices:
        raise RuntimeError(
            "turn_taking metric not in model vocabulary. "
            "Train with the correct token list."
        )
    return tokenizer.vocab_indices[meta_key], tokenizer.vocab_indices[val0_key]


# ---------------------------------------------------------------------------
# Per-window turn-taking probability extraction
# ---------------------------------------------------------------------------

def extract_tt_probs(
    model,
    audio_chunk: np.ndarray,
    tt_meta_idx: int,
    tt_val0_idx: int,
    device: str,
) -> np.ndarray:
    """Encode one audio window and return a (N_CLASSES,) probability array.

    Greedily decodes the AR metric sequence until the turn_taking@meta_label
    token appears, then reads the softmax distribution over the 5 value tokens.
    Probability array is in LabelIndex order (C, NA, IN, BC, T).
    """
    model_device = next(model.parameters()).device
    chunk_t = torch.tensor(audio_chunk, dtype=torch.float32).unsqueeze(0).to(model_device)
    chunk_l = torch.tensor([len(audio_chunk)], dtype=torch.long).to(model_device)

    with torch.no_grad():
        audio_enc, audio_enc_lengths = model.encode(chunk_t, chunk_l)

        sos_id = model.sos
        seq = torch.tensor([[sos_id]], dtype=torch.long, device=model_device)
        found = False

        for _ in range(50):  # safety cap at 25 metric pairs
            seq_l = torch.tensor([seq.shape[1]], device=model_device)
            dec_out, _ = model.decoder(audio_enc, audio_enc_lengths, seq, seq_l)
            next_tok = dec_out[0, -1, :].argmax().item()

            if next_tok == tt_meta_idx:
                # Next step: value distribution over the 5 TT classes
                seq_with_meta = torch.cat(
                    [seq, torch.tensor([[tt_meta_idx]], device=model_device)], dim=1
                )
                seq_l2 = torch.tensor([seq_with_meta.shape[1]], device=model_device)
                dec_out2, _ = model.decoder(
                    audio_enc, audio_enc_lengths, seq_with_meta, seq_l2
                )
                val_logits = dec_out2[0, -1, tt_val0_idx: tt_val0_idx + N_CLASSES]
                token_probs = F.softmax(val_logits.float(), dim=0).cpu().numpy()
                found = True
                break

            seq = torch.cat(
                [seq, torch.tensor([[next_tok]], device=model_device)], dim=1
            )
            if next_tok == model.eos:
                break

    if not found:
        token_probs = np.ones(N_CLASSES, dtype=np.float32) / N_CLASSES

    # Remap from token order to LabelIndex order
    label_probs = np.zeros(N_CLASSES, dtype=np.float32)
    for token_i, label_i in enumerate(TOKEN_TO_LABELINDEX):
        label_probs[label_i] = token_probs[token_i]

    return label_probs


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def read_audio(wav_cmd_or_path: str, sr: int = 16000) -> np.ndarray:
    """Load audio with retry on I/O error."""
    try:
        data, file_sr = soundfile.read(wav_cmd_or_path, dtype="float32")
    except Exception:
        time.sleep(5)
        data, file_sr = soundfile.read(wav_cmd_or_path, dtype="float32")

    if data.ndim > 1:
        data = data.mean(axis=1)
    if file_sr != sr:
        raise ValueError(f"Expected {sr} Hz, got {file_sr} Hz from {wav_cmd_or_path}")
    return data


# ---------------------------------------------------------------------------
# Per-utterance inference
# ---------------------------------------------------------------------------

def infer_utterance(
    model,
    audio: np.ndarray,
    sr: int,
    tt_meta_idx: int,
    tt_val0_idx: int,
    context_s: float,
    stride_s: float,
    device: str,
) -> List[np.ndarray]:
    """Return list of per-40ms-frame probability arrays for one utterance.

    Slides a context_s window at stride_s (typically 0.04 s = 40 ms).
    The first MIN_START_S seconds are skipped (Talking Turns convention).
    """
    context_samples = int(context_s * sr)
    stride_samples  = int(stride_s  * sr)
    min_start_sample = int(MIN_START_S * sr)

    all_probs: List[np.ndarray] = []
    frame_end = min_start_sample + stride_samples  # first frame ends here

    while frame_end <= len(audio):
        frame_start = max(0, frame_end - context_samples)
        chunk = audio[frame_start:frame_end]

        probs = extract_tt_probs(model, chunk, tt_meta_idx, tt_val0_idx, device)
        all_probs.append(probs)
        frame_end += stride_samples

    return all_probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",  required=True)
    parser.add_argument("--wavscp",     required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--context-s",  type=float, default=4.0,
                        help="Audio context window in seconds (default 4.0)")
    parser.add_argument("--stride-s",   type=float, default=0.04,
                        help="Inference stride in seconds (default 0.04 = 40 ms)")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output    = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s …", model_dir)
    model = load_model(model_dir, args.device)
    model.eval()

    tt_meta_idx, tt_val0_idx = get_tt_token_indices(model.metric_tokenizer)
    logger.info("turn_taking@meta_label=%d  turn_taking@0=%d", tt_meta_idx, tt_val0_idx)
    logger.info("Context: %.2fs, stride: %.3fs", args.context_s, args.stride_s)

    wavscp_entries = []
    with open(args.wavscp) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                wavscp_entries.append((parts[0], parts[1]))

    logger.info("Processing %d utterances …", len(wavscp_entries))

    with open(output, "w") as out_f:
        for utt_id, wav_path in wavscp_entries:
            try:
                audio = read_audio(wav_path)
            except Exception as e:
                logger.warning("Skipping %s: %s", utt_id, e)
                continue

            all_probs = infer_utterance(
                model=model,
                audio=audio,
                sr=16000,
                tt_meta_idx=tt_meta_idx,
                tt_val0_idx=tt_val0_idx,
                context_s=args.context_s,
                stride_s=args.stride_s,
                device=args.device,
            )

            if not all_probs:
                logger.warning("No frames for %s (audio too short?)", utt_id)
                continue

            probs_str = " ".join(
                ",".join(f"{p:.6f}" for p in frame_probs)
                for frame_probs in all_probs
            )
            out_f.write(f"{utt_id} {probs_str}\n")
            logger.info("%s: %d frames", utt_id, len(all_probs))

    logger.info("Written to %s", output)


if __name__ == "__main__":
    main()
