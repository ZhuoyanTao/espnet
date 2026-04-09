#!/usr/bin/env python3
"""Create a MetricTokenizer tokens.json that includes both quality metrics and
turn-taking event categories (NA, BC, I, T, C).

The output format is the same as used by espnet2/universa/metric_tokenizer/:
    {
        "tokenizer": { <metric_name>: [thresholds or categories] },
        "VOCAB":     [ token strings in order ],
        "offset":    { <metric_name>: [start_idx_in_vocab, n_tokens_for_metric] }
    }

Quality-metric thresholds (500-percentile buckets) follow the existing ANCHOR
tokens.json convention.  Turn-taking is added as a categorical metric whose
5 classes map to:
    turn_taking@0 → NA   (nobody talking)
    turn_taking@1 → BC   (backchannel)
    turn_taking@2 → I    (interruption / overlap)
    turn_taking@3 → T    (turn change)
    turn_taking@4 → C    (continuation — same speaker)

Usage
-----
python create_turn_taking_token_list.py \\
    [--base-tokens PATH_TO_EXISTING_TOKENS_JSON] \\
    --output OUTPUT_TOKENS_JSON

If --base-tokens is given the existing quality metrics are read from it and
turn_taking is appended.  If omitted, a minimal quality-metric vocabulary is
generated from hard-coded ANCHOR percentile thresholds so the script is
self-contained.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Hard-coded ANCHOR quality-metric thresholds (500-bin percentile tokenisation)
# Each list has 499 threshold boundaries → 500 bins.
# These match the 'metric_500_percentile_overall_base_w-numerical' token list.
# ---------------------------------------------------------------------------

def _linspace(lo, hi, n):
    """Return n-1 interior breakpoints spanning [lo, hi] inclusive."""
    step = (hi - lo) / n
    return [round(lo + i * step, 6) for i in range(1, n)]


# 500 bins → 499 thresholds for each metric
_N_BINS = 500

QUALITY_METRICS = {
    # MOS-type metrics in [1, 5]
    "nisqa_mos_pred":   _linspace(1.0, 5.0, _N_BINS),
    "utmos":            _linspace(1.0, 5.0, _N_BINS),
    "utmosv2":          _linspace(1.0, 5.0, _N_BINS),
    # DNSMOS overall in [1, 5]
    "dns_overall":      _linspace(1.0, 5.0, _N_BINS),
    # PLCMOS in [1, 5]
    "plcmos":           _linspace(1.0, 5.0, _N_BINS),
}

# Turn-taking classes in LabelIndex order (C=0, NA=1, IN=2, BC=3, T=4)
# We store them in that canonical order so token indices match LabelIndex.
TURN_TAKING_CATEGORIES = ["C", "NA", "I", "T", "BC"]
# Index map:
#   turn_taking@0 → C  (Continuation)
#   turn_taking@1 → NA (Silence)
#   turn_taking@2 → I  (Interruption)
#   turn_taking@3 → T  (Turn change)
#   turn_taking@4 → BC (Backchannel)
# NOTE: LabelIndex in compute_turn_take_metrics uses C=0,NA=1,IN=2,BC=3,T=4.
# Our inference script maps predicted token index → LabelIndex correctly.


def build_token_list(base_tokens_path: Optional[str], output_path: str):
    """Build the combined tokens.json."""

    if base_tokens_path is not None:
        with open(base_tokens_path) as f:
            base = json.load(f)
        tokenizer_conf = base["tokenizer"]
        existing_vocab = list(base["VOCAB"])
        existing_offset = dict(base["offset"])
        print(f"Loaded {len(existing_vocab)} tokens from {base_tokens_path}")
    else:
        # Build from hard-coded ANCHOR thresholds
        tokenizer_conf = {}
        existing_vocab = []
        existing_offset = {}
        for metric_name, thresholds in QUALITY_METRICS.items():
            start = len(existing_vocab)
            existing_vocab.append(f"{metric_name}@meta_label")
            n_value_tokens = len(thresholds) + 1  # N thresholds → N+1 bins
            for i in range(n_value_tokens):
                existing_vocab.append(f"{metric_name}@{i}")
            existing_offset[metric_name] = [start, 1 + n_value_tokens]
            tokenizer_conf[metric_name] = thresholds
        print(f"Built {len(existing_vocab)} quality-metric tokens from built-in thresholds.")

    # ------------------------------------------------------------------
    # Append turn-taking tokens
    # ------------------------------------------------------------------
    if "turn_taking" in tokenizer_conf:
        print("Warning: 'turn_taking' already present in base tokens; skipping append.")
    else:
        tt_start = len(existing_vocab)
        existing_vocab.append("turn_taking@meta_label")
        n_classes = len(TURN_TAKING_CATEGORIES)
        for i in range(n_classes):
            existing_vocab.append(f"turn_taking@{i}")
        n_tt_tokens = 1 + n_classes  # meta_label + n_classes value tokens
        existing_offset["turn_taking"] = [tt_start, n_tt_tokens]
        tokenizer_conf["turn_taking"] = TURN_TAKING_CATEGORIES
        print(
            f"Appended turn_taking tokens: "
            f"turn_taking@meta_label + {n_classes} class tokens "
            f"(start={tt_start})."
        )

    combined = {
        "tokenizer": tokenizer_conf,
        "VOCAB": existing_vocab,
        "offset": existing_offset,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Wrote {len(existing_vocab)} total tokens to {out}")
    print("Metrics:", list(tokenizer_conf.keys()))
    print(f"Total vocab size (incl 4 special): {len(existing_vocab) + 4}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-tokens",
        default=None,
        help="Path to existing ANCHOR tokens.json to extend (optional).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the new tokens.json.",
    )
    args = parser.parse_args()
    build_token_list(args.base_tokens, args.output)


if __name__ == "__main__":
    main()
