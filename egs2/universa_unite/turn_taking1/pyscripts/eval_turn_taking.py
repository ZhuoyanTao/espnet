#!/usr/bin/env python3
"""Evaluate turn-taking predictions against Switchboard reference labels.

Calls compute_turn_take_metrics (from egs2/swbd/slu1/pyscripts/utils/) to
produce the same Metrics A-E + ROC-AUC reported in Talking Turns (Arora et al.
ICLR 2025).

Usage
-----
python eval_turn_taking.py \\
    --hyp  path/to/decode_test/text \\
    --ref  path/to/Test_Two_Channel_Label_Mono.csv \\
    --output-dir path/to/eval_results

The slu1 pyscripts must be on PYTHONPATH (or pass --slu1-root).
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hyp",        required=True, help="Model hypothesis text file (from run_turn_taking_inference.py)")
    parser.add_argument("--ref",        required=True, help="Test_Two_Channel_Label_Mono.csv reference file")
    parser.add_argument("--output-dir", required=True, help="Directory to write evaluation results")
    parser.add_argument(
        "--slu1-root",
        default=None,
        help="Path to egs2/swbd/slu1 (adds pyscripts/utils to PYTHONPATH). "
             "Set this if compute_turn_take_metrics is not already importable.",
    )
    args = parser.parse_args()

    # Add slu1 pyscripts to path if needed
    if args.slu1_root is not None:
        slu1_root = Path(args.slu1_root)
        sys.path.insert(0, str(slu1_root / "pyscripts"))

    try:
        from utils.compute_turn_take_metrics import (
            ModelParam,
            ScoreResult,
            compute_turn_decisions,
            compute_turn_likelihoods,
        )
    except ImportError as e:
        print(
            f"Cannot import compute_turn_take_metrics: {e}\n"
            "Pass --slu1-root /path/to/egs2/swbd/slu1 or add its pyscripts/ to PYTHONPATH.",
            file=sys.stderr,
        )
        sys.exit(1)

    labels = ["C", "NA", "IN", "BC", "T"]

    hyp_arr = list(open(args.hyp))
    ref_arr = list(open(args.ref))

    # ref  → ground-truth turn-taking decisions (CSV format)
    # hyp  → model likelihood predictions (text format)
    true_dict, turn_dict = compute_turn_decisions(ref_arr)
    pred_dict = compute_turn_likelihoods(
        hyp_arr,
        ModelParam.min_start_time.value,
        ModelParam.chunk_length.value,
    )

    assert len(true_dict) > 0, "Reference dict is empty — check ref CSV path."
    assert len(pred_dict) > 0, "Prediction dict is empty — check hyp text path."

    scorer = ScoreResult(
        true_dict, pred_dict, turn_dict, labels, human_human=True
    )

    f1 = scorer.compute_F1()
    roc_auc = scorer.compute_roc_auc()

    print("\n=== Turn-Taking Evaluation Results ===")
    print(f"F1:      {f1}")
    print(f"ROC-AUC: {roc_auc}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "F1": float(f1) if hasattr(f1, "__float__") else f1,
        "ROC_AUC": float(roc_auc) if hasattr(roc_auc, "__float__") else roc_auc,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {out_dir}/results.json")


if __name__ == "__main__":
    main()
