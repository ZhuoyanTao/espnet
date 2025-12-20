#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd


def load_utt_result_json(path: Path) -> pd.DataFrame:
    """Load utt_result.json produced by universa_eval.py (utt-level)."""
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Typical keys: "utt", "metrics": {...}, etc.
            utt = d["utt"]
            metrics = d.get("metrics", d.get("pred_metrics", {}))

            # Try to pull nisqa-related keys robustly
            # You can tweak these names once you see the exact structure
            cand_keys = [
                "nisqa_mos",
                "nisqa_mos_pred",
                "nisqa_mos_score",
            ]
            mos_pred = None
            for k in cand_keys:
                if k in metrics:
                    mos_pred = metrics[k]
                    break

            rows.append({"utt": utt, "nisqa_mos_pred": mos_pred})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Merge Universa NISQA MOS predictions with ground-truth NISQA MOS CSV "
                    "and compute simple stats."
    )
    parser.add_argument(
        "--utt-result",
        type=Path,
        required=True,
        help="Path to utt_result.json from universa_eval.py (utt-level).",
    )
    parser.add_argument(
        "--gt-csv",
        type=Path,
        required=True,
        help="Ground-truth NISQA MOS CSV, with at least columns: 'file' and 'mos'.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Where to save merged CSV (utt, mos_gt, mos_pred).",
    )
    args = parser.parse_args()

    if not args.utt_result.is_file():
        raise FileNotFoundError(f"utt_result.json not found: {args.utt_result}")
    if not args.gt_csv.is_file():
        raise FileNotFoundError(f"Ground-truth CSV not found: {args.gt_csv}")

    # 1) load universa predictions
    df_pred = load_utt_result_json(args.utt_result)

    # 2) load NISQA ground truth
    df_gt = pd.read_csv(args.gt_csv)

    # Assume NISQA CSV has columns: 'file' and 'mos'
    # Adjust column names if needed.
    if "file" in df_gt.columns:
        df_gt = df_gt.rename(columns={"file": "utt"})
    if "MOS" in df_gt.columns and "mos" not in df_gt.columns:
        df_gt = df_gt.rename(columns={"MOS": "mos"})

    if "utt" not in df_gt.columns or "mos" not in df_gt.columns:
        raise ValueError(
            f"Expected gt-csv to have columns 'utt' and 'mos' (or 'file' and 'mos'/'MOS'). "
            f"Got columns: {list(df_gt.columns)}"
        )

    df_gt = df_gt[["utt", "mos"]].rename(columns={"mos": "nisqa_mos_gt"})

    # 3) merge
    df = df_gt.merge(df_pred, on="utt", how="inner")

    if df.empty:
        raise RuntimeError(
            "Merged DataFrame is empty. "
            "Check that 'utt' names in utt_result.json and the NISQA CSV match."
        )

    # 4) basic stats
    print("Head of merged df:")
    print(df.head())
    print()
    print("Describe:")
    print(df[["nisqa_mos_gt", "nisqa_mos_pred"]].describe())

    # Correlation
    try:
        corr = df[["nisqa_mos_gt", "nisqa_mos_pred"]].corr().iloc[0, 1]
        print()
        print(f"Pearson correlation (gt vs pred): {corr:.4f}")
    except Exception as e:
        print("Could not compute correlation:", e)

    # 5) save
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nMerged CSV written to: {args.out_csv}")


if __name__ == "__main__":
    main()
