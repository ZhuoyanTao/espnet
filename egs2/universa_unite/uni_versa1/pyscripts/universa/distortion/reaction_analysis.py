#!/usr/bin/env python3
"""
reaction_analysis.py

Loads clean and distorted inference outputs (metric.scp files in
exp/<universa_exp>/inference_<tag>/<test_set>/metric.scp), extracts
plcmos and utmos, and computes per-utterance reaction time.

Reaction time definition:
  Δ(utt, prefix) = |score_distorted(utt, prefix) - score_clean(utt, prefix)|
  reaction_prefix(utt) = first prefix (seconds) where Δ > threshold
                         None if Δ never exceeds threshold

metric.scp format (one line per utterance):
  utt_id {"plcmos": 3.45, "utmos": 2.89, ...}

Utterance ID conventions:
  Clean:     fileid_X__p002000   (in prefix_overall_dev_all_2s_pred/metric.scp)
  Distorted: dist_fileid_X__p002000  (in prefix_distorted_overall_dev_2s_pred/metric.scp)

Usage:
  python reaction_analysis.py \
    --recipe_dir    /work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1 \
    --manifest      exp/distortion_manifest.csv \
    --models \
        ANCHOR:exp/universa_train_aruniversa_prefix_full_raw_fs16000_defer_full_metatrue.bak.bak.bak.bak \
        ARECHO:exp/your_arecho_exp_dir \
    --inference_tag  inference_defer_full_metatrue_14epoch \
    --clean_test_set_pattern   "prefix_overall_dev_all_{sec}s_pred" \
    --dist_test_set_pattern    "prefix_distorted_overall_dev_{sec}s_pred" \
    --prefix_lengths 2 4 6 8 \
    --metrics        plcmos utmos \
    --threshold      0.3 \
    --out_csv        exp/distortion_results/reaction_time_per_utt.csv \
    --out_summary    exp/distortion_results/reaction_time_summary.csv

All paths relative to --recipe_dir unless absolute.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# {utt_id: {metric_name: float}}
ScoreMap = Dict[str, Dict[str, float]]


# ---------------------------------------------------------------------------
# metric.scp reader
# ---------------------------------------------------------------------------

def prefix_suffix(sec: float) -> str:
    """fileid_X → fileid_X__p002000 suffix."""
    ms = int(round(sec * 1000))
    return f"__p{ms:06d}"


def read_metric_scp(path: Path, metrics: List[str]) -> ScoreMap:
    """
    Read an ESPnet metric.scp file (JSON-per-line format).
    Returns {utt_id: {metric: value}} for the requested metrics.
    Only includes entries where ALL requested metrics are present.
    """
    scores: ScoreMap = {}
    if not path.exists():
        log.warning("metric.scp not found: %s", path)
        return scores

    missing_count = 0
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                log.warning("%s:%d  malformed line (skipped).", path, lineno)
                continue
            utt_id, json_str = parts
            try:
                blob = json.loads(json_str)
            except json.JSONDecodeError as exc:
                log.warning("%s:%d  JSON parse error: %s", path, lineno, exc)
                continue

            entry: Dict[str, float] = {}
            all_present = True
            for m in metrics:
                val = blob.get(m)
                if val is None:
                    all_present = False
                    break
                try:
                    entry[m] = float(val)
                except (TypeError, ValueError):
                    all_present = False
                    break

            if all_present:
                scores[utt_id] = entry
            else:
                missing_count += 1

    log.info("  Loaded %d scores from %s  (%d missing requested metrics)",
             len(scores), path, missing_count)
    return scores


# ---------------------------------------------------------------------------
# Manifest reader
# ---------------------------------------------------------------------------

def read_manifest(path: Path) -> List[Dict[str, str]]:
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status", "").strip() == "ok":
                rows.append(row)
    log.info("Manifest: %d valid rows from %s", len(rows), path)
    return rows


# ---------------------------------------------------------------------------
# Score loading
# ---------------------------------------------------------------------------

def load_scores_for_all_prefixes(
    exp_dir: Path,
    inference_tag: str,
    test_set_pattern: str,
    prefix_lengths: List[float],
    metrics: List[str],
) -> Dict[float, ScoreMap]:
    """
    Returns {prefix_sec: ScoreMap} for all requested prefixes.
    Path: exp_dir / inference_tag / test_set / metric.scp
    """
    all_scores: Dict[float, ScoreMap] = {}
    for sec in prefix_lengths:
        test_set   = test_set_pattern.format(sec=int(sec))
        score_path = exp_dir / inference_tag / test_set / "metric.scp"
        log.info("  Loading %s ...", score_path)
        all_scores[sec] = read_metric_scp(score_path, metrics)
    return all_scores


# ---------------------------------------------------------------------------
# Reaction time computation
# ---------------------------------------------------------------------------

def compute_utterance_record(
    orig_id: str,
    dist_id: str,
    prefix_lengths_sorted: List[float],
    clean_scores: Dict[float, ScoreMap],
    dist_scores:  Dict[float, ScoreMap],
    threshold: float,
    metrics: List[str],
) -> Dict:
    """
    Compute Δ at each prefix for each metric and find the reaction prefix.

    Clean utt_id key : {orig_id}__p{ms:06d}
    Distorted utt_id : {dist_id}__p{ms:06d}

    Reaction prefix = first prefix where ANY metric Δ > threshold.
    """
    record: Dict = {
        "orig_utt_id":        orig_id,
        "dist_utt_id":        dist_id,
        "reaction_prefix_sec": None,
        "ever_triggered":     False,
        "triggered_metric":   "",
    }
    max_deltas: Dict[str, float] = {m: 0.0 for m in metrics}

    for sec in prefix_lengths_sorted:
        suf = prefix_suffix(sec)
        clean_key = f"{orig_id}{suf}"
        dist_key  = f"{dist_id}{suf}"

        clean_entry = clean_scores.get(sec, {}).get(clean_key)
        dist_entry  = dist_scores.get(sec,  {}).get(dist_key)

        sec_int = int(sec)
        for m in metrics:
            c_val = clean_entry.get(m) if clean_entry else None
            d_val = dist_entry.get(m)  if dist_entry  else None
            record[f"clean_{m}_{sec_int}s"] = f"{c_val:.6f}" if c_val is not None else ""
            record[f"dist_{m}_{sec_int}s"]  = f"{d_val:.6f}" if d_val is not None else ""

            if c_val is not None and d_val is not None:
                delta = abs(d_val - c_val)
                record[f"delta_{m}_{sec_int}s"] = f"{delta:.6f}"
                max_deltas[m] = max(max_deltas[m], delta)

                if record["reaction_prefix_sec"] is None and delta > threshold:
                    record["reaction_prefix_sec"] = sec
                    record["ever_triggered"]       = True
                    record["triggered_metric"]     = m
            else:
                record[f"delta_{m}_{sec_int}s"] = ""

    for m in metrics:
        record[f"max_delta_{m}"] = f"{max_deltas[m]:.6f}"

    return record


def compute_summary(
    records: List[Dict],
    prefix_lengths: List[float],
    model_name: str,
    metric_names: List[str],
    threshold: float,
) -> Dict:
    """
    Aggregate reaction-time statistics across utterances.
    Utterances that never trigger are assigned max_prefix + 1 (conservative).
    """
    max_prefix = max(prefix_lengths)
    reaction_times = []
    never_triggered = 0

    for r in records:
        rt = r["reaction_prefix_sec"]
        if rt is None:
            reaction_times.append(max_prefix + 1.0)
            never_triggered += 1
        else:
            reaction_times.append(float(rt))

    arr = np.array(reaction_times)
    summary = {
        "model":               model_name,
        "metrics":             "+".join(metric_names),
        "threshold":           threshold,
        "n_utterances":        len(records),
        "n_triggered":         len(records) - never_triggered,
        "n_never_triggered":   never_triggered,
        "mean_reaction_sec":   float(np.mean(arr)),
        "median_reaction_sec": float(np.median(arr)),
        "std_reaction_sec":    float(np.std(arr)),
        "min_reaction_sec":    float(np.min(arr)),
        "max_reaction_sec":    float(np.max(arr)),
    }
    for sec in prefix_lengths:
        summary[f"pct_react_by_{int(sec)}s"] = float(np.mean(arr <= sec))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def abs_or_rel(value: str, recipe_dir: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else recipe_dir / p


def parse_models(model_args: List[str], recipe_dir: Path) -> List[Tuple[str, Path]]:
    """Parse ['NAME:path', ...] → [(name, abs_path), ...]."""
    result = []
    for entry in model_args:
        if ":" not in entry:
            raise ValueError(f"--models entries must be NAME:EXP_DIR, got: {entry!r}")
        name, exp_path = entry.split(":", 1)
        result.append((name.strip(), abs_or_rel(exp_path.strip(), recipe_dir)))
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute distortion reaction time from ESPnet metric.scp predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--recipe_dir", required=True,
                   help="Absolute path to the ESPnet recipe root.")
    p.add_argument("--manifest", default="exp/distortion_manifest.csv",
                   help="CSV manifest from distortion_generator.py.")
    p.add_argument("--models", nargs="+", required=True, metavar="NAME:EXP_DIR",
                   help="One or more models as NAME:exp/path. "
                        "Example: ANCHOR:exp/universa_train_..._full ...")
    p.add_argument("--inference_tag", required=True,
                   help="Inference tag (subdirectory name under exp/<model>/). "
                        "Example: inference_defer_full_metatrue_14epoch")
    p.add_argument("--clean_test_set_pattern",
                   default="prefix_overall_dev_all_{sec}s_pred",
                   help="Pattern for CLEAN test set names. {sec} → integer prefix seconds.")
    p.add_argument("--dist_test_set_pattern",
                   default="prefix_distorted_overall_dev_{sec}s_pred",
                   help="Pattern for DISTORTED test set names.")
    p.add_argument("--prefix_lengths", nargs="+", type=float,
                   default=[2.0, 4.0, 6.0, 8.0],
                   help="Prefix lengths in seconds (must match existing inference outputs).")
    p.add_argument("--metrics", nargs="+", default=["plcmos", "utmos"],
                   help="Metric keys to extract from JSON (lowercase, as in metric.scp).")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Δ threshold above which a reaction is declared.")
    p.add_argument("--out_csv", default="exp/distortion_results/reaction_time_per_utt.csv",
                   help="Per-utterance results CSV.")
    p.add_argument("--out_summary", default="exp/distortion_results/reaction_time_summary.csv",
                   help="Per-model summary CSV.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    recipe_dir    = Path(args.recipe_dir)
    manifest_path = abs_or_rel(args.manifest, recipe_dir)
    out_csv       = abs_or_rel(args.out_csv,     recipe_dir)
    out_summary   = abs_or_rel(args.out_summary, recipe_dir)

    if not recipe_dir.is_dir():
        log.error("recipe_dir not found: %s", recipe_dir)
        sys.exit(1)
    if not manifest_path.exists():
        log.error("Manifest not found: %s", manifest_path)
        sys.exit(1)

    manifest_rows  = read_manifest(manifest_path)
    models         = parse_models(args.models, recipe_dir)
    prefix_lengths = sorted(args.prefix_lengths)
    metrics        = [m.lower() for m in args.metrics]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    all_records:  List[Dict] = []
    summary_rows: List[Dict] = []

    for model_name, exp_dir in models:
        log.info("\n=== Model: %s  (%s) ===", model_name, exp_dir)

        if not exp_dir.is_dir():
            log.error("  exp_dir not found: %s", exp_dir)
            continue

        # Load clean scores for all prefixes
        log.info("  Loading CLEAN scores ...")
        clean_scores = load_scores_for_all_prefixes(
            exp_dir, args.inference_tag,
            args.clean_test_set_pattern, prefix_lengths, metrics,
        )

        # Load distorted scores for all prefixes
        log.info("  Loading DISTORTED scores ...")
        dist_scores = load_scores_for_all_prefixes(
            exp_dir, args.inference_tag,
            args.dist_test_set_pattern, prefix_lengths, metrics,
        )

        # Per-utterance reaction time
        model_records: List[Dict] = []
        missing_pairs = 0

        for row in manifest_rows:
            orig_id = row["orig_utt_id"]
            dist_id = row["dist_utt_id"]

            # Quick check: does this utterance appear in at least one prefix?
            suf0 = prefix_suffix(prefix_lengths[0])
            if (f"{orig_id}{suf0}" not in clean_scores.get(prefix_lengths[0], {}) and
                    f"{dist_id}{suf0}" not in dist_scores.get(prefix_lengths[0], {})):
                missing_pairs += 1

            rec = compute_utterance_record(
                orig_id, dist_id,
                prefix_lengths, clean_scores, dist_scores,
                args.threshold, metrics,
            )
            rec["model"]     = model_name
            rec["threshold"] = args.threshold
            rec["distortions"] = row.get("distortions", "")
            rec["distortion_time"] = row.get("distortion_time", "")
            model_records.append(rec)

        if missing_pairs > 0:
            log.warning(
                "  %d / %d utterances had no score at the first prefix (%ds). "
                "Check that inference completed for all test sets.",
                missing_pairs, len(manifest_rows), int(prefix_lengths[0])
            )

        # Summary statistics
        stats = compute_summary(
            model_records, prefix_lengths, model_name, metrics, args.threshold
        )
        log.info(
            "  Mean reaction: %.2f s  |  Median: %.2f s  |  "
            "Triggered: %d / %d",
            stats["mean_reaction_sec"], stats["median_reaction_sec"],
            stats["n_triggered"], stats["n_utterances"],
        )
        summary_rows.append(stats)
        all_records.extend(model_records)

    # -----------------------------------------------------------------------
    # Write per-utterance CSV
    # -----------------------------------------------------------------------
    if all_records:
        # Collect field order: fixed fields first, then dynamic per-prefix fields
        fixed = [
            "model", "threshold", "orig_utt_id", "dist_utt_id",
            "distortions", "distortion_time",
            "reaction_prefix_sec", "ever_triggered", "triggered_metric",
        ]
        seen = set(fixed)
        dynamic: List[str] = []
        for sec in prefix_lengths:
            for m in metrics:
                for tmpl in [f"clean_{m}_{int(sec)}s", f"dist_{m}_{int(sec)}s", f"delta_{m}_{int(sec)}s"]:
                    if tmpl not in seen:
                        dynamic.append(tmpl)
                        seen.add(tmpl)
        for m in metrics:
            key = f"max_delta_{m}"
            if key not in seen:
                dynamic.append(key)
                seen.add(key)
        fieldnames = fixed + dynamic

        with open(out_csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_records)
        log.info("\nPer-utterance CSV: %s", out_csv)

    # -----------------------------------------------------------------------
    # Write summary CSV
    # -----------------------------------------------------------------------
    if summary_rows:
        fixed_s = [
            "model", "metrics", "threshold",
            "n_utterances", "n_triggered", "n_never_triggered",
            "mean_reaction_sec", "median_reaction_sec",
            "std_reaction_sec", "min_reaction_sec", "max_reaction_sec",
        ]
        dynamic_s = [k for k in summary_rows[0] if k not in set(fixed_s)]
        fieldnames_s = fixed_s + dynamic_s

        with open(out_summary, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames_s, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)
        log.info("Summary CSV: %s", out_summary)

    # -----------------------------------------------------------------------
    # Console comparison table
    # -----------------------------------------------------------------------
    if summary_rows:
        print(f"\n{'='*68}")
        print(f"  Metrics: {', '.join(metrics)}   Threshold: {args.threshold}")
        print(f"{'='*68}")
        hdr = f"  {'Model':<14}  {'Mean RT(s)':>10}  {'Median RT(s)':>12}  {'Triggered':>9}  {'N':>5}"
        print(hdr)
        print(f"  {'-'*62}")
        for s in summary_rows:
            print(
                f"  {s['model']:<14}  {s['mean_reaction_sec']:>10.3f}  "
                f"{s['median_reaction_sec']:>12.3f}  "
                f"{s['n_triggered']:>9}  {s['n_utterances']:>5}"
            )
        print(f"{'='*68}")

        for sec in prefix_lengths:
            print(f"\n  Cumulative reaction by {int(sec)}s:")
            for s in summary_rows:
                pct = s.get(f"pct_react_by_{int(sec)}s", 0.0)
                print(f"    {s['model']:<14}  {pct*100:5.1f}%")

        if len(summary_rows) == 2:
            a, b = summary_rows
            diff = a["mean_reaction_sec"] - b["mean_reaction_sec"]
            winner = a["model"] if diff < 0 else b["model"]
            print(f"\n  {winner} reacts {abs(diff):.3f} s earlier on average.\n")


if __name__ == "__main__":
    main()
