#!/usr/bin/env python3
import os
import json
import shutil

# =========================
# CONFIGURATION
# =========================

BASE_DIR = "dump/raw"
FULL_SPLIT = "overall_dev"   # source of full utterance labels
PREFIX_SPLITS = ["prefix_train", "prefix_valid", "prefix_test"]

OUTPUT_SUFFIX = "_full"

REQUIRED_METRICS = ["utmos", "dns_overall", "dns_p808", "plcmos"]

# =========================
# LOAD FULL UTTERANCE LABELS
# =========================

full_metric_path = os.path.join(BASE_DIR, FULL_SPLIT, "metric.scp")

if not os.path.exists(full_metric_path):
    raise FileNotFoundError(f"Full metric file not found: {full_metric_path}")

print(f"\nLoading full utterance metrics from: {full_metric_path}")

full_metrics = {}
total_lines = 0
kept_lines = 0

with open(full_metric_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        total_lines += 1

        # handle space or tab separation
        if "\t" in line:
            key, js = line.split("\t", 1)
        else:
            key, js = line.split(" ", 1)

        try:
            data = json.loads(js)
        except:
            continue

        # Only keep required metrics if present
        if not all(m in data for m in REQUIRED_METRICS):
            continue

        full_metrics[key] = {
            "utmos_full": data["utmos"],
            "dns_overall_full": data["dns_overall"],
            "dns_p808_full": data["dns_p808"],
            "plcmos_full": data["plcmos"],
        }

        kept_lines += 1

print(f"Total full lines: {total_lines}")
print(f"Valid full entries kept: {kept_lines}")
print(f"Unique full IDs: {len(full_metrics)}")

# =========================
# PROCESS PREFIX SPLITS
# =========================

for split in PREFIX_SPLITS:

    in_dir = os.path.join(BASE_DIR, split)
    out_dir = os.path.join(BASE_DIR, split + OUTPUT_SUFFIX)

    print(f"\nProcessing split: {split}")
    print(f"Input dir: {in_dir}")
    print(f"Output dir: {out_dir}")

    if not os.path.exists(in_dir):
        print("Skipping (not found).")
        continue

    os.makedirs(out_dir, exist_ok=True)

    # Copy non-metric files
    for fname in ["wav.scp", "ref_wav.scp", "metric2id", "metric2type"]:
        src = os.path.join(in_dir, fname)
        dst = os.path.join(out_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)

    in_metric_path = os.path.join(in_dir, "metric.scp")
    out_metric_path = os.path.join(out_dir, "metric.scp")

    if not os.path.exists(in_metric_path):
        print("No metric.scp found, skipping.")
        continue

    total_prefix = 0
    attached_full = 0
    missing_full = 0

    with open(in_metric_path, "r") as fin, open(out_metric_path, "w") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            total_prefix += 1

            if "\t" not in line:
                continue

            key, js = line.split("\t", 1)

            try:
                chunk_metrics = json.loads(js)
            except:
                continue

            base_id = key.split("__")[0]

            if base_id in full_metrics:
                chunk_metrics.update(full_metrics[base_id])
                attached_full += 1
            else:
                missing_full += 1

            fout.write(key + "\t" + json.dumps(chunk_metrics) + "\n")

    print(f"Total prefix samples: {total_prefix}")
    print(f"Full labels attached: {attached_full}")
    print(f"Missing full labels: {missing_full}")
    print(f"Coverage: {attached_full / total_prefix:.4f}")

print("\nDone.")