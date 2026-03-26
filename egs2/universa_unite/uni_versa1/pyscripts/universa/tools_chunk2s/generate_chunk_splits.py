#!/usr/bin/env python3

import os
import json
import shutil

SRC_DIR = "/work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1/dump/raw/prefix_overall_dev_all"
BASE_OUT_DIR = "/work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1/dump/raw"

SPLITS = {
    "2s": "__p002000",
    "4s": "__p004000",
    "6s": "__p006000",
    "8s": "__p008000",
}


def filter_scp_by_suffix(input_path, output_path, suffix):
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            uttid = line.split()[0]
            if uttid.endswith(suffix):
                fout.write(line)


def filter_metric_scp(input_path, output_path, suffix):
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue

            uttid, json_str = line.strip().split(" ", 1)

            if not uttid.endswith(suffix):
                continue

            metrics = json.loads(json_str)

            # Remove *_full metrics
            filtered_metrics = {
                k: v for k, v in metrics.items() if not k.endswith("_full")
            }

            fout.write(f"{uttid} {json.dumps(filtered_metrics, ensure_ascii=False)}\n")


def filter_metric2(input_path, output_path):
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            key = line.split()[0]
            if not key.endswith("_full"):
                fout.write(line)


def copy_file(input_path, output_path):
    shutil.copyfile(input_path, output_path)


def main():
    for split_name, suffix in SPLITS.items():
        out_dir = os.path.join(BASE_OUT_DIR, f"prefix_overall_dev_all_{split_name}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"Generating {out_dir} ...")

        # wav.scp
        filter_scp_by_suffix(
            os.path.join(SRC_DIR, "wav.scp"),
            os.path.join(out_dir, "wav.scp"),
            suffix,
        )

        # ref_wav.scp
        filter_scp_by_suffix(
            os.path.join(SRC_DIR, "ref_wav.scp"),
            os.path.join(out_dir, "ref_wav.scp"),
            suffix,
        )

        # metric.scp
        filter_metric_scp(
            os.path.join(SRC_DIR, "metric.scp"),
            os.path.join(out_dir, "metric.scp"),
            suffix,
        )

        # metric2type
        filter_metric2(
            os.path.join(SRC_DIR, "metric2type"),
            os.path.join(out_dir, "metric2type"),
        )

        # metric2id
        filter_metric2(
            os.path.join(SRC_DIR, "metric2id"),
            os.path.join(out_dir, "metric2id"),
        )

        # feats_type (copy directly)
        copy_file(
            os.path.join(SRC_DIR, "feats_type"),
            os.path.join(out_dir, "feats_type"),
        )

    print("All splits generated successfully.")


if __name__ == "__main__":
    main()