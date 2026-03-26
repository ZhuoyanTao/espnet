#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import shutil

# Metrics to convert from base name → *_chunk
METRICS_TO_RENAME = [
    "utmos",
    "dns_overall",
    "dns_p808",
    "plcmos",
]

def rename_metrics(input_path: Path):
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    output_path = input_path.with_suffix(".scp.new")
    backup_path = input_path.with_suffix(".scp.bak")

    print(f"Reading: {input_path}")
    print(f"Temporary new file: {output_path}")

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                key, json_str = line.split("\t", 1)
            except ValueError:
                print(f"Skipping malformed line: {line}")
                continue

            data = json.loads(json_str)

            new_data = {}
            for k, v in data.items():
                # Rename only exact base metrics
                if k in METRICS_TO_RENAME:
                    new_data[k + "_chunk"] = v
                else:
                    new_data[k] = v

            fout.write(f"{key}\t{json.dumps(new_data)}\n")

    # Backup original
    print(f"Backing up original to: {backup_path}")
    shutil.move(input_path, backup_path)

    # Replace original with new
    print(f"Replacing original with modified file.")
    shutil.move(output_path, input_path)

    print("Done successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python rename_prefix_metrics.py path/to/metric.scp")
        sys.exit(1)

    rename_metrics(Path(sys.argv[1]))