#!/usr/bin/env python3
"""
generate_chunk_splits.py

Creates:
  - prefix_overall_dev_all_2s, _4s, _6s, _8s
  - prefix_overall_dev_all_2s_pred, _4s_pred, _6s_pred, _8s_pred

Source dir:
  /work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1/dump/raw/prefix_overall_dev_all

Output parent:
  /work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1/dump/raw
"""
import os
import json
import shutil
from pathlib import Path

SRC_DIR = Path("/work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1/dump/raw/prefix_overall_dev_all")
BASE_OUT_DIR = Path("/work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1/dump/raw")

SPLITS = {
    "2s": "__p002000",
    "4s": "__p004000",
    "6s": "__p006000",
    "8s": "__p008000",
}

FILES_TO_FILTER = ["wav.scp", "ref_wav.scp"]
METRIC_SCP = "metric.scp"
METRIC2TYPE = "metric2type"
METRIC2ID = "metric2id"
FEATS_TYPE = "feats_type"


def read_lines(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def write_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip("\n") + "\n")


def filter_scp_by_suffix(lines, suffix):
    out = []
    for ln in lines:
        if not ln.strip():
            continue
        uttid = ln.split()[0]
        if uttid.endswith(suffix):
            out.append(ln)
    return out


def filter_metric_scp_lines(lines, suffix, keep_full=False):
    """
    lines: list of 'uttid json'
    keep_full=False -> remove keys ending with _full
    keep_full=True  -> keep only keys ending with _full (EXCEPT key_full)
    """
    out = []
    for ln in lines:
        if not ln.strip():
            continue

        parts = ln.split(" ", 1)
        if len(parts) != 2:
            continue

        uttid, json_str = parts

        if not uttid.endswith(suffix):
            continue

        try:
            metrics = json.loads(json_str)
        except Exception:
            continue

        if keep_full:
            # Keep only *_full but exclude key_full
            filtered = {
                k: v
                for k, v in metrics.items()
                if k.endswith("_full") and k != "key_full"
            }
        else:
            # Remove *_full
            filtered = {
                k: v
                for k, v in metrics.items()
                if not k.endswith("_full")
            }

        if not filtered:
            continue

        out.append(f"{uttid} {json.dumps(filtered, ensure_ascii=False)}")

    return out


def filter_metric2_lines(lines, keep_full=False):
    """
    Filter metric2type / metric2id lines by key name suffix.
    keep_full=False -> keep keys NOT ending with _full
    keep_full=True  -> keep keys ending with _full
    """
    out = []
    for ln in lines:
        if not ln.strip():
            continue
        key = ln.split()[0]
        if keep_full:
            if key.endswith("_full"):
                out.append(ln)
        else:
            if not key.endswith("_full"):
                out.append(ln)
    return out


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        return True
    return False


def safe_read(path: Path):
    return read_lines(path) if path.exists() else []


def make_split(out_dir: Path, suffix: str, pred: bool):
    """
    pred=False -> create normal split (remove _full keys)
    pred=True  -> create _pred split (keep only _full keys)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) wav.scp and ref_wav.scp
    for fname in FILES_TO_FILTER:
        src_lines = safe_read(SRC_DIR / fname)
        filtered = filter_scp_by_suffix(src_lines, suffix)
        write_lines(out_dir / fname, filtered)

    # 2) metric.scp
    src_metric_lines = safe_read(SRC_DIR / METRIC_SCP)
    metric_lines_out = filter_metric_scp_lines(src_metric_lines, suffix, keep_full=pred)
    write_lines(out_dir / METRIC_SCP, metric_lines_out)

    # 3) metric2type & metric2id
    for fname in (METRIC2TYPE, METRIC2ID):
        src_lines = safe_read(SRC_DIR / fname)
        filtered = filter_metric2_lines(src_lines, keep_full=pred)
        write_lines(out_dir / fname, filtered)

    # 4) feats_type (copy unchanged if exists)
    copy_if_exists(SRC_DIR / FEATS_TYPE, out_dir / FEATS_TYPE)


def main():
    if not SRC_DIR.exists():
        print(f"ERROR: source dir does not exist: {SRC_DIR}")
        return

    for split_name, suffix in SPLITS.items():
        # normal split (remove _full)
        out_dir = BASE_OUT_DIR / f"prefix_overall_dev_all_{split_name}"
        print(f"Creating {out_dir}")
        make_split(out_dir, suffix, pred=False)

        # pred split (keep only _full)
        out_dir_pred = BASE_OUT_DIR / f"prefix_overall_dev_all_{split_name}_pred"
        print(f"Creating {out_dir_pred}")
        make_split(out_dir_pred, suffix, pred=True)

    print("Done.")


if __name__ == "__main__":
    main()