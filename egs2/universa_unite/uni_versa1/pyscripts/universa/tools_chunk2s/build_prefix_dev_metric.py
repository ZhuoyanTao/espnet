#!/usr/bin/env python3
import json
import glob
import os
from collections import defaultdict

# ====== Configuration ======
root = "/work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1"

# 🔁 Use overall_dev instead of overall_base
full_metric_path = os.path.join(root, "dump/raw/overall_dev/metric.scp")

# 🔁 Different prefix folders
nisqa_prefix_glob = os.path.join(
    root,
    "prefix_debug/versa_label_gpu_nisqa_res/result/*.gpu.txt"
)

ut_prefix_glob = os.path.join(
    root,
    "prefix_debug/versa_label_cpu/result/*.result.cpu.txt"
)

# 🔁 Single output directory (no split)
out_dir = os.path.join(
    root,
    "dump/raw/prefix_overall_dev_all"
)

dummy_ref_path = os.path.join(root, "dump/raw/prefix_train/dummy.wav")
wav_path_template = os.path.join("prefix_debug/wavs", "{id}.wav")

os.makedirs(out_dir, exist_ok=True)
# ============================================


def load_full_metrics(path):
    full = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            key, j = line.split(" ", 1)
            full[key] = json.loads(j)
    return full


def load_jsonl_files(glob_pattern, key_field="key"):
    out = []
    for fname in sorted(glob.glob(glob_pattern)):
        with open(fname, "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    d = json.loads(ln)
                except json.JSONDecodeError:
                    continue

                if key_field not in d:
                    continue

                # 🔥 Remove ALL DNSMOS-related metrics from prefix
                keys_to_remove = []

                for k in d.keys():
                    if (
                        "dnsmos" in k.lower()
                        or k == "dns_overall"
                        or k == "dns_p808"
                    ):
                        keys_to_remove.append(k)

                for k in keys_to_remove:
                    del d[k]

                out.append(d)

    return out



def merge_prefix_with_full(prefix_entries, full_metrics):
    merged = {}

    # Step 1: group prefix lengths by full_id
    prefix_groups = defaultdict(list)

    for e in prefix_entries:
        prefix_key = e["key"]
        if "__p" in prefix_key:
            full_id, prefix_length = prefix_key.split("__p", 1)
            prefix_groups[full_id].append((prefix_key, prefix_length))
        else:
            prefix_groups[prefix_key].append((prefix_key, None))

    # Step 2: determine full prefix per group (largest length)
    full_prefix_map = {}

    for full_id, entries in prefix_groups.items():
        max_len = -1
        max_key = None

        for pk, plen in entries:
            if plen is not None:
                try:
                    val = int(plen)
                except:
                    continue
                if val > max_len:
                    max_len = val
                    max_key = pk

        full_prefix_map[full_id] = max_key

    # Step 3: merge
    for e in prefix_entries:
        prefix_key = e["key"]

        if "__p" in prefix_key:
            full_id, _ = prefix_key.split("__p", 1)
        else:
            full_id = prefix_key

        full_meta = full_metrics.get(full_id)
        merged_dict = dict(e)

        if full_meta:

            is_full_prefix = prefix_key == full_prefix_map.get(full_id)

            if is_full_prefix:
                # For FULL utterance:
                # overwrite/add full metrics WITHOUT _full suffix
                for k, v in full_meta.items():
                    merged_dict[k] = v

                # Also remove any accidental *_full keys if they exist
                for k in list(merged_dict.keys()):
                    if k.endswith("_full"):
                        del merged_dict[k]

            else:
                # For NON-full prefixes:
                # append *_full
                for k, v in full_meta.items():
                    merged_dict[f"{k}_full"] = v

        merged[prefix_key] = merged_dict

    return merged

# ================= RUN =================

print("Loading full dev metrics...")
full_metrics = load_full_metrics(full_metric_path)

print("Loading NISQA prefixes...")
nisqa_prefixes = load_jsonl_files(nisqa_prefix_glob)

print("Loading CPU prefixes...")
ut_prefixes = load_jsonl_files(ut_prefix_glob)

# Merge prefix metrics
prefix_by_key = {}

for d in nisqa_prefixes:
    prefix_by_key[d["key"]] = dict(d)

for d in ut_prefixes:
    k = d["key"]
    if k in prefix_by_key:
        prefix_by_key[k].update(d)
    else:
        prefix_by_key[k] = dict(d)

merged = merge_prefix_with_full(
    list(prefix_by_key.values()),
    full_metrics
)

# ================= WRITE =================

metric_scp_path = os.path.join(out_dir, "metric.scp")
ref_scp_path = os.path.join(out_dir, "ref_wav.scp")
wav_scp_path = os.path.join(out_dir, "wav.scp")

with open(metric_scp_path, "w", encoding="utf-8") as mf, \
     open(ref_scp_path, "w", encoding="utf-8") as rf, \
     open(wav_scp_path, "w", encoding="utf-8") as wf:

    for pk in sorted(merged.keys()):
        body = dict(merged[pk])
        del body["key"]

        mf.write(f"{pk} {json.dumps(body, ensure_ascii=False)}\n")
        rf.write(f"{pk} {dummy_ref_path}\n")
        wf.write(f"{pk} {wav_path_template.format(id=pk)}\n")

print("Dev dataset built successfully.")

# Copy metric schema from train
train_schema_dir = os.path.join(
    root,
    "dump/raw/prefix_overall_base_train"
)

for fname in ["metric2id", "metric2type"]:
    src = os.path.join(train_schema_dir, fname)
    dst = os.path.join(out_dir, fname)

    if os.path.exists(src):
        with open(src, "r") as sf, open(dst, "w") as df:
            df.write(sf.read())

print("metric2id and metric2type copied.")
print("All done.")