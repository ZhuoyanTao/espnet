#!/usr/bin/env python3
"""
build_prefix_metric_and_split.py

1) Loads full-utterance metrics (metric.scp) -> map full_id -> dict
2) Loads all GPU prefix result files from NISQA folder and UT/PLCMOS folder (JSONL lines)
3) For each prefix entry:
     - determine full_id by splitting at "__p"
     - merge the prefix JSON with the full-utterance metrics, but append every key from full as "<key>_full"
4) Write a combined metric.scp (one JSON per line keyed by prefix id)
5) Group prefixes by full_id, shuffle groups (seeded), do 80/20 split by groups (so all prefixes of a full uttr go to same split)
6) Create train/valid dirs with metric.scp, metric2id, metric2type, ref_wav.scp, wav.scp
"""
import json
import glob
import os
import random
from collections import defaultdict

# ====== Configuration (edit if your paths differ) ======
root = "/work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1"
full_metric_path = os.path.join(root, "dump/raw/overall_base/metric.scp")
nisqa_prefix_glob = os.path.join(root, "prefix_overall_base/versa_label_gpu_nisqa_res/result/*.gpu.txt")
ut_prefix_glob = os.path.join(root, "prefix_overall_base/versa_label_gpu/result/*.result.gpu.txt")
prefix_wavs_dir = os.path.join(root, "prefix_overall_base/wavs")
out_parent = os.path.join(root, "dump/raw")
out_train_dir = os.path.join(out_parent, "prefix_overall_base_train")
out_valid_dir = os.path.join(out_parent, "prefix_overall_base_valid")
dummy_ref_path = os.path.join(out_parent, "prefix_train/dummy.wav")   # per your sample
wav_path_template = os.path.join("prefix_debug/wavs", "{id}.wav")    # what goes into wav.scp (relative path)
random_seed = 42
train_frac = 0.8
# =======================================================

os.makedirs(out_train_dir, exist_ok=True)
os.makedirs(out_valid_dir, exist_ok=True)

def load_full_metrics(path):
    full = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line=line.strip()
            if not line:
                continue
            # format: <key> <json>
            try:
                key, j = line.split(" ", 1)
            except ValueError:
                continue
            try:
                d = json.loads(j)
            except json.JSONDecodeError:
                # try to be robust by fixing trailing commas etc (not attempted here)
                raise
            full[key] = d
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

from collections import defaultdict

def merge_prefix_with_full(prefix_entries, full_metrics):
    merged = {}
    groups = defaultdict(list)

    # Step 1: group prefix lengths by full_id
    prefix_groups = defaultdict(list)

    for e in prefix_entries:
        prefix_key = e["key"]

        if "__p" in prefix_key:
            full_id, prefix_length = prefix_key.split("__p", 1)
            prefix_groups[full_id].append((prefix_key, prefix_length))
        else:
            full_id = prefix_key
            prefix_groups[full_id].append((prefix_key, None))

        groups[full_id].append(prefix_key)

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
                # FULL utterance:
                # overwrite with base metrics (no _full)
                for k, v in full_meta.items():
                    merged_dict[k] = v

                # remove accidental *_full keys if any
                for k in list(merged_dict.keys()):
                    if k.endswith("_full"):
                        del merged_dict[k]

            else:
                # NON-full prefix:
                for k, v in full_meta.items():
                    merged_dict[f"{k}_full"] = v

        merged[prefix_key] = merged_dict

    return merged, groups

# Load
print("Loading full metrics from:", full_metric_path)
full_metrics = load_full_metrics(full_metric_path)
print("Full metrics loaded:", len(full_metrics))

print("Loading NISQA prefix metrics from:", nisqa_prefix_glob)
nisqa_prefixes = load_jsonl_files(nisqa_prefix_glob)
print("NISQA prefix entries:", len(nisqa_prefixes))

print("Loading UT/PLCMOS prefix metrics from:", ut_prefix_glob)
ut_prefixes = load_jsonl_files(ut_prefix_glob)
print("UT prefix entries:", len(ut_prefixes))

# Combine prefix entries: some prefixes may appear in both lists (NISQA vs UT). Merge by key (prefix) preferring UT fields to override NISQA if conflict.
prefix_by_key = {}
for d in nisqa_prefixes:
    prefix_by_key[d["key"]] = dict(d)
for d in ut_prefixes:
    k = d["key"]
    if k in prefix_by_key:
        prefix_by_key[k].update(d)
    else:
        prefix_by_key[k] = dict(d)
print("Total unique prefix keys:", len(prefix_by_key))

all_prefix_entries = list(prefix_by_key.values())

# Merge with full metrics (append <metric>_full)
merged, groups = merge_prefix_with_full(all_prefix_entries, full_metrics)
print("Merged prefix entries:", len(merged))
print("Groups (full utterances):", len(groups))

# Write combined big metric.scp for reference
combined_metric_scp = os.path.join(out_parent, "prefix_overall_base_all_metric.scp")
with open(combined_metric_scp, "w", encoding="utf-8") as fh:
    for prefix_key, d in sorted(merged.items()):
        # remove the "key" field from the JSON body to match sample metric.scp output format
        body = dict(d)
        if "key" in body:
            del body["key"]
        fh.write(f"{prefix_key} {json.dumps(body, ensure_ascii=False)}\n")
print("Wrote combined metric.scp:", combined_metric_scp)

# Now do 80/20 split by full utterance groups
group_items = list(groups.items())  # (full_id, [prefix_keys...])
random.Random(random_seed).shuffle(group_items)
n_train_groups = int(len(group_items) * train_frac)
train_groups = group_items[:n_train_groups]
valid_groups = group_items[n_train_groups:]

def write_split(split_groups, out_dir):
    # collect prefix keys
    prefix_keys = []
    for full_id, keys in split_groups:
        prefix_keys.extend(keys)
    # metric.scp
    metric_scp_path = os.path.join(out_dir, "metric.scp")
    with open(metric_scp_path, "w", encoding="utf-8") as fh:
        for pk in sorted(prefix_keys):
            d = merged[pk]
            body = dict(d)
            if "key" in body:
                del body["key"]
            fh.write(f"{pk} {json.dumps(body, ensure_ascii=False)}\n")
    # ref_wav.scp -> every entry points to dummy_ref_path
    ref_scp_path = os.path.join(out_dir, "ref_wav.scp")
    with open(ref_scp_path, "w", encoding="utf-8") as fh:
        for pk in sorted(prefix_keys):
            fh.write(f"{pk} {dummy_ref_path}\n")
    # wav.scp -> point to prefix_wavs directory: use the path template (relative)
    wav_scp_path = os.path.join(out_dir, "wav.scp")
    with open(wav_scp_path, "w", encoding="utf-8") as fh:
        for pk in sorted(prefix_keys):
            wav_rel = wav_path_template.format(id=pk)
            fh.write(f"{pk} {wav_rel}\n")
    return len(prefix_keys)

train_count = write_split(train_groups, out_train_dir)
valid_count = write_split(valid_groups, out_valid_dir)
print(f"Wrote train: {train_count} prefix lines to {out_train_dir}")
print(f"Wrote valid: {valid_count} prefix lines to {out_valid_dir}")

# Create metric2id and metric2type files (use the lists provided in the prompt)
# Base metric names (EXACT ORDER YOU PROVIDED)
base_metric_names = [
"srmr","language","nisqa_mos_pred","nisqa_noi_pred","nisqa_dis_pred","nisqa_col_pred","nisqa_loud_pred",
"sheet_ssqa","utmos","utmosv2","dns_overall","dns_p808","plcmos","singmos","scoreq_nr",
"se_sdr","se_sar","se_si_snr","se_ci_sdr","pam_score","speaking_rate",
"audiobox_aesthetics_CE","audiobox_aesthetics_CU","audiobox_aesthetics_PC","audiobox_aesthetics_PQ",
"asvspoof_score","real_language","qwen_speaker_count","qwen_speaker_gender","qwen_speaker_age",
"qwen_speech_impairment","qwen_voice_pitch","qwen_pitch_range","qwen_voice_type","qwen_speech_volume_level",
"qwen_language","qwen_speech_register","qwen_vocabulary_complexity","qwen_speech_purpose",
"qwen_speech_emotion","qwen_speech_clarity","qwen_speech_rate","qwen_speaking_style",
"qwen_laughter_crying","qwen_speech_background_environment","qwen_recording_quality","qwen_channel_type",
"snr_simulation","rir_room_size","nomad","emotion_similarity","noresqa_score","speech_bert","speech_bleu",
"speech_token_distance","scoreq_ref","asr_match_error_rate","ref_text_length","pred_text_length",
"spk_similarity","rt60","visqol","pysepm_fwsegsnr","pysepm_llr","pysepm_wss","pysepm_cd","pysepm_c_sig",
"pysepm_c_bak","pysepm_c_ovl","pysepm_csii_high","pysepm_csii_mid","pysepm_csii_low","pysepm_ncm",
"mcd","f0rmse","f0corr","pesq","stoi","sdr","sar","si_snr","ci_sdr","nisqa_real_mos","wer","cer",
"urgent_mos","voicemos_real_mos"
]

# Append _full metrics
metric_names = base_metric_names + [m + "_full" for m in base_metric_names]

# Start everything numerical
metric_types = {name: "numerical" for name in metric_names}

# Categorical base metrics
categorical_base = {
"language","real_language","qwen_speaker_gender","qwen_speaker_age",
"qwen_speech_impairment","qwen_voice_pitch","qwen_pitch_range",
"qwen_voice_type","qwen_speech_volume_level","qwen_language",
"qwen_speech_register","qwen_vocabulary_complexity","qwen_speech_purpose",
"qwen_speech_emotion","qwen_speech_clarity","qwen_speaking_style",
"qwen_laughter_crying","qwen_speech_background_environment",
"qwen_recording_quality","qwen_channel_type","rir_room_size"
}

# Special case: qwen_speech_rate is numerical (as you specified)
if "qwen_speech_rate" in categorical_base:
    categorical_base.remove("qwen_speech_rate")

# Mark base categoricals
for c in categorical_base:
    metric_types[c] = "categorical"

# Mark _full categoricals
for c in categorical_base:
    metric_types[c + "_full"] = "categorical"

    
# write metric2id (one per line)
for out_dir in (out_train_dir, out_valid_dir):
    metric2id_path = os.path.join(out_dir, "metric2id")
    metric2type_path = os.path.join(out_dir, "metric2type")
    with open(metric2id_path, "w", encoding="utf-8") as fh:
        for name in metric_names:
            fh.write(f"{name}\n")
    with open(metric2type_path, "w", encoding="utf-8") as fh:
        for name in metric_names:
            t = metric_types.get(name, "numerical")
            fh.write(f"{name} {t}\n")
print("Wrote metric2id and metric2type into train/valid dirs")

print("All done.")