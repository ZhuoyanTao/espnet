import os
import csv
import json

prefixes = ["2", "4", "6", "8"]

CLEAN_BASE = "/work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1/dump/raw"
ANCHOR_BASE = "/work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1/exp/universa_train_aruniversa_prefix_full_raw_fs16000_defer_full_metatrue.bak.bak/inference_defer_full_metatrue_14epoch"
ARECHO_BASE = "/work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1/exp/universa_universa_ar_overall_base_token_wavlm_large/inference_defer_full_metafalse_valid.loss.best"

METRICS = [
    "dns_overall",
    "dns_p808",

    "utmos",
    "utmosv2",
    "plcmos",

    "nisqa_mos_pred",
    "nisqa_noi_pred",
    "nisqa_dis_pred",

    "se_si_snr",
    "se_sdr",

    "asr_match_error_rate",
    "speech_bert",

    "spk_similarity",
]

output_rows = []


def load_json_metrics(path):
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            utt, json_str = line.split(" ", 1)
            metrics = json.loads(json_str)

            entry = {}

            for m in METRICS:
                entry[m] = metrics.get(m)
                entry[m + "_full"] = metrics.get(m + "_full")

            data[utt] = entry

    return data


for p in prefixes:
    print(f"\nProcessing prefix {p}s")

    clean_path = f"{CLEAN_BASE}/prefix_overall_dev_all_{p}s_pred/metric.scp"
    anchor_path = f"{ANCHOR_BASE}/prefix_distorted_overall_dev_{p}s_pred.bak/metric.scp"
    arecho_path = f"{ARECHO_BASE}/prefix_distorted_overall_dev_{p}s_pred/metric.scp"

    clean = load_json_metrics(clean_path)
    anchor = load_json_metrics(anchor_path)
    arecho = load_json_metrics(arecho_path)

    for dist_utt in anchor:
        base_utt = dist_utt.replace("dist_", "")

        if base_utt not in clean:
            continue
        if dist_utt not in arecho:
            continue

        row = [base_utt, int(p)]

        for m in METRICS:

            # clean
            clean_chunk = clean[base_utt].get(m)
            clean_full = clean[base_utt].get(m + "_full")

            if clean_chunk is None:
                clean_chunk = clean_full

            # anchor
            anchor_chunk = anchor[dist_utt].get(m)
            anchor_full = anchor[dist_utt].get(m + "_full")

            # arecho
            arecho_chunk = arecho[dist_utt].get(m)
            arecho_full = arecho[dist_utt].get(m + "_full")

            # append raw values
            row.extend([
                clean_chunk,
                clean_full,

                anchor_chunk,
                anchor_full,

                arecho_chunk,
                arecho_full,
            ])

            # append deltas
            def safe_delta(a, b):
                if a is None or b is None:
                    return None
                return float(a) - float(b)

            row.extend([
                safe_delta(anchor_chunk, clean_chunk),
                safe_delta(anchor_full, clean_full),

                safe_delta(arecho_chunk, clean_chunk),
                safe_delta(arecho_full, clean_full),
            ])

        output_rows.append(row)


print(f"\nTotal aligned rows: {len(output_rows)}")


# Build header dynamically
header = ["utt_id", "prefix_seconds"]

for m in METRICS:
    header.extend([
        f"clean_{m}",
        f"clean_{m}_full",

        f"anchor_{m}",
        f"anchor_{m}_full",

        f"arecho_{m}",
        f"arecho_{m}_full",

        f"delta_anchor_vs_clean_{m}",
        f"delta_anchor_vs_clean_{m}_full",

        f"delta_arecho_vs_clean_{m}",
        f"delta_arecho_vs_clean_{m}_full",
    ])


with open("master_reaction_analysis_all_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(output_rows)

print("Done. Wrote master_reaction_analysis_all_metrics.csv")