#!/usr/bin/env python3
import json
import random
from pathlib import Path

import numpy as np
import torch
from kaldiio import ReadHelper

# Your helper
from espnet2.universa.ar_universa.ar_universa_utils import chunk_audio

"""
What this test does:
1) Reads wav from dump/raw/overall_base/wav.scp (Kaldi ark refs)
2) Chunks each utterance into 2s windows (non-overlap by default)
3) Optionally reads metric.scp and prints a few keys (e.g., nisqa_mos_pred)
4) Prints stats for sanity
"""

def load_metric_dict(metric_scp: str):
    m = {}
    with open(metric_scp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: <utt> <json>
            try:
                utt, js = line.split(maxsplit=1)
                m[utt] = json.loads(js)
            except Exception:
                # skip malformed lines
                continue
    return m

def main():
    root = Path(__file__).resolve()
    # adjust this if you place the test elsewhere
    base = Path("/work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1")
    scp = str(base / "dump/raw/overall_base/wav.scp")
    metric_scp = str(base / "dump/raw/overall_base/metric.scp")

    assert Path(scp).is_file(), f"Missing: {scp}"

    metric_dict = None
    if Path(metric_scp).is_file():
        metric_dict = load_metric_dict(metric_scp)

    # sample a small set for fast sanity
    sample_n = 10
    utts = []

    # collect keys efficiently
    with open(scp, "r", encoding="utf-8") as f:
        for line in f:
            utt = line.split()[0]
            utts.append(utt)

    print("Total utts listed:", len(utts))
    random.shuffle(utts)
    utts = utts[:sample_n]

    total_chunks = 0
    chunk_lens = []

    targets = set(utts)
    processed = 0

    with ReadHelper("scp:" + scp) as reader:
        for utt, (fs, wav) in reader:
            if utt not in targets:
                continue

            # wav is 1D np.array
            wav_t = torch.from_numpy(wav).float().unsqueeze(0)  # (1, T)

            chunks = chunk_audio(wav_t, fs, chunk_s=2.0, hop_s=2.0)
            total_chunks += len(chunks)
            chunk_lens.extend([c.shape[-1] for c in chunks])

            print("=" * 60)
            print("utt:", utt)
            print("fs:", fs)
            print("wav samples:", wav.shape[0])
            print("num 2s chunks:", len(chunks))
            expected = int(np.ceil(wav.shape[0] / (2.0 * fs)))
            print("expected ~chunks (ceil):", expected)
            if len(chunks) > 0:
                print("first chunk shape:", tuple(chunks[0].shape))

            if metric_dict and utt in metric_dict:
                md = metric_dict[utt]
                keys = [
                    "nisqa_mos_pred",
                    "utmos",
                    "utmosv2",
                    "dns_overall",
                    "plcmos",
                    "real_language",
                    "language",
                ]
                show = {k: md[k] for k in keys if k in md}
                if show:
                    print("metrics subset:", show)

            processed += 1
            if processed >= len(targets):
                break


    print("\n" + "#" * 60)
    print("Sampled utts:", len(utts))
    print("Total chunks:", total_chunks)
    print("Found sampled utts in reader:", processed, "/", len(targets))


    if chunk_lens:
        print("Chunk len (samples) min/mean/max:",
              int(min(chunk_lens)),
              float(np.mean(chunk_lens)),
              int(max(chunk_lens)))
        print("Expected 2s @16k =", 32000)



if __name__ == "__main__":
    main()
