# ar_universa_chunk_input.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from espnet2.universa.ar_universa.ar_universa_utils import (
    load_wav_scp,
    read_wav_scp_entry,
    safe_metric_list,
    infer_with_metrics,
    infer_chunks,
)

# If you already have a standard model loader in your repo,
# import and use it instead of re-writing.
# Example placeholder:
from espnet2.tasks.universa import UniversaTask  # adjust if your task name differs


def load_model(config: str, checkpoint: str, device: str = "cuda"):
    device = torch.device(device)
    model, _ = UniversaTask.build_model_from_file(config, checkpoint, device=device)
    model.to(device)
    model.eval()
    return model


def parse_metric_scp(metric_scp_path: str) -> Dict[str, dict]:
    out = {}
    with open(metric_scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt, js = line.split(maxsplit=1)
            out[utt] = json.loads(js)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--wav_scp", default="dump/raw/overall_base/wav.scp")
    p.add_argument("--metric_scp", default="dump/raw/overall_base/metric.scp")
    p.add_argument("--out_jsonl", default="exp/ar_universa_overall_base_pred.jsonl")
    p.add_argument("--device", default="cuda")

    p.add_argument("--do_chunks", action="store_true")
    p.add_argument("--chunk_s", type=float, default=2.0)
    p.add_argument("--hop_s", type=float, default=None)

    args = p.parse_args()

    model = load_model(args.config, args.checkpoint, device=args.device)

    wav_map = load_wav_scp(args.wav_scp)
    gt_metrics = Path(args.metric_scp).exists()
    gt_map = parse_metric_scp(args.metric_scp) if gt_metrics else {}

    # Pick a "general" preferred subset that you *know* exists
    preferred = [
        "nisqa_mos_pred",
        "utmos",
        "utmosv2",
        "dns_overall",
        "plcmos",
        "singmos",
        "srmr",
        "scoreq_nr",
    ]
    metric_list = safe_metric_list(model.metric2id, preferred)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as wf:
        for utt, value in wav_map.items():
            audio, fs = read_wav_scp_entry(value)

            pred_global = infer_with_metrics(
                model,
                audio,
                fs,
                metric_list=metric_list,
            )

            record = {
                "utt": utt,
                "fs": fs,
                "pred_global": {
                    k: (v.item() if torch.is_tensor(v) and v.numel() == 1 else v)
                    for k, v in pred_global.items()
                    if isinstance(k, str) and k not in ["encoded_feat"]
                },
            }

            if gt_metrics and utt in gt_map:
                record["metric_scp"] = gt_map[utt]

            if args.do_chunks:
                # If you want a specific chunk metric, list it here only if present
                chunk_metric_candidates = ["mos_chunk2s", "nisqa_mos_pred"]
                chunk_metric_list = [m for m in chunk_metric_candidates if m in model.metric2id]
                if len(chunk_metric_list) == 0:
                    chunk_metric_list = metric_list

                preds = infer_chunks(
                    model,
                    audio,
                    fs,
                    chunk_s=args.chunk_s,
                    hop_s=args.hop_s,
                    metric_list=chunk_metric_list,
                )

                # Store only lightweight fields
                record["pred_chunks"] = [
                    {
                        k: (v.item() if torch.is_tensor(v) and v.numel() == 1 else v)
                        for k, v in pd.items()
                        if isinstance(k, str) and k not in ["encoded_feat"]
                    }
                    for pd in preds
                ]

            wf.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
