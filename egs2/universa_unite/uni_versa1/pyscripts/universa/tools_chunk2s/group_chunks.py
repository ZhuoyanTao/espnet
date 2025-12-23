#!/usr/bin/env python3
import argparse, json
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric_scp", required=True)
    ap.add_argument("--chunk_map", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    # chunk_id -> meta
    meta = {}
    with open(args.chunk_map, "r") as f:
        for line in f:
            o = json.loads(line)
            meta[o["chunk_id"]] = o

    # utt -> list of (start_s, metrics_dict)
    per_utt = defaultdict(list)
    with open(args.metric_scp, "r") as f:
        for line in f:
            chunk_id, metrics_json = line.strip().split(maxsplit=1)
            if chunk_id not in meta:
                continue
            m = meta[chunk_id]
            metrics = json.loads(metrics_json)
            per_utt[m["utt"]].append((m["start_s"], metrics, m))

    with open(args.out_jsonl, "w") as out:
        for utt, items in per_utt.items():
            items.sort(key=lambda x: x[0])
            fs = items[0][2]["fs"]
            chunk_metrics = [it[1] for it in items]
            out.write(json.dumps({
                "utt": utt,
                "fs": fs,
                "num_chunks": len(items),
                "chunk_s": items[0][2]["end_s"] - items[0][2]["start_s"],
                "hop_s": (items[1][0] - items[0][0]) if len(items) > 1 else None,
                "chunk_metrics": chunk_metrics,
            }) + "\n")

    print(f"Wrote {args.out_jsonl}")

if __name__ == "__main__":
    main()
