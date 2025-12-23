#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import torch
import torchaudio
from kaldiio import ReadHelper

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_wav_scp", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--chunk_s", type=float, default=2.0)
    ap.add_argument("--hop_s", type=float, default=2.0)
    ap.add_argument("--max_utts", type=int, default=0, help="0=all")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    wav_out = out_dir / "wavs"
    wav_out.mkdir(parents=True, exist_ok=True)

    chunk_wav_scp = out_dir / "wav.scp"
    chunk_map = out_dir / "chunk_map.jsonl"

    n = 0
    with open(chunk_wav_scp, "w") as f_scp, open(chunk_map, "w") as f_map:
        with ReadHelper("scp:" + str(args.in_wav_scp)) as reader:
            for utt, (fs, wav_np) in reader:
                n += 1
                if args.max_utts > 0 and n > args.max_utts:
                    break

                wav = torch.from_numpy(wav_np).float()
                T = wav.numel()
                chunk_len = int(round(args.chunk_s * fs))
                hop_len = int(round(args.hop_s * fs))

                # If too short to make even one full chunk, skip (tail discard policy)
                if T < chunk_len:
                    continue

                cidx = 0
                
                for start in range(0, T - chunk_len + 1, hop_len):
                    end = start + chunk_len  # full chunk only
                    chunk = wav[start:end]

                    # (optional safety; should never trigger now)
                    if chunk.numel() != chunk_len:
                        continue

                    chunk_id = f"{utt}__c{cidx:05d}"
                    out_path = wav_out / f"{chunk_id}.wav"

                    torchaudio.save(str(out_path), chunk.unsqueeze(0), sample_rate=int(fs))

                    f_scp.write(f"{chunk_id} {out_path}\n")
                    f_map.write(json.dumps({
                        "chunk_id": chunk_id,
                        "utt": utt,
                        "fs": int(fs),
                        "start_s": float(start / fs),
                        "end_s": float(end / fs),
                    }) + "\n")

                    cidx += 1

    print(f"Wrote: {chunk_wav_scp}")
    print(f"Wrote: {chunk_map}")

if __name__ == "__main__":
    main()
