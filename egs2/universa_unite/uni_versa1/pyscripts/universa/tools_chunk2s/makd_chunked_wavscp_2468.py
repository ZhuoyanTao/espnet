#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import torch
import torchaudio
from kaldiio import ReadHelper

PREFIX_SCHEDULE = [2.0, 4.0, 6.0, 8.0]  # seconds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_wav_scp", required=True)
    ap.add_argument("--out_dir", required=True)
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
                
                dur_s = T / fs

                prefixes = []
                for p in PREFIX_SCHEDULE:
                    if p < dur_s:
                        prefixes.append(p)

                # always include full utterance
                prefixes.append(dur_s)

                for cidx, p in enumerate(prefixes):
                    end = int(round(p * fs))
                    chunk = wav[:end]

                    chunk_id = f"{utt}__p{int(p*1000):06d}"  # p in ms
                    out_path = wav_out / f"{chunk_id}.wav"

                    torchaudio.save(str(out_path), chunk.unsqueeze(0), sample_rate=int(fs))

                    f_scp.write(f"{chunk_id} {out_path}\n")
                    f_map.write(json.dumps({
                        "chunk_id": chunk_id,
                        "utt": utt,
                        "fs": int(fs),
                        "start_s": 0.0,
                        "end_s": float(p),
                        "is_full": p >= dur_s - 1e-3,
                    }) + "\n")

        print(f"Wrote: {chunk_wav_scp}")
        print(f"Wrote: {chunk_map}")

if __name__ == "__main__":
    main()
