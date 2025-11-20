#!/usr/bin/env python3
import argparse, csv, os, sys, glob, collections

# RTTM columns (Kaldi/standard):
# 0 SPEAKER, 1 rec_id, 2 <chan or 1>, 3 start_sec, 4 dur_sec, 5 <..>, 6 <..>, 7 spk, 8 <..>

def read_manifest(p):
    rows = []
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            split = r["split"].strip()
            rec_id = r["rec_id"].strip()
            wav = r["wav_path"].strip()
            rows.append((split, rec_id, wav))
    return rows

def iter_rttm_lines(rttm):
    if os.path.isdir(rttm):
        files = sorted(glob.glob(os.path.join(rttm, "*.rttm")))
    else:
        files = [rttm]
    for fp in files:
        with open(fp) as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"): 
                    continue
                yield ln

def load_rttm(rttm):
    # dict[rec_id] -> list[(start, end, spk)]
    by_rec = collections.defaultdict(list)
    for ln in iter_rttm_lines(rttm):
        parts = ln.split()
        if len(parts) < 9 or parts[0] != "SPEAKER":
            continue
        rec = parts[1]
        start = float(parts[3]); dur = float(parts[4]); end = start + dur
        spk = parts[7]
        by_rec[rec].append((start, end, spk))
    return by_rec

def sec(x):
    return f"{x:.3f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with split,rec_id,wav_path")
    ap.add_argument("--rttm", required=True, help="RTTM file or directory")
    ap.add_argument("--fs", default="8k", choices=["8k","16k"])
    ap.add_argument("--num_spk", type=int, default=3, help="max speakers per recording")
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    rttm_by_rec = load_rttm(args.rttm)

    per_split = {"train":[], "dev":[], "test":[]}
    rec2wav = {}
    for split, rec, wav in rows:
        if split not in per_split:
            per_split[split] = []
        per_split[split].append(rec)
        rec2wav[rec] = wav

    for split, recs in per_split.items():
        if not recs:
            continue
        out = os.path.join(args.outdir, split)
        os.makedirs(out, exist_ok=True)

        # Collect first
        wav_pairs = set()          # (rec, wav)
        seg_rows  = {}             # utt -> (rec, st, en)
        u2s_rows  = {}             # utt -> spk
        rttm_rows = []             # list[str]

        for rec in sorted(set(recs)):
            if rec not in rec2wav:
                print(f"[WARN] {split}: {rec} missing in manifest", file=sys.stderr)
                continue
            wav = rec2wav[rec]
            if not os.path.isfile(wav):
                print(f"[WARN] WAV not found: {wav}", file=sys.stderr)
                continue
            wav_pairs.add((rec, wav))

            segs = rttm_by_rec.get(rec, [])
            if not segs:
                print(f"[WARN] no RTTM entries for {rec}", file=sys.stderr)
                continue

            # cap speakers by total duration
            dur_by_spk = collections.Counter()
            for st, en, sp in segs:
                dur_by_spk[sp] += (en - st)
            keep_spks = {sp for sp, _ in dur_by_spk.most_common(args.num_spk)}
            dropped = sorted(set(s for _, _, s in segs) - keep_spks)
            if dropped:
                print(f"[INFO] {rec}: dropping speakers {dropped} (limit {args.num_spk})", file=sys.stderr)

            for st, en, sp in segs:
                if sp not in keep_spks:
                    continue
                utt = f"{rec}_{int(round(st*1000)):07d}-{int(round(en*1000)):07d}_{sp}"
                # de-dup by utt id
                if utt in seg_rows:
                    continue
                seg_rows[utt] = (rec, sec(st), sec(en))
                u2s_rows[utt] = sp
                rttm_rows.append(f"SPEAKER {rec} 1 {sec(st)} {sec(en-st)} <NA> <NA> {sp} <NA>")

        # Sort and write atomically
        with open(os.path.join(out, "wav.scp"), "w") as f:
            for rec, wav in sorted(wav_pairs, key=lambda x: x[0]):   # sort by rec
                print(f"{rec} {wav}", file=f)

        with open(os.path.join(out, "segments"), "w") as f:
            for utt in sorted(seg_rows.keys()):                      # sort by utt
                rec, st, en = seg_rows[utt]
                print(f"{utt} {rec} {st} {en}", file=f)

        with open(os.path.join(out, "utt2spk"), "w") as f:
            for utt in sorted(u2s_rows.keys()):                      # sort by utt
                print(f"{utt} {u2s_rows[utt]}", file=f)

        with open(os.path.join(out, "rttm"), "w") as f:
            for ln in sorted(rttm_rows):
                print(ln, file=f)

        # build spk2utt (fix_data_dir.sh will also re-create it)
        os.system(f"utils/utt2spk_to_spk2utt.pl {out}/utt2spk > {out}/spk2utt")


    print("Done.", file=sys.stderr)

if __name__ == "__main__":
    main()
