import argparse, csv, json, os, re, sys
from pathlib import Path

# Heuristics: try to find reasonable column names
CAND_UTT = ["utt", "utt_id", "uttid", "id", "key", "name"]
CAND_PATH = ["path", "wav", "wav_path", "filepath", "file", "audio_path"]
CAND_MOS = ["mos", "score", "mean_opinion_score", "overall_mos", "rating"]

def pick(cols, cands):
    lc = [c.lower() for c in cols]
    for c in cands:
        if c.lower() in lc:
            return cols[lc.index(c.lower())]
    return None

def read_table(p):
    delim = "\t" if p.suffix.lower() in [".tsv"] else ","
    with p.open("r", encoding="utf-8") as f:
        sniffer = csv.Sniffer()
        head = f.read(4096)
        f.seek(0)
        try:
            if sniffer.has_header(head):
                reader = csv.DictReader(f, delimiter=delim)
                rows = list(reader)
                return rows, reader.fieldnames
        except Exception:
            pass
        # fallback: assume headered
        reader = csv.DictReader(f, delimiter=delim)
        rows = list(reader)
        return rows, reader.fieldnames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voicemos-root", required=True, help="Path to extracted Zenodo folder (contains main/ …)")
    ap.add_argument("--out-root", required=True, help="ESPnet recipe root with data/ subdir")
    ap.add_argument("--train-name", default="train")
    ap.add_argument("--dev-name", default="dev")
    ap.add_argument("--test-name", default="test")
    ap.add_argument("--split-hints", nargs="*", default=["bvcc", "main", "metadata", "split", "list", "csv", "tsv"])
    args = ap.parse_args()

    vm = Path(args.voicemos_root)
    out = Path(args.out_root) / "data"
    out.mkdir(exist_ok=True, parents=True)

    # Find candidate tables for train/dev/test
    tables = list(vm.rglob("*.tsv")) + list(vm.rglob("*.csv"))
    # Prefer files with “train/dev/test” in name
    def score_table(p):
        n = p.name.lower()
        s = 0
        for k in ["train","dev","valid","test","bvcc","main"]:
            if k in n: s += 1
        return s
    tables.sort(key=score_table, reverse=True)

    buckets = {"train":None, "dev":None, "test":None}
    for p in tables:
        rows, cols = read_table(p)
        if not rows or not cols: continue
        c_utt  = pick(cols, CAND_UTT)
        c_path = pick(cols, CAND_PATH)
        c_mos  = pick(cols, CAND_MOS)
        if not c_path or not c_mos:
            continue
        lname = p.name.lower()
        if "train" in lname and buckets["train"] is None: buckets["train"] = (p, rows, c_utt, c_path, c_mos)
        elif ("dev" in lname or "valid" in lname) and buckets["dev"] is None: buckets["dev"] = (p, rows, c_utt, c_path, c_mos)
        elif "test" in lname and buckets["test"] is None: buckets["test"] = (p, rows, c_utt, c_path, c_mos)

    # If still missing, just take top 3 best-scoring tables as fallback
    if any(v is None for v in buckets.values()):
        fallback = [t for t in tables[:3] if read_table(t)[0]]
        for split, tup in zip(["train","dev","test"], fallback):
            if buckets[split] is None:
                rows, cols = read_table(tup)
                c_utt  = pick(cols, CAND_UTT)
                c_path = pick(cols, CAND_PATH)
                c_mos  = pick(cols, CAND_MOS)
                if c_path and c_mos:
                    buckets[split] = (tup, rows, c_utt, c_path, c_mos)

    for split, target in [("train",args.train_name), ("dev",args.dev_name), ("test",args.test_name)]:
        if buckets[split] is None:
            print(f"[WARN] Could not auto-locate a {split} table. Skipping.", file=sys.stderr)
            continue
        p, rows, c_utt, c_path, c_mos = buckets[split]
        ddir = out / target
        ddir.mkdir(parents=True, exist_ok=True)
        with (ddir/"wav.scp").open("w", encoding="utf-8") as fw, \
             (ddir/"metric.scp").open("w", encoding="utf-8") as fm, \
             (ddir/"text").open("w", encoding="utf-8") as ft, \
             (ddir/"utt2spk").open("w", encoding="utf-8") as fu:
            for i, r in enumerate(rows):
                # utterance id
                utt = (r.get(c_utt) or "").strip()
                if not utt:
                    # make one from filename
                    utt = Path(r[c_path]).stem
                utt = re.sub(r"\s+", "_", utt)
                wav_path = r[c_path].strip()
                mos = float(r[c_mos])
                # Write wav.scp (absolute if possible)
                if not os.path.isabs(wav_path):
                    # Try relative to the file’s directory
                    wp = (p.parent / wav_path).resolve()
                    wav_path = str(wp)
                fw.write(f"{utt} {wav_path}\n")
                # metric.scp: single metric named 'mos'
                fm.write(f"{utt} mos:{mos}\n")
                # dummy text / utt2spk (not used by ARECHO but keeps utils happy)
                ft.write(f"{utt} DUMMY\n")
                fu.write(f"{utt} spk\n")
        print(f"[OK] Wrote {split} → {ddir}")

    # Also write metric2type and metric2id for train (uni_versa.sh stage 6 can also derive them)
    train_dir = out / args.train_name
    with (train_dir/"metric2type").open("w") as f:
        f.write("mos numerical\n")
    with (train_dir/"metric2id").open("w") as f:
        f.write("mos 0\n")

if __name__ == "__main__":
    main()
