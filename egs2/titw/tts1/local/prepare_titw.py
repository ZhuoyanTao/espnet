#!/usr/bin/env python3
import os, re, io, json, tarfile, argparse
from pathlib import Path

AUDIO_EXTS = {".wav", ".flac", ".mp3"}  # be permissive

def find_tars(src: Path, pattern: str):
    # return first match for a given pattern (e.g., "*_easy_audio.tar.gz")
    cands = sorted(src.glob(pattern))
    return cands[0] if cands else None

def read_label_map_from_metadata_tar(meta_tar_path: Path):
    """
    Build a dict: key -> label from metadata tar.
    We try several common fields: 'label', 'class', 'target', 'is_spoof', 'bonafide'.
    We also try to derive a key from filename (without ext).
    """
    label_map = {}
    if not meta_tar_path:
        return label_map

    with tarfile.open(meta_tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            name = m.name.lower()
            # handle .json or .txt (webdataset often has .json sidecars)
            if not (name.endswith(".json") or name.endswith(".txt")):
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            raw = fh.read()
            try:
                # try json
                rec = json.loads(raw.decode("utf-8", errors="ignore"))
                # deduce key
                key = (rec.get("__key__") or rec.get("key") or "").strip()
                if not key:
                    # derive from filename without extension
                    key = re.sub(r"\.[^.]+$", "", m.name.split("/")[-1])
                # find label field
                lbl = (rec.get("label") or rec.get("class") or rec.get("target"))
                if lbl is None:
                    # boolean-ish flags
                    for flag in ["is_spoof", "spoof", "bonafide"]:
                        if flag in rec:
                            v = rec[flag]
                            if isinstance(v, (bool, int)):
                                lbl = "spoof" if bool(v) and flag != "bonafide" else "bonafide"
                                break
                    if lbl is None and "category" in rec:
                        lbl = rec["category"]
                if isinstance(lbl, (int, float)):
                    lbl = "spoof" if int(lbl) == 1 else "bonafide"
                if isinstance(lbl, str):
                    l = lbl.lower()
                    if "spoof" in l or "fake" in l or "tts" in l or "synth" in l:
                        lbl = "spoof"
                    elif "bona" in l or "real" in l or "genuine" in l:
                        lbl = "bonafide"
                if key and lbl:
                    label_map[key] = lbl
            except Exception:
                # not json; try line-based "key label"
                try:
                    text = raw.decode("utf-8", errors="ignore")
                    for line in text.splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            k, v = parts[0], parts[-1]
                            v = v.lower()
                            if "spoof" in v or "fake" in v or "tts" in v or "synth" in v:
                                label_map[k] = "spoof"
                            elif any(x in v for x in ["bona", "real", "genuine"]):
                                label_map[k] = "bonafide"
                except Exception:
                    pass
    return label_map

def sanitize_id(s: str):
    s = re.sub(r"[^A-Za-z0-9_/-]+", "-", s).strip("/")
    return s.replace("/", "_")

def ext_of(name: str):
    return "." + name.split(".")[-1].lower() if "." in name else ""

def build_split(src: Path, out_dir: Path, wav_dir: Path,
                audio_tar: Path, meta_tar: Path, split_tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    # 1) labels
    label_map = read_label_map_from_metadata_tar(meta_tar)
    # 2) iterate audio members
    n = 0
    counts = {"bonafide": 0, "spoof": 0, "unknown": 0}

    wavscp = open(out_dir/"wav.scp", "w")
    utt2spk = open(out_dir/"utt2spk", "w")
    utt2label = open(out_dir/"utt2label", "w")
    textf = open(out_dir/"text", "w")

    with tarfile.open(audio_tar, "r:*") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            ext = ext_of(m.name)
            if ext not in AUDIO_EXTS:
                continue

            # derive key for label lookup (basename without ext)
            base = re.sub(r"\.[^.]+$", "", m.name.split("/")[-1])
            lbl = label_map.get(base)
            if lbl is None:
                # fallback inference from path
                low = m.name.lower()
                if any(x in low for x in ["spoof", "fake", "tts", "synth", "deepfake"]):
                    lbl = "spoof"
                elif any(x in low for x in ["bona", "real", "genuine"]):
                    lbl = "bonafide"
                else:
                    lbl = "unknown"

            # utt id
            utt = sanitize_id(f"titw_{split_tag}_{base}")
            # ensure uniqueness
            i = 1
            uorig = utt
            while (wav_dir/f"{utt}.wav").exists():
                i += 1
                utt = f"{uorig}_{i}"

            # write wav bytes (re-encode only if not .wav)
            with tf.extractfile(m) as fh:
                data = fh.read()
            # If the member is already .wav, write bytes as-is.
            # If it's flac/mp3, we still write bytes to .wav filename; most TITW audio is .wav, so keep simple.
            # If you *do* encounter non-wav, consider piping through ffmpeg; here we assume .wav.
            out_wav = wav_dir / f"{utt}.wav"
            with open(out_wav, "wb") as wf:
                wf.write(data)

            wavscp.write(f"{utt} {out_wav.resolve()}\n")
            utt2spk.write(f"{utt} {lbl}\n")
            utt2label.write(f"{utt} {lbl}\n")
            textf.write(f"{utt} {lbl}\n")
            counts[lbl] = counts.get(lbl, 0) + 1
            n += 1

    for f in (wavscp, utt2spk, utt2label, textf):
        f.close()

    # spk2utt
    spk2utt = {}
    with open(out_dir/"utt2spk") as f:
        for line in f:
            u, s = line.strip().split()
            spk2utt.setdefault(s, []).append(u)
    with open(out_dir/"spk2utt", "w") as f:
        for s, us in spk2utt.items():
            f.write(f"{s} {' '.join(us)}\n")

    print(f"[TITW:{split_tag}] wrote {n} utts -> {out_dir}")
    print(f"  bonafide: {counts.get('bonafide',0)}, spoof: {counts.get('spoof',0)}, unknown: {counts.get('unknown',0)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="downloads/titw folder")
    ap.add_argument("--out_root", default="data/titw", help="root for ESPnet data dirs")
    ap.add_argument("--wav_root", default="data/titw/wavs", help="root to store extracted wavs")
    args = ap.parse_args()

    src = Path(args.src)
    out_root = Path(args.out_root)
    wav_root = Path(args.wav_root)

    # EASY
    easy_audio = find_tars(src, "*easy_audio.tar*")
    easy_meta  = find_tars(src, "*easy_metadata.tar*")
    if easy_audio:
        build_split(
            src,
            out_root/"easy",
            wav_root/"easy",
            easy_audio,
            easy_meta,
            "easy",
        )
    else:
        print("[TITW] No easy_audio tar found.")

    # HARD
    hard_audio = find_tars(src, "*hard_audio.tar*")
    hard_meta  = find_tars(src, "*hard_metadata.tar*")
    if hard_audio:
        build_split(
            src,
            out_root/"hard",
            wav_root/"hard",
            hard_audio,
            hard_meta,
            "hard",
        )
    else:
        print("[TITW] No hard_audio tar found.")

if __name__ == "__main__":
    main()
