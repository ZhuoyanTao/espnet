#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import defaultdict

# Keep L1/L2 as-is (or uncomment LABEL_MAP to remap to a/c)
ALLOWED_LABELS = {"L1", "L2"}
LABEL_MAP = {}  # e.g., {"L1": "a", "L2": "c"}


def make_uttid(rec: str, onset: float, dur: float, lab: str,
               scale: int = 10000, suffix: str = "") -> str:
    """Always speaker-first: <spk>_<rec>_<start>-<end>[suffix]."""
    s = int(round(onset * scale))
    e = int(round((onset + dur) * scale))
    return f"{lab}_{rec}_{s:07d}-{e:07d}{suffix}"


def parse_line(line: str):
    """
    Accept only DISPLACE-style LANGUAGE lines:
      LANGUAGE <rec> 1 <tbeg> <tdur> <na> <na> <label> <na> <na>
    where <label> is L1/L2 (or will be remapped to a/c).
    """
    t = line.strip().split()
    if not t or t[0].upper() != "LANGUAGE":
        return None
    if len(t) < 8:
        return None

    rec = t[1]
    try:
        tbeg = float(t[3])
        tdur = float(t[4])
    except Exception:
        return None
    if tdur <= 0:
        return None

    raw_label = t[7]
    if raw_label not in ALLOWED_LABELS:
        return None

    label = LABEL_MAP.get(raw_label, raw_label)
    return rec, tbeg, tdur, label


def rttm_files(split_dir: Path):
    fs = sorted(split_dir.glob("*_LANGUAGE.rttm"))
    if not fs:
        fs = sorted(split_dir.glob("*.rttm"))
    return fs


def process(in_dir: Path,
            out_dir: Path,
            scale: int,
            rttm_kind: str = "LANGUAGE",
            dup_factor: int = 1):
    """
    dup_factor = 1  -> original behavior
    dup_factor = 4  -> each segment becomes 4 utts with different IDs
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = rttm_files(in_dir)
    print(f"[INFO] {in_dir}: {len(files)} RTTMs")
    if not files:
        raise FileNotFoundError(f"No RTTMs in {in_dir}")

    u2s_pairs = []
    s2u = defaultdict(list)
    kept_segments = []  # (rec, tbeg, tdur, lab) – no duplication here
    seg_rows = []       # (utt, rec, tbeg, tend)

    seen, parsed = 0, 0

    for f in files:
        with f.open() as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith(";"):
                    continue
                seen += 1
                p = parse_line(line)
                if p is None:
                    continue
                rec, tbeg, tdur, lab = p
                kept_segments.append((rec, tbeg, tdur, lab))
                tend = tbeg + tdur
                parsed += 1

                # Duplicate utts if requested
                if dup_factor == 1:
                    # Keep original naming for non-duplicated splits
                    utt = make_uttid(rec, tbeg, tdur, lab, scale=scale, suffix="")
                    u2s_pairs.append((utt, lab))
                    s2u[lab].append(utt)
                    seg_rows.append((utt, rec, tbeg, tend))
                else:
                    for d in range(dup_factor):
                        suffix = f"_d{d}"
                        utt = make_uttid(rec, tbeg, tdur, lab, scale=scale, suffix=suffix)
                        u2s_pairs.append((utt, lab))
                        s2u[lab].append(utt)
                        seg_rows.append((utt, rec, tbeg, tend))

    print(f"[INFO] lines seen={seen}, parsed={parsed}")
    if parsed == 0:
        raise RuntimeError(f"No LANGUAGE entries with allowed labels in {in_dir}")

    # dedup + sort
    u2s_pairs = sorted(set(u2s_pairs))
    seg_rows = sorted(set(seg_rows))
    for k in list(s2u.keys()):
        s2u[k] = sorted(set(s2u[k]))

    # ---- write utt2spk / spk2utt
    with (out_dir / "utt2spk").open("w") as fw:
        for utt, lab in u2s_pairs:
            fw.write(f"{utt} {lab}\n")

    with (out_dir / "spk2utt").open("w") as fw:
        for lab in sorted(s2u.keys()):
            fw.write(lab + " " + " ".join(s2u[lab]) + "\n")

    # ---- write segments (utt rec tbeg tend)
    with (out_dir / "segments").open("w") as fw:
        for utt, rec, tbeg, tend in seg_rows:
            fw.write(f"{utt} {rec} {tbeg:.3f} {tend:.3f}\n")

    # ---- write a FILTERED espnet_rttm (only LANGUAGE L1/L2)
    # Note: we *don't* duplicate the actual RTTM content; that describes the real audio.
    with (out_dir / "espnet_rttm").open("w") as fw:
        for rec, tbeg, tdur, lab in kept_segments:
            fw.write(f"{rttm_kind} {rec} 1 {tbeg:.3f} {tdur:.3f} <NA> <NA> {lab} <NA> <NA>\n")

    print(f"[OK] wrote {out_dir/'utt2spk'} ({len(u2s_pairs)} rows)")
    print(f"[OK] wrote {out_dir/'spk2utt'} ({len(s2u)} labels)")
    print(f"[OK] wrote {out_dir/'segments'} ({len(seg_rows)} rows)")
    print(f"[OK] wrote {out_dir/'espnet_rttm'} (filtered {len(kept_segments)} segments)")
    return len(u2s_pairs), len(s2u)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base_displace_dir",
        required=True,
        type=Path,
        help="e.g., /work/nvme/bbjs/ttao3/displace_audio",
    )
    ap.add_argument("--scale", type=int, default=10000)
    args = ap.parse_args()

    # (split_name) -> (out_dir_name, in_path, dup_factor)
    mapping = {
        "train": ("displace_train", args.base_displace_dir / "train", 1),
        "dev": ("displace_dev", args.base_displace_dir / "dev", 1),
        "test": ("displace_test", args.base_displace_dir / "test", 1),

        # New: 4×-duplicated versions for dev and test
        "dev_4dup": ("displace_dev_4dup", args.base_displace_dir / "dev", 4),
        "test_4dup": ("displace_test_4dup", args.base_displace_dir / "test", 4),
    }

    for split, (out_name, in_path, dup_factor) in mapping.items():
        print(f"=== {split} (dup_factor={dup_factor}) ===")
        try:
            process(in_path, Path("data") / out_name, args.scale, dup_factor=dup_factor)
        except Exception as e:
            print(f"[ERROR] {split}: {e}")
