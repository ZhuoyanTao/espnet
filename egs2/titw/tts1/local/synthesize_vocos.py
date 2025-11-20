#!/usr/bin/env python3
import argparse, os, glob, sys
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

def eprint(*a): print(*a, file=sys.stderr)

# -------------------- Stats / denorm --------------------
def load_stats(stats_npz: Path):
    if not stats_npz or not str(stats_npz) or not Path(stats_npz).exists():
        return None, None, None
    S = np.load(stats_npz)
    mean  = S.get("mean")
    istd  = S.get("istd")   # inverse std (1/std)
    scale = S.get("scale")  # sometimes used instead of istd
    if mean is None or (istd is None and scale is None):
        raise RuntimeError(f"Stats file missing fields: {stats_npz} (need mean and istd/scale)")
    return mean, istd, scale

def denorm_dir(norm_dir: Path, denorm_dir: Path, mean, istd, scale, overwrite=False):
    denorm_dir.mkdir(parents=True, exist_ok=True)
    existing = list(denorm_dir.glob("*.npy"))
    if existing and not overwrite:
        eprint(f"[skip] denorm exists with files: {denorm_dir}")
        return
    n = 0
    for p in sorted(Path(norm_dir).glob("*.npy")):
        x = np.load(p)
        if istd is not None:
            y = x * (1.0 / istd) + mean
        else:
            y = x * scale + mean
        np.save(denorm_dir / p.name, y.astype(np.float32, copy=False))
        n += 1
    eprint(f"[ok] denorm -> {denorm_dir} ({n} files)")

# -------------------- Vocos synth --------------------
def to_BCT(mel_np: np.ndarray, expected_bins: int) -> torch.Tensor:
    """
    Accept (T, M) or (M, T). Return (1, M, T) with M==expected_bins (float32).
    """
    if mel_np.ndim != 2:
        raise ValueError(f"mel ndim={mel_np.ndim}, expected 2")
    h, w = mel_np.shape
    if w == expected_bins:
        # (T, M) -> (M, T)
        mel_np = mel_np.T
    elif h == expected_bins:
        # already (M, T)
        pass
    else:
        raise ValueError(
            f"Expected {expected_bins} mel bins; got shape {mel_np.shape}. "
            f"Use a Vocos checkpoint that matches your mel bins or convert upstream."
        )
    return torch.from_numpy(mel_np.astype(np.float32, copy=False)).unsqueeze(0)  # (1,M,T)

def run_vocos(denorm_dir: Path, wav_dir: Path, ckpt: str, sr: int, device: str,
              expected_bins: int, overwrite=False):
    from vocos import Vocos
    wav_dir.mkdir(parents=True, exist_ok=True)
    v = Vocos.from_pretrained(ckpt).to(device).eval()  # NOTE: no map_location in this API
    n = 0
    mlist = sorted(denorm_dir.glob("*.npy"))
    for m in mlist:
        outp = wav_dir / f"{m.stem}.wav"
        if outp.exists() and not overwrite:
            continue
        mel_np = np.load(m)
        mel_t = to_BCT(mel_np, expected_bins).to(device)  # (1, M, T)
        with torch.no_grad():
            wav = v.decode(mel_t).cpu().squeeze().numpy()
        sf.write(outp, wav, sr)
        n += 1
    eprint(f"[ok] vocos -> {wav_dir} ({n} wavs)")

def build_wav_scp(feats_scp: Path, wav_dir: Path, out_scp: Path):
    if not feats_scp.exists():
        eprint(f"[warn] feats.scp not found at {feats_scp}; skipping wav.scp")
        return
    lines = []
    with open(feats_scp) as f:
        for line in f:
            if not line.strip(): continue
            utt, path = line.strip().split(maxsplit=1)
            base = Path(path).stem  # e.g., foo.npy -> foo
            wavp = wav_dir / f"{base}.wav"
            if wavp.exists():
                lines.append(f"{utt} {wavp}\n")
    if lines:
        with open(out_scp, "w") as g:
            g.writelines(lines)
        eprint(f"[ok] wav.scp -> {out_scp} ({len(lines)} entries)")
    else:
        eprint("[warn] No matching wavs for feats.scp entries; wav.scp not written")

# -------------------- Orchestration --------------------
def process_output_dir(out_dir: Path, stats_npz: Path, ckpt: str, sr: int, device: str,
                       expected_bins: int, overwrite=False, denorm_only=False, synth_only=False):
    norm = out_dir / "norm"
    denorm = out_dir / "denorm"
    feats_scp = out_dir / "norm" / "feats.scp"
    wav_dir = out_dir / "vocos_wav"

    if not norm.exists():
        eprint(f"[skip] {out_dir}: no 'norm/' found")
        return

    mean, istd, scale = load_stats(stats_npz) if stats_npz else (None, None, None)

    # If no stats given/found -> use norm directly (your case: normalize=None)
    use_norm_direct = (mean is None and istd is None and scale is None)

    if not synth_only:
        if use_norm_direct:
            denorm = norm  # shortcut: real log-mels already
            eprint(f"[info] Using 'norm/' directly (no stats)")
        else:
            denorm_dir(norm, denorm, mean, istd, scale, overwrite=overwrite)

    if not denorm_only:
        run_vocos(denorm, wav_dir, ckpt, sr, device, expected_bins, overwrite=overwrite)
        build_wav_scp(feats_scp, wav_dir, wav_dir / "wav.scp")

def main():
    ap = argparse.ArgumentParser(description="Synthesize ESPnet TTS mels with Vocos (supports optional de-normalization)")
    ap.add_argument("--base", type=str, required=True, help="Decode log dir containing output.* subdirs")
    ap.add_argument("--stats", type=str, default="", help="Path to feats_stats.npz (optional; if omitted, uses norm/ directly)")
    ap.add_argument("--vocos-ckpt", type=str, default="charactr/vocos-mel-24khz", help="Vocos checkpoint or HF repo id")
    ap.add_argument("--sr", type=int, default=24000, help="Sample rate expected by the vocoder")
    ap.add_argument("--mel-bins", type=int, default=100, help="Expected mel bins for the Vocos checkpoint (e.g., 100 for charactr/vocos-mel-24khz)")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="Device for Vocos")
    ap.add_argument("--task-id", type=int, default=None, help="Process only output.<task-id>")
    ap.add_argument("--outputs", type=str, default=None, help="Comma-separated list like 1,5,9 (overrides --task-id)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing denorm/wavs")
    ap.add_argument("--denorm-only", action="store_true", help="Only produce denorm/*.npy")
    ap.add_argument("--synth-only", action="store_true", help="Only synthesize (assumes denorm exists or uses norm/)")
    args = ap.parse_args()

    base = Path(args.base)
    stats_npz = Path(args.stats) if args.stats else None

    # Decide which output dirs to process
    if args.outputs:
        out_dirs = [base / f"output.{x.strip()}" for x in args.outputs.split(",") if x.strip()]
    elif args.task_id is not None:
        out_dirs = [base / f"output.{args.task_id}"]
    else:
        out_dirs = sorted([p for p in base.glob("output.*") if p.is_dir()])

    if not out_dirs:
        eprint(f"No output.* directories found under {base}")
        sys.exit(1)

    eprint(f"[info] Processing {len(out_dirs)} output dirs under {base}")
    for od in out_dirs:
        eprint(f"\n=== {od} ===")
        process_output_dir(
            od, stats_npz, args.vocos_ckpt, args.sr, args.device, args.mel_bins,
            overwrite=args.overwrite, denorm_only=args.denorm_only, synth_only=args.synth_only,
        )

if __name__ == "__main__":
    main()
