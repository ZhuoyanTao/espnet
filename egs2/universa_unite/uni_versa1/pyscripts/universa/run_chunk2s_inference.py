#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from kaldiio import ReadHelper
import yaml

from espnet2.universa.ar_universa.ar_universa import ARUniversa
from espnet2.universa.ar_universa.ar_universa_utils import chunk_audio


from pathlib import Path
import yaml
import torch

from espnet2.universa.ar_universa.ar_universa import ARUniversa


from pathlib import Path
import torch
from espnet2.tasks.universa import UniversaTask  # adjust exact name if needed

from pathlib import Path
import torch
import yaml

from espnet2.tasks.universa import UniversaTask  # this import is already working

def collect_numeric(prefix, obj, out_dict):
    """Recursively collect numeric (float-like) leaves from nested dict/list/arrays/tensors."""
    import numpy as np
    import torch

    # Scalars
    if isinstance(obj, (int, float, np.floating)):
        if prefix:  # avoid empty key
            out_dict[prefix] = float(obj)
        return

    # Numpy arrays
    if isinstance(obj, np.ndarray):
        if obj.size == 1:
            out_dict[prefix] = float(obj.reshape(-1)[0])
        elif obj.size > 0:
            # heuristic: mean if multi-valued
            out_dict[prefix] = float(obj.mean())
        return

    # Torch tensors
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            out_dict[prefix] = float(obj.view(-1)[0].item())
        elif obj.numel() > 0:
            # heuristic: mean if multi-valued
            out_dict[prefix] = float(obj.mean().item())
        return

    # Dicts
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            collect_numeric(new_prefix, v, out_dict)
        return

    # Lists / tuples
    if isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            new_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            collect_numeric(new_prefix, v, out_dict)
        return

    # Everything else: ignore
    return

def pad_stack_1d(chunks, device):
    """
    chunks: list of torch.Tensor, each shape (1, T)
    returns:
      wav_pad: (B, Tmax)
      lengths: (B,)
    """
    lens = [c.shape[-1] for c in chunks]
    Tmax = max(lens)
    B = len(chunks)
    wav_pad = torch.zeros(B, Tmax, device=device, dtype=chunks[0].dtype)
    for i, c in enumerate(chunks):
        wav_pad[i, : c.shape[-1]] = c.squeeze(0)
    lengths = torch.tensor(lens, device=device, dtype=torch.long)
    return wav_pad, lengths



def load_model(expdir: Path, device: str = "cuda"):
    """
    Load ARUniversa model from an exp dir containing config.yaml and a .pth checkpoint.
    """
    config_path = expdir / "config.yaml"
    ckpt_path = expdir / "valid.loss.best.pth"   # or latest.pth / 31epoch.pth / etc.

    assert config_path.is_file(), f"Missing config: {config_path}"
    assert ckpt_path.is_file(), f"Missing checkpoint: {ckpt_path}"

    # Use the ESPnet task helper (same as universa_inference.py)
    model, train_args = UniversaTask.build_model_from_file(
        config_file=str(config_path),   # <- key name must be config_file
        model_file=str(ckpt_path),
        device=device,
    )

    model.to(device)
    model.eval()
    return model



def run_chunk2s_inference(
    model,
    wav_scp: str,
    output_path: str,
    metric_list=None,
    max_utts: int = 1000,
    chunk_s: float = 2.0,
    hop_s: float = 2.0,
    chunk_bs: int = 1,
    device: str = "cuda",
):
    """
    For each utt in wav_scp:
      1. Read waveform via kaldiio
      2. Chunk into 2s windows
      3. Run ARUniversa inference on each chunk
      4. Save per-utt chunk metrics into JSONL
    """
    wav_scp = Path(wav_scp)
    assert wav_scp.is_file(), f"Missing wav.scp: {wav_scp}"

    out_f = open(output_path, "w", encoding="utf-8")

    # These are the metrics you *prefer* to keep IF they exist
    preferred_keep_keys = [
        "nisqa_mos_pred",
        "utmos",
        "utmosv2",
        "dns_overall",
        "plcmos",
    ]

    # -------- pick the "core" universa object (with set_inference) --------
    if hasattr(model, "set_inference"):
        target = model
    elif hasattr(model, "universa") and hasattr(model.universa, "set_inference"):
        target = model.universa
    elif hasattr(model, "model") and hasattr(model.model, "set_inference"):
        target = model.model
    else:
        target = None

    # -------- discover what metrics the tokenizer actually knows --------
    all_metric_names = []
    if target is not None and hasattr(target, "metric_tokenizer"):
        mtok = target.metric_tokenizer

        # 1) Names defined in the tokenizer config (most canonical)
        if hasattr(mtok, "tokenizer_config"):
            all_metric_names = list(mtok.tokenizer_config.keys())

        # 2) Sanity: metric_offset keys should match (or be a subset)
        if hasattr(mtok, "metric_offset"):
            offset_keys = list(mtok.metric_offset.keys())
            if not all_metric_names:
                all_metric_names = offset_keys
            else:
                # keep only those that appear in both, to be safe
                all_metric_names = [
                    m for m in all_metric_names if m in offset_keys
                ]

    # -------- robust metric_list construction --------
    if metric_list is None:
        # If tokenizer told us the metric names, use those
        if all_metric_names:
            metric_list = all_metric_names
        # Otherwise, try metric2id on different wrappers
        elif hasattr(model, "metric2id"):
            metric_list = list(model.metric2id.keys())
        elif hasattr(model, "universa") and hasattr(model.universa, "metric2id"):
            metric_list = list(model.universa.metric2id.keys())
        elif hasattr(model, "model") and hasattr(model.model, "metric2id"):
            metric_list = list(model.model.metric2id.keys())
        else:
            # absolute last fallback: just your preferred metric names
            metric_list = preferred_keep_keys.copy()

    # If we know what metrics the tokenizer supports, intersect
    if all_metric_names:
        metric_list = [m for m in metric_list if m in all_metric_names]

    # If we somehow killed everything, fall back to whatever we know
    if not metric_list:
        metric_list = all_metric_names if all_metric_names else preferred_keep_keys.copy()

    # For saving to JSON later, we’ll use:
    #   - intersection of preferred_keep_keys and metric_list if possible
    #   - otherwise, all metric_list
    keep_keys = [k for k in preferred_keep_keys if k in metric_list]
    if not keep_keys:
        keep_keys = metric_list[:]  # keep everything the model predicts

    # -------- now it's safe to call set_inference --------
    if target is not None:
        target.set_inference(
            beam_size=1,
            metric_list=metric_list,
            skip_meta_label_score=False,
            save_token_seq=False,
        )
    assert getattr(target, "search_module", None) is not None, "search_module not set; did set_inference run?"


    num_done = 0
    with ReadHelper("scp:" + str(wav_scp)) as reader:
        for utt, (fs, wav_np) in reader:
            if max_utts > 0 and num_done >= max_utts:
                break

            # 1. Convert to tensor (1, T)
            wav = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)
            T = wav.shape[-1]

            # 2. Create 2s chunks
            chunks = chunk_audio(
                wav,
                fs,
                chunk_s=chunk_s,
                hop_s=hop_s,
                keep_tail=True,
            )
            

            chunk_metrics = []

            # process chunks in minibatches
        # process chunks in minibatches
        for s in range(0, len(chunks), chunk_bs):
            mb = [c.to(device) for c in chunks[s : s + chunk_bs]]
            mb_wav, mb_lens = pad_stack_1d(mb, device=device)   # (B, T), (B,)

            assert hasattr(target, "frontend") and target.frontend is not None, \
                "No target.frontend found. This model expects features; need a frontend."

            with torch.no_grad():
                # 1) waveform -> features  (expect (B, T_frames, F) e.g. F=80)
                feats, feats_lens = target.frontend(mb_wav, mb_lens)

                # 2) optional normalize (only if the model is configured for it)
                if getattr(target, "use_normalize", False):
                    # normalize expects features here
                    with torch.cuda.amp.autocast(False):
                        feats, feats_lens = target.normalize(feats, feats_lens)

                # 3) encoder
                audio_enc, audio_enc_lens, _ = target.audio_encoder(feats, feats_lens)

                # 4) match encode() concatenation behavior if decoder expects ref branches
                enc_list = [audio_enc]
                if getattr(target, "use_ref_audio", False):
                    enc_list.append(torch.zeros_like(audio_enc))
                if getattr(target, "use_ref_text", False):
                    enc_list.append(torch.zeros_like(audio_enc))
                audio_enc = torch.cat(enc_list, dim=-1)

                # ---- PROBE (run once) ----
                print(f"mb_wav={tuple(mb_wav.shape)} feats={tuple(feats.shape)} audio_enc={tuple(audio_enc.shape)}", flush=True)
                print(f"mb_lens={mb_lens.tolist()} feats_lens={feats_lens.tolist()} audio_enc_lens={audio_enc_lens.tolist()}", flush=True)
                raise SystemExit("encode probe done")

                # Beam search + tokenize per chunk (still per-item, but encoder work is batched)
                for i in range(audio_enc.size(0)):
                    Tenc = audio_enc.size(1)          # padded time length in this minibatch
                    Li = int(audio_enc_lens[i].item())# true time length for this sample
                    if Li != Tenc:
                        print(f"[PAD DETECTED] utt={utt} mb_start={s} i={i} Li={Li} Tenc={Tenc}", flush=True)
                    # nbest hyps for this chunk
                    enc_i = audio_enc[i, :Li]     # important
                    nbest_hyps = target.search_module.forward(enc_i)
                    # nbest_hyps = target.search_module.forward(audio_enc[i])
                    yseq = nbest_hyps[0].yseq

                    # convert token seq -> metric dict (same format you were using)
                    pred = target.metric_tokenizer.tokenseq2metric(yseq, return_dict=True)
                    pred["use_tokenizer_metrics"] = True
                    pred["sequential_metrics"] = True

                    # ---- extract metrics directly from pred (your same logic) ----
                    metrics_subset = {}

                    for k, v in pred.items():
                        if k in {"use_tokenizer_metrics", "sequential_metrics",
                                "encoded_feat", "token_seq"}:
                            continue
                        if k not in keep_keys:
                            continue

                        if isinstance(v, list) and len(v) == 1 and isinstance(v[0], (float, int)):
                            metrics_subset[k] = float(v[0])
                        elif isinstance(v, (float, int)):
                            metrics_subset[k] = float(v)
                        else:
                            metrics_subset[k] = v

                    if not metrics_subset:
                        for k, v in pred.items():
                            if k in {"use_tokenizer_metrics", "sequential_metrics",
                                    "encoded_feat", "token_seq"}:
                                continue
                            if isinstance(v, list) and len(v) == 1 and isinstance(v[0], (float, int)):
                                metrics_subset[k] = float(v[0])
                            elif isinstance(v, (float, int)):
                                metrics_subset[k] = float(v)

                    chunk_metrics.append(metrics_subset)


            # 3. Aggregate & write JSONL
            out_obj = {
                "utt": utt,
                "fs": int(fs),
                "num_samples": int(T),
                "num_chunks": len(chunks),
                "chunk_s": float(chunk_s),
                "hop_s": float(hop_s),
                "chunk_metrics": chunk_metrics,
            }
            out_f.write(json.dumps(out_obj) + "\n")
            num_done += 1

            if num_done % 50 == 0:
                print(f"Processed {num_done} utterances...", flush=True)

    out_f.close()
    print(f"Finished. Wrote {num_done} utterances to {output_path}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expdir",
        type=str,
        required=True,
        help="Experiment directory (contains config.yaml and .pth)",
    )
    parser.add_argument(
        "--wav_scp",
        type=str,
        default="dump/raw/overall_base/wav.scp",
        help="Path to wav.scp",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="chunk2s_metrics.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--max_utts",
        type=int,
        default=200,
        help="Max number of utterances (0 = all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--chunk_s",
        type=float,
        default=2.0,
        help="Chunk length in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--hop_s",
        type=float,
        default=None,
        help="Hop length in seconds (default: same as chunk_s)",
    )
    parser.add_argument(
    "--chunk_bs",
    type=int,
    default=1,
    help="Batch size over chunks within an utterance (encoder is batched; beam search still per-chunk)",
    )
    args = parser.parse_args()

    expdir = Path(args.expdir)
    model = load_model(expdir, device=args.device)

    # if hop_s not given, default to chunk_s (your docstring promise)
    hop_s = args.hop_s if args.hop_s is not None else args.chunk_s

    run_chunk2s_inference(
        model=model,
        wav_scp=args.wav_scp,
        output_path=args.output,
        metric_list=None,           # or list of metric names if you want a subset
        max_utts=args.max_utts,
        chunk_s=args.chunk_s,       # <-- use cli
        hop_s=hop_s,                # <-- use cli or default
        chunk_bs=args.chunk_bs,
        device=args.device,
    )



if __name__ == "__main__":
    main()
