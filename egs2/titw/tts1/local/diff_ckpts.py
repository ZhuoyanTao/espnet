#!/usr/bin/env python
import argparse, torch, itertools, re
from collections import OrderedDict, defaultdict
try:
    from safetensors.torch import load_file as load_safetensors
except Exception:
    load_safetensors = None

PREFIXES = [
    "module.", "tts.", "tts_model.", "generator.", "model.", "net.", "net_g.", "network.", "state_dict.", "encoder.", "decoder."
]

def load_any(path):
    if path.endswith(".safetensors"):
        if load_safetensors is None:
            raise RuntimeError("pip install safetensors")
        sd = load_safetensors(path, device="cpu")
        return OrderedDict(sd)
    else:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        # unwrap common wrappers
        if isinstance(obj, dict):
            for k in ["model", "state_dict", "net", "params", "module"]:
                if k in obj and isinstance(obj[k], dict):
                    return OrderedDict(obj[k])
        if isinstance(obj, OrderedDict):
            return obj
        # sometimes ESPnet wraps deeper
        for k in list(obj.keys()):
            if isinstance(obj[k], dict):
                return OrderedDict(obj[k])
        raise RuntimeError(f"Unrecognized checkpoint structure for {path}")

def strip_prefixes(k):
    for p in PREFIXES:
        if k.startswith(p):
            k = k[len(p):]
    return k

def norm_keys(sd):
    out = OrderedDict()
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if v.dtype.is_floating_point or v.dtype.is_complex or v.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bfloat16, torch.float16, torch.float32, torch.float64):
            out[strip_prefixes(k)] = v
    return out

def suffixes(k, max_parts=3):
    parts = k.split(".")
    sufs = []
    for r in range(1, max_parts+1):
        sufs.append(".".join(parts[-r:]))
    return sufs

def flatten_vecs(pairs):
    vecs_a, vecs_b, keys, lens = [], [], [], []
    for k, (a, b) in pairs.items():
        vecs_a.append(a.float().reshape(-1))
        vecs_b.append(b.float().reshape(-1))
        keys.append(k)
        lens.append(a.numel())
    return torch.cat(vecs_a), torch.cat(vecs_b), keys, lens

def rel_change(a, b):
    num = torch.linalg.vector_norm(a - b)
    den = torch.linalg.vector_norm(b) + 1e-12
    return (num / den).item()

def cosine(a, b):
    num = torch.dot(a, b)
    den = (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + 1e-12)
    return (num / den).item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("finetuned")
    ap.add_argument("base")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    sd_ft_raw = load_any(args.finetuned)
    sd_base_raw = load_any(args.base)

    # quick LoRA heuristic
    loraish = [k for k in sd_ft_raw.keys() if re.search(r"(lora_(up|down)|lora\.(A|B)|lora_alpha)", k)]
    if len(loraish) > 10:
        print("WARNING: Fine-tuned checkpoint looks like LoRA adapters only.")
        print("         You’ll want to MERGE the adapters into the base before comparing (see section B).")

    sd_ft = norm_keys(sd_ft_raw)
    sd_base = norm_keys(sd_base_raw)

    # Try exact normalized match first
    pairs = {}
    unmatched_ft = set(sd_ft.keys())
    unmatched_base = set(sd_base.keys())
    for k, ta in sd_ft.items():
        tb = sd_base.get(k, None)
        if tb is not None and ta.shape == tb.shape:
            pairs[k] = (ta, tb)
            unmatched_ft.discard(k)
            unmatched_base.discard(k)

    # If too few matches, try suffix matching (2-3 components)
    if len(pairs) < 10:
        index_base = defaultdict(list)
        for kb, vb in sd_base.items():
            for suf in suffixes(kb, 3):
                index_base[(suf, vb.shape)].append(kb)

        for k in list(unmatched_ft):
            ta = sd_ft[k]
            matched = False
            for suf in suffixes(k, 3):
                cands = index_base.get((suf, ta.shape), [])
                if cands:
                    kb = cands[0]
                    pairs[k] = (ta, sd_base[kb])
                    unmatched_ft.discard(k)
                    unmatched_base.discard(kb)
                    matched = True
                    break

    if not pairs:
        print("ERROR: Still no overlapping tensors with matching shapes after normalization & suffix matching.")
        print("Common causes:\n  * Comparing LoRA-only adapters to a full base model\n  * Mismatched architectures (different F5 variants / vocoder head sizes)\n")
        print("Tips:\n  - Check a few keys from each file to see their namespaces differ:")
        print("    python - <<'PY'\nimport torch\nfrom safetensors.torch import load_file as lf\np1='{}'; p2='{}'\nprint('--- FT keys sample ---')\ntry:\n d1=torch.load(p1,map_location='cpu',weights_only=True)\nexcept: d1=lf(p1)\nprint(list(d1 if isinstance(d1,dict) else {} )[:40])\nprint('--- BASE keys sample ---')\ntry:\n d2=torch.load(p2,map_location='cpu',weights_only=True)\nexcept: d2=lf(p2)\nprint(list(d2 if isinstance(d2,dict) else {} )[:40])\nPY".format(args.finetuned, args.base))
        return

    v_ft, v_base, keys, lens = flatten_vecs(pairs)
    cos = cosine(v_ft, v_base)
    rc = rel_change(v_ft, v_base)
    print(f"== Global comparison over {len(keys)} matched tensors ==")
    print(f"Cosine similarity: {cos:.6f}")
    print(f"Relative L2 change: {rc:.6e}")

    # per-layer
    per = []
    i0 = 0
    for k, n in zip(keys, lens):
        a = v_ft[i0:i0+n]
        b = v_base[i0:i0+n]
        per.append((rel_change(a, b), k, n))
        i0 += n
    per.sort(reverse=True, key=lambda x: x[0])

    print(f"\n== Top {args.top} layers by relative change ==")
    for rc_i, k, n in per[:args.top]:
        print(f"{rc_i:12.6e}  {k}  (params: {n})")

    # coverage
    diffs = torch.abs(v_ft - v_base)
    base_norms = torch.abs(v_base)
    rel = diffs / (base_norms + 1e-12)
    for t in [1e-4, 1e-3, 1e-2, 1e-1]:
        pct = (rel > t).float().mean().item() * 100.0
        print(f"> {t:.0e}: {pct:6.2f}% of parameters")

    # brief unmatched stats (can be large)
    if len(unmatched_ft) or len(unmatched_base):
        print(f"\nUnmatched FT tensors: {len(unmatched_ft)} | Unmatched BASE tensors: {len(unmatched_base)}")
        print("Examples FT:", list(itertools.islice(iter(unmatched_ft), 5)))
        print("Examples BASE:", list(itertools.islice(iter(unmatched_base), 5)))

if __name__ == "__main__":
    main()
