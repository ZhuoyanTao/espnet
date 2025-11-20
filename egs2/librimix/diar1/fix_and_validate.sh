# From egs2/librimix/diar1
SRC=/work/nvme/bbjs/ttao3/datasets/synth_out_oct17_1

for s in dev test; do
  d="data/synth_${s}"
  echo "=> Rebuilding $d/wav.scp from $d/rttm and $SRC"

  # rec IDs present in this split's RTTM
  awk '$1=="SPEAKER"{print $2}' "$d/rttm" | LC_ALL=C sort -u > "$d/.recs"

  # map all source wavs: <rec> <abs_path>
  { for w in "$SRC"/*.wav; do b=$(basename "$w" .wav); printf "%s %s\n" "$b" "$w"; done; } \
    | LC_ALL=C sort -k1,1 > "$d/.allwav"

  # join to produce wav.scp for only recs in this split
  LC_ALL=C join -1 1 -2 1 "$d/.recs" "$d/.allwav" > "$d/wav.scp"
  rm -f "$d/.recs" "$d/.allwav"

  wc -l "$d/wav.scp"

  echo "=> Rebuilding segments/utt2spk/spk2utt from $d/rttm"
  python3 - "$d" <<'PY'
import sys, collections
d=sys.argv[1]; rttm=f"{d}/rttm"; wavscp=f"{d}/wav.scp"
recs=set(x.split()[0] for x in open(wavscp) if x.strip())

def sec(x):
    x=float(x); s=f"{x:.3f}".rstrip("0").rstrip(".")
    return s if s else "0"

dup=collections.Counter()
with open(rttm) as f, open(f"{d}/segments.tmp","w") as so, open(f"{d}/utt2spk.tmp","w") as uo:
    for line in f:
        p=line.strip().split()
        if len(p)<8 or p[0]!="SPEAKER": continue
        rec=p[1]
        if rec not in recs: continue
        st=float(p[3]); en=st+float(p[4]); sp=p[7]  # language tag (en_in/hi_in/ml_in)
        base=f"{sp}-{rec}_{int(round(st*1000)):07d}-{int(round(en*1000)):07d}_{sp}"
        dup[base]+=1
        utt = base if dup[base]==1 else f"{base}-v{dup[base]}"
        so.write(f"{utt} {rec} {sec(st)} {sec(en)}\n")
        uo.write(f"{utt} {sp}\n")
PY

  LC_ALL=C sort -u -k1,1 "$d/segments.tmp" > "$d/segments"
  LC_ALL=C sort -u -k1,1 "$d/utt2spk.tmp" > "$d/utt2spk"
  rm -f "$d/segments.tmp" "$d/utt2spk.tmp"

  utils/utt2spk_to_spk2utt.pl "$d/utt2spk" > "$d/spk2utt"
  utils/fix_data_dir.sh "$d" >/dev/null
  utils/validate_data_dir.sh --no-feats --no-text "$d" || exit 1
done
