# Distortion Experiment: Step-by-Step Instructions

Run these steps from inside the ESPnet recipe root:

```
RECIPE=/work/nvme/bbjs/ttao3/espnet/egs2/universa_unite/uni_versa1
cd $RECIPE
```

All scripts accept paths relative to `--recipe_dir`; you can use relative or absolute.

---

## Step 1 — Sample utterances and inject distortions

```bash
python /path/to/distortion_generator.py \
  --recipe_dir    $RECIPE \
  --src_dump_dir  dump/raw/overall_dev \
  --audio_out_dir dump/raw/distorted_audio/full \
  --manifest_out  exp/distortion_manifest.csv \
  --n_samples     100 \
  --distortion_time 1.5 \
  --distortions   white_noise packet_drop \
  --noise_duration  0.10 \
  --drop_duration   0.20 \
  --snr_db          5.0 \
  --seed            42
```

**What it does:**
- Reads `dump/raw/overall_dev/wav.scp` (Kaldi ark format, relative paths resolved against `$RECIPE`)
- Filters utterances ≥ 2.0 s (= `distortion_time + 0.5`), samples 100 reproducibly
- Injects white noise at 1.5 s (100 ms, SNR 5 dB), then zeros 1.5–1.7 s (packet drop)
- Writes plain PCM16 wav files to `dump/raw/distorted_audio/full/`
- Writes `exp/distortion_manifest.csv` mapping `orig_utt_id ↔ dist_utt_id`

**Runtime:** ~5–10 min on a login node (CPU, 100 utterances).

**To add a new distortion later:**
```python
# In distortion_generator.py:
@register_distortion("my_distortion")
def my_distortion(audio, sr, start_sec, **kwargs):
    ...
    return audio
# Then pass --distortions white_noise packet_drop my_distortion
```

---

## Step 2 — Build ESPnet dump directories

```bash
python /path/to/dataset_builder.py \
  --recipe_dir     $RECIPE \
  --manifest       exp/distortion_manifest.csv \
  --src_overall_dev  dump/raw/overall_dev \
  --src_prefix_pattern "dump/raw/prefix_overall_dev_all_{sec}s_pred" \
  --prefix_lengths 2 4 6 8
```

**What it does:**

Creates under `dump/raw/`:

| Directory | Utterance IDs | Audio |
|-----------|---------------|-------|
| `distorted_overall_dev/` | `dist_fileid_X` | full distorted wav |
| `prefix_distorted_overall_dev_2s_pred/` | `dist_fileid_X__p002000` | truncated distorted wav |
| `prefix_distorted_overall_dev_4s_pred/` | `dist_fileid_X__p004000` | truncated distorted wav |
| `prefix_distorted_overall_dev_6s_pred/` | `dist_fileid_X__p006000` | truncated distorted wav |
| `prefix_distorted_overall_dev_8s_pred/` | `dist_fileid_X__p008000` | truncated distorted wav |

Each directory contains `wav.scp`, `utt2spk`, `spk2utt`, `feats_type`, `utt2num_samples`,
and copies of `metric.scp` / `ref_wav.scp` from the corresponding CLEAN prefix dump
(with utt_id keys remapped to `dist_*`).

Truncated wav files are written to `dump/raw/distorted_audio/prefix_2s/`, `prefix_4s/`, etc.

**Runtime:** ~10–20 min on a login node.

---

## Step 3 — Run inference via uni_versa.sh

Run inference on each distorted prefix dataset the same way you run it on the clean ones.
Replace `YOUR_INFERENCE_TAG` with whatever tag your model produces
(e.g. `inference_defer_full_metatrue_14epoch`).

```bash
cd $RECIPE

./uni_versa.sh \
  --stage           9 \
  --stop_stage      10 \
  --train_config    conf/train_aruniversa_prefix_full.yaml \
  --universa_exp    exp/universa_train_aruniversa_prefix_full_raw_fs16000_defer_full_metatrue.bak.bak.bak.bak \
  --test_sets       "prefix_distorted_overall_dev_2s_pred \
                     prefix_distorted_overall_dev_4s_pred \
                     prefix_distorted_overall_dev_6s_pred \
                     prefix_distorted_overall_dev_8s_pred" \
  --use_ref_wav     true \
  --use_ref_text    false \
  --gpu_inference   false \
  --inference_nj    128 \
  --train_args      "--defer_full_meta true" \
  --inference_args  "--defer_full_meta true"
```

Run the same command for ARECHO with its own `--universa_exp` directory.

**Inference outputs** appear at:
```
exp/<universa_exp>/<inference_tag>/prefix_distorted_overall_dev_2s_pred/metric.scp
exp/<universa_exp>/<inference_tag>/prefix_distorted_overall_dev_4s_pred/metric.scp
...
```

**Note:** The CLEAN prefix datasets (`prefix_overall_dev_all_{N}s_pred`) must already
have inference outputs before running `reaction_analysis.py`. These are the existing
datasets you have already run inference on.

---

## Step 4 — Verify inference outputs

Check that the score files exist and have the right format:

```bash
INFER_DIR=$RECIPE/exp/universa_train_aruniversa_prefix_full_raw_fs16000_defer_full_metatrue.bak.bak.bak.bak
TAG=inference_defer_full_metatrue_14epoch

# Check distorted prefix outputs
for N in 2 4 6 8; do
    f="${INFER_DIR}/${TAG}/prefix_distorted_overall_dev_${N}s_pred/metric.scp"
    echo -n "${N}s: "
    [ -f "$f" ] && wc -l < "$f" || echo "MISSING"
done

# Check a few utt_ids — should look like dist_fileid_XXXXXX__p002000
head -2 "${INFER_DIR}/${TAG}/prefix_distorted_overall_dev_2s_pred/metric.scp" | cut -d' ' -f1
```

---

## Step 5 — Compute reaction time

```bash
python /path/to/reaction_analysis.py \
  --recipe_dir    $RECIPE \
  --manifest      exp/distortion_manifest.csv \
  --models \
      ANCHOR:exp/universa_train_aruniversa_prefix_full_raw_fs16000_defer_full_metatrue.bak.bak.bak.bak \
      ARECHO:exp/your_arecho_exp_dir \
  --inference_tag inference_defer_full_metatrue_14epoch \
  --clean_test_set_pattern  "prefix_overall_dev_all_{sec}s_pred" \
  --dist_test_set_pattern   "prefix_distorted_overall_dev_{sec}s_pred" \
  --prefix_lengths 2 4 6 8 \
  --metrics        plcmos utmos \
  --threshold      0.3 \
  --out_csv        exp/distortion_results/reaction_time_per_utt.csv \
  --out_summary    exp/distortion_results/reaction_time_summary.csv
```

**Threshold guidance:**
- If `n_triggered` is very low (< 20%), lower threshold to `0.15`
- If almost everything triggers at 2 s, raise threshold to `0.5`
- Check `max_delta_plcmos` in the per-utterance CSV to calibrate

**Console output example:**
```
====================================================================
  Metrics: plcmos, utmos   Threshold: 0.3
====================================================================
  Model           Mean RT(s)  Median RT(s)  Triggered      N
  --------------------------------------------------------------
  ANCHOR               3.421         4.000         78    100
  ARECHO               5.103         6.000         61    100
====================================================================

  ANCHOR reacts 1.682 s earlier on average.
```

---

## Output Files

### `reaction_time_per_utt.csv`
One row per (model × utterance):

| Column | Description |
|--------|-------------|
| `model` | ANCHOR or ARECHO |
| `orig_utt_id` | Original clean utt_id |
| `dist_utt_id` | Distorted utt_id (`dist_` prefix) |
| `reaction_prefix_sec` | Earliest prefix (s) where Δ > threshold; empty if never |
| `ever_triggered` | True / False |
| `triggered_metric` | Which metric crossed threshold first |
| `clean_plcmos_2s` | Clean PLCMOS at 2 s prefix |
| `dist_plcmos_2s` | Distorted PLCMOS at 2 s prefix |
| `delta_plcmos_2s` | \|clean − dist\| at 2 s |
| `max_delta_plcmos` | Max delta across all prefixes |
| *(same for utmos and for 4s, 6s, 8s)* | |

### `reaction_time_summary.csv`
One row per model:

| Column | Description |
|--------|-------------|
| `mean_reaction_sec` | Mean reaction prefix (non-triggered → max_prefix + 1) |
| `median_reaction_sec` | Median reaction prefix |
| `std_reaction_sec` | Standard deviation |
| `n_triggered` | Utterances where Δ > threshold at some prefix |
| `n_never_triggered` | Utterances that never exceeded threshold |
| `pct_react_by_2s` | Fraction reacting at or before 2 s |
| `pct_react_by_4s` | … 4 s, and so on |

---

## Statistical Analysis (downstream)

```python
import pandas as pd
from scipy import stats

df = pd.read_csv("exp/distortion_results/reaction_time_per_utt.csv")
anchor = df[df.model == "ANCHOR"]["reaction_prefix_sec"].dropna()
arecho = df[df.model == "ARECHO"]["reaction_prefix_sec"].dropna()
t_stat, p_val = stats.ttest_ind(anchor, arecho)
print(f"t={t_stat:.3f}  p={p_val:.4f}")
```

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `kaldiio returned None` | Ark path or offset wrong | Check `head -1 dump/raw/overall_dev/wav.scp` matches actual file location |
| `metric.scp not found` for distorted set | Inference not yet run | Complete Step 3 before Step 5 |
| `n_triggered = 0` | Threshold too high, or utt_id mismatch | Check `head -1 <metric.scp>` for the `dist_fileid_X__p002000` pattern |
| `pipe command in wav.scp` | Source wav.scp uses Kaldi pipes | Pre-decode with `utils/copy_data_dir.sh` |
| `src_prefix_dump_dir not found` warning | Clean prefix dump dir missing | Ensure `dump/raw/prefix_overall_dev_all_2s_pred/` exists before Step 2 |

---

## Extending

**More distortions:** Decorate a new function with `@register_distortion("name")` in
`distortion_generator.py`. No other file changes needed.

**More prefix lengths:** Add values to `--prefix_lengths` in all three scripts. New
dump directories and inference runs are created automatically.

**Different distortion time:** Re-run Step 1 with a new `--distortion_time` and a
separate `--audio_out_dir` / `--manifest_out` to avoid overwriting.
