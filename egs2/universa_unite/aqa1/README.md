# Prefix audio quality assessment on universa_unite

This recipe trains an autoregressive quality model that listens to the first
2, 4, 6 or 8 s of a clip and, in one decoded sequence, rates the audio heard so
far and forecasts the metrics of the whole clip (`<metric>_full`). This is the
ANCHOR method ([Tao, Shi, Shim and Watanabe, 2026](https://arxiv.org/abs/2606.10233)).
The model is ARECHO (`universa: ar_universa`, see the
[AQA guide](../../TEMPLATE/aqa1/README.md#arecho-and-legacy-models)): frozen
WavLM Large features, a 4-block Transformer encoder and a 12-block metric
decoder over metric and value tokens, with sinusoidal positions. The
configuration name gives the architecture: `conf/tuning/train_ar_prefix.yaml`
(linked as `conf/train.yaml`) and `conf/tuning/decode_ar_expected.yaml`
(`conf/decode.yaml`).

## Released model

> **TODO:** Hugging Face link of the released model (epoch 25). Whether it can
> be released is not decided yet; see the data licences below.

The package holds `25epoch.pth` and a `config.yaml` in which the metric names,
types and token info (2.8 MB) are embedded, so inference needs no other file:

```python
from espnet2.bin.aqa_inference import AqaInference

predict = AqaInference.from_pretrained(
    "TODO/model_id", readout="expected", value_token_range="legacy",
    silence_reference_sec=1.0,
    fixed_metric_name_order="utmos plcmos utmos_full plcmos_full",
)
scores = predict(prefix)  # mono float32 audio at 16 kHz, e.g. the first 2 s
```

The token info is not in this recipe: it is large and belongs to the weights,
since its token IDs index the decoder's embedding and output rows.

The model was trained with the original UniVERSA tokenizer, which encodes a
categorical label like a number: by the bin of the first label that sorts after
it. All of the model's label lists are sorted, so this gives label `i` the token
`<metric>@<i + 1>`, as ESPnet does, except for the last label, which gets the
token of the label before it. The checkpoint therefore decodes with ESPnet as
is; its token info's `categorical_scheme` key is ignored. Four clips, each
decoded whole and as a 2 s prefix, gave identical tokens, labels and scores
with the original tokenizer and with ESPnet's. In the 22 metrics below (and
their `_full` twins), the last label shares the token of the label before it
and is never predicted; the table lists these two labels.

| Metric (and its `_full` twin) | Predicted / never predicted |
| --- | --- |
| `language` | `<zho>` / `<zul>` |
| `real_language` | `zh` / `zul` |
| `qwen_speaker_gender` | `Multiple speakers with mixed genders` / `Non-binary/unclear` |
| `qwen_speaker_age` | `Teen` / `Young adult` |
| `qwen_speech_impairment` | `Other impairment` / `Stuttering/disfluency` |
| `qwen_voice_pitch` | `Low` / `Very high` |
| `qwen_pitch_range` | `Narrow range` / `Wide range` |
| `qwen_voice_type` | `Tremulous` / `Whispered` |
| `qwen_speech_volume_level` | `Variable` / `Very quiet` |
| `qwen_language` | `Russian` / `Spanish` |
| `qwen_speech_register` | `Standard register` / `Technical register` |
| `qwen_vocabulary_complexity` | `General` / `Technical` |
| `qwen_speech_purpose` | `Narrative` / `Persuasive` |
| `qwen_speech_emotion` | `Sad` / `Surprised` |
| `qwen_speech_clarity` | `Low clarity` / `Medium clarity` |
| `qwen_speech_rate` | `Slow` / `Very slow` |
| `qwen_speaking_style` | `Hesitant` / `Professional` |
| `qwen_laughter_crying` | `Contains other emotional sounds` / `No laughter or crying` |
| `qwen_speech_background_environment` | `Outdoor urban` / `Quiet indoor` |
| `qwen_recording_quality` | `Poor` / `Professional` |
| `qwen_channel_type` | `Surveillance/hidden mic` / `Telephone/VoIP` |
| `rir_room_size` | `none` / `small` |

The model had no whole-clip targets for the 18 `qwen_*_full` metrics other than
`qwen_language_full`: all their value tokens have the same weights, so their
output is arbitrary. `conf/decode.yaml` does not request them.

## Stages

`./run.sh` calls the shared [aqa.sh](../../TEMPLATE/aqa1/aqa.sh). Stage 1 runs
`local/data.sh` (its own stages below); stages 2-9 format audio, filter by
length, collect statistics, train, decode and score each test set with
`pyscripts/utils/aqa_eval.py`. Pass data options as
`./run.sh --local_data_opts "--token_info /path/to/tokens.json"`.

| `local/data.sh` stage | What it does |
| --- | --- |
| 1 | Resamples `data/<split>_whole/wav.scp` (and `ref_wav.scp`) to 16 kHz FLAC. |
| 2 | Cuts the 2/4/6/8 s prefixes of every clip (and of its reference) with `pyscripts/utils/make_prefix_wav_scp.py`; a prefix must be shorter than its clip. Keys are `<utt>__p<ms>`, e.g. `utt1__p002000`. |
| 3 | Labels whole clips (`--whole_versa_config`) and prefixes (`--versa_config`) with VERSA through `scripts/utils/versa_eval.sh`; reference-based metrics only where a reference exists. |
| 4 | Merges VERSA labels with `data/<split>_whole/metric.scp`, gives every prefix row its clip's labels as `<metric>_full` (`pyscripts/utils/add_full_metric_targets.py`) and writes `data/<split>` with whole and prefix rows, each with a 1 s silent reference clip. |
| 5 | Copies `--token_info` to `data/token_info.json`, appending new numerical metrics (and, with `--init_model`, decoder rows) with `pyscripts/utils/extend_metric_vocab.py`; keeps only labels the token info encodes and writes `data/metric2id`. |

Prepare `data/<split>_whole/` for `train`, `valid` and `dev` (option
`--splits`): a `wav.scp` of whole clips at any rate, optionally a
`ref_wav.scp` of time-aligned clean references for some clips, and optionally a
`metric.scp` of labels from elsewhere (listening-test MOS, simulation settings,
transcripts' error rates, labels of the whole clip from other tools). Its rows
keyed `<utt>` label the clip; rows keyed `<utt>__p<ms>` label a prefix.

Stage 3 needs [VERSA](https://github.com/wavlab-speech/versa) at revision
`563a47f`, which labelled the released model's data (later revisions renamed
some modules and outputs), and GPUs: VERSA runs `--nj` jobs on the current
host. `conf/versa_so_far.yaml` gives the 15 no-reference and 13
reference-based values labelled on prefixes (`--drop_metrics` removes the
`sir` and `sar` outputs of `signal_metric`). Set its NISQA `model_path` to the
weights from VERSA's `tools/setup_nisqa.sh`. ARECHO labelled whole clips with
VERSA's `egs/universa_prepare/universa_prepare.yaml`; pass it as
`--whole_versa_config` for that larger set. Audio is scored at 16 kHz, the rate
the model hears; at other rates VERSA `563a47f` rates PLCMOS and SingMOS at the
wrong rate when several `pseudo_mos` predictors are listed.

## Data

The released model was trained on 1,407,904 rows (whole clips and their
prefixes) from the sources below. The ARECHO part has the whole-clip labels of
ARECHO's data (VERSA `universa_prepare`; rate-dependent ones recomputed from
the 16 kHz audio for NISQA and simulated clips) and 15 VERSA "so far" labels on
every prefix, plus 13 reference-based ones on simulated clips. The expansion
part has the 15 VERSA labels on whole clips and prefixes. Both parts also carry
five attributes (below).

| Source | Part | How to obtain | Licence |
| --- | --- | --- | --- |
| CoVoST 2 / Common Voice, AISHELL-1, AMI (headset) | ARECHO | OWSM data preparation: `egs2/owsm_v1/s2t1` (CoVoST 2, AISHELL-1), `egs2/owsm_v3/s2t1` (AMI) | Common Voice CC0; AISHELL-1 Apache 2.0; AMI CC BY 4.0 |
| BABEL | ARECHO | LDC releases; set `BABEL_<id>` in `db.sh` and run `egs2/owsm_v3/s2t1/local/prepare_babel.sh` | LDC licence |
| URGENT 2024 simulation (`fileid_*`, with clean references) | ARECHO | Simulate with the [URGENT 2024 challenge](https://github.com/urgent-challenge/urgent2024_challenge) scripts | Per source corpus |
| URGENT 2024 blind-test system outputs (with listener MOS) | ARECHO | `blind_test` / `blind_test_mos` of [urgent2024-sqa](https://huggingface.co/datasets/urgent-challenge/urgent2024-sqa) | CC BY-NC-SA 4.0 |
| NISQA Corpus (with MOS) | ARECHO | [NISQA](https://github.com/gabrielmittag/NISQA) | Non-commercial research; per sub-corpus |
| VoiceMOS 2022 BVCC (with MOS) | ARECHO | Zenodo record of the VoiceMOS Challenge 2022 | Terms of the record |
| URGENT 2024 / 2025 validation and non-blind test outputs; URGENT 2025 blind-test MOS | expansion | [urgent2024-sqa](https://huggingface.co/datasets/urgent-challenge/urgent2024-sqa), [urgent2025-sqa](https://huggingface.co/datasets/urgent-challenge/urgent2025-sqa) | CC BY-NC-SA 4.0 |
| SOMOS, TMHINT-QI, Tencent (with MOS), TTSDS2 (unrated clips), CHiME-7 UDASE | expansion | [URGENT 2026 track 2 pack](https://huggingface.co/datasets/urgent-challenge/urgent26_track2_sqa) | SOMOS CC BY-NC-SA 4.0; TTSDS2 MIT; CHiME-7 non-commercial (CHiME-5); TMHINT-QI and Tencent unverified |
| MLAAD (subset) | expansion | [mueller91/MLAAD](https://huggingface.co/datasets/mueller91/MLAAD) (gated) | CC BY-NC 4.0 |
| CodecFake+ CoRS (25 % of VCTK utterances) and CoSG | expansion | [CodecFake/CodecFake_Plus_Dataset](https://huggingface.co/datasets/CodecFake/CodecFake_Plus_Dataset) | MIT; VCTK CC BY 4.0 |
| PSTN calls (with MOS) | expansion | Microsoft PSTN corpus, `train.zip` at `challenge.blob.core.windows.net/pstn` | Unclear (no licence file) |
| EARS, Expresso | expansion | [EARS](https://github.com/facebookresearch/ears_dataset) (`EARS` in `db.sh`), [Expresso](https://speechbot.github.io/expresso/) | CC BY-NC 4.0 |

Listener MOS is a target per corpus: `nisqa_real_mos`, `voicemos_real_mos`,
`urgent_mos`, `somos_real_mos`, `tmhint_real_mos`, `pstn_real_mos`,
`tencent_real_mos` and `urgent25_real_mos`, each with its `_full` twin on
prefixes. Several sources are non-commercial, so a model trained on all of them
is for research use.

This recipe does not rebuild these from the release:

- ARECHO's selection of clips, its train/valid/dev/test split, its simulation
  run and its whole-clip labels are not public; rebuild them with the sources
  and VERSA configuration above.
- The five attributes `f0_median`, `f0_std` (PENN), `snr_est` (Brouhaha),
  `speech_ratio` (Silero VAD) and `bg_event` (PANNs AudioSet tags grouped into
  events) came from scripts outside VERSA. Supply them, if wanted, in
  `data/<split>_whole/metric.scp` for clips and prefixes; `bg_event` uses the
  token info's label list.
- The 500 h expansion mix (domain shares, per-system caps), the listener-MOS
  splits and the removal of clips that overlap the dev and test sets were made
  with scripts outside this recipe.
- Training started from an unreleased ARECHO checkpoint (epoch 3 of a run on
  ARECHO's scale data) whose decoder was extended to the token info.
  `espnet/arecho_base_v0.1-large-decoder` has the same architecture; extend it
  with `--init_model` and pass `--train_args "--init_param data/init_model.pth"`.

## Inference settings

`conf/decode.yaml` reproduces the released model's results:
`--readout expected` reports E[X] over each numerical metric's value tokens,
`--value_token_range legacy` uses the original UniVERSA value-token range,
`--defer_full_meta true` (the training config's setting, which inference
follows by default) decodes `<metric>_full` after the prefix metrics, and
`--silence_reference_sec 1.0` feeds the silent reference clip it was trained
with when none is given. `fixed_metric_name_order` decodes the 20 no-reference
"so far" metrics and the model's 73 `_full` metrics; adapt it to another metric
vocabulary or set it to `""` to decode every metric. ARECHO decodes one clip
per batch.

## Results

Epoch 25 of the released model on the universa_unite dev and test sets, 2-8 s
prefixes, decoded with `conf/decode.yaml`. `_full` predictions are scored
against the clip's labels (VERSA metrics, plus NISQA listener MOS on
`generation_test`), so-far predictions against VERSA labels of the prefix;
rate-dependent labels were recomputed from the 16 kHz audio. nMAE (MAE divided by
the label's standard deviation) and PCC are averaged over the numerical metrics
that have labels. "Unseen" leaves out clips that were in the training data of
the ARECHO checkpoint the model started from; on `generation_test` that leaves
the BVCC clips.

Whole-clip forecast (`<metric>_full`), nMAE / PCC:

| Set | Metrics | Prefixes at 2 s | 2 s | 4 s | 6 s | 8 s |
| --- | --- | --- | --- | --- | --- | --- |
| dev | 57 | 8,473 | 0.429 / 0.775 | 0.374 / 0.821 | 0.364 / 0.839 | 0.367 / 0.845 |
| dev, unseen | 57 | 6,167 | 0.468 / 0.749 | 0.403 / 0.811 | 0.392 / 0.828 | 0.387 / 0.839 |
| generation_test | 23 | 1,868 | 0.413 / 0.826 | 0.363 / 0.862 | 0.335 / 0.883 | 0.328 / 0.890 |
| generation_test, unseen (BVCC) | 22 | 875 | 0.530 / 0.730 | 0.491 / 0.724 | 0.484 / 0.724 | 0.510 / 0.727 |
| enhancement_simu_test | 55 | 2,760 | 0.483 / 0.741 | 0.418 / 0.798 | 0.398 / 0.811 | 0.394 / 0.822 |
| enhancement_simu_test, unseen | 55 | 477 | 0.485 / 0.757 | 0.423 / 0.807 | 0.405 / 0.822 | 0.401 / 0.837 |
| enhancement_real_test | 36 | 2,850 | 0.452 / 0.768 | 0.384 / 0.824 | 0.374 / 0.836 | 0.420 / 0.838 |

So far (UTMOS, PLCMOS and the five NISQA scores of the prefix), nMAE / PCC:

| Set | 2 s | 4 s | 6 s | 8 s |
| --- | --- | --- | --- | --- |
| dev | 0.318 / 0.890 | 0.281 / 0.911 | 0.284 / 0.910 | 0.272 / 0.917 |
| dev, unseen | 0.353 / 0.863 | 0.312 / 0.889 | 0.311 / 0.891 | 0.291 / 0.905 |
| generation_test | 0.402 / 0.830 | 0.351 / 0.874 | 0.329 / 0.893 | 0.317 / 0.903 |
| generation_test, unseen (BVCC) | 0.500 / 0.722 | 0.496 / 0.730 | 0.540 / 0.685 | 0.559 / 0.694 |
| enhancement_simu_test | 0.399 / 0.839 | 0.380 / 0.858 | 0.369 / 0.866 | 0.357 / 0.868 |
| enhancement_simu_test, unseen | 0.402 / 0.836 | 0.390 / 0.853 | 0.383 / 0.851 | 0.379 / 0.861 |
| enhancement_real_test | 0.329 / 0.877 | 0.306 / 0.894 | 0.286 / 0.906 | 0.280 / 0.917 |

dev holds URGENT 2024 blind-test outputs and simulated clips;
`enhancement_simu_test` simulated clips; `enhancement_real_test` URGENT 2024
blind-test outputs; `generation_test` NISQA and BVCC clips.

## Notes

- The released model's original config also set `embedding_dim: 256`,
  `use_rope: true` and `sym_sos` / `sym_eos`. Its code ignored them (sinusoidal
  positions, `embedding_size` 512), so `conf/tuning/train_ar_prefix.yaml` leaves
  them out; it loads the released weights with `strict=True`.
- Training used 4 GPUs (`--ngpu 4`), 25 epochs of 7,794 iterations and AMP.
