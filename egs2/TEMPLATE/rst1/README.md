# Speech restoration (rst)

`rst.sh` is the shared driver for the restoration recipes: a frozen SSL
encoder with a LoRA feature predictor (stages 4-5) followed by a vocoder trained
on those features (stages 6-8), with inference, scoring, results, packing and
Hugging Face upload behind them. It does not go through `enh.sh` because the
predictor is scored on SSL features rather than waveforms and the vocoder
stages sit in the middle of the pipeline.

| Stage | What |
|---|---|
| 1 | `local/data.sh`: `data/{train,dev}_fp` (predictor sets), `data/{train,dev}_voc` (48 kHz vocoder sets), `data/<test_set>` (with `text`), `data/noise_pool` |
| 2 | Resample the predictor and test sets to 16 kHz |
| 3 | Simulate the RIR pool (`--rir_pool_size`, needs `pyroomacoustics`) |
| 4-5 | Predictor statistics and training (`rst_train`) |
| 6-8 | Vocoder statistics, pretraining on ground-truth features, finetuning on predicted features (`rst_vocoder_train`; the config's `vocoder_type` picks dac, hifigan, cfm or periodwave) |
| 9 | Inference (`rst_inference`), or with `--external_vocoder` a released TorchScript vocoder |
| 10 | Recipe-local scoring (`local/score.py`, optional) |
| 11 | VERSA scoring, reference-free and, with `--ref_wav_scp`, reference-based |
| 12 | `RESULTS.md` from the stage-11 averages |
| 13 | Pack predictor + vocoder (`--skip_packing false`) |
| 14 | Upload to Hugging Face (`--skip_upload_hf false --hf_repo user/name`) |

The reference recipe, with the model description, the results and the
acknowledgements, is `egs2/libritts_r/rst1/README.md`. `egs2/mini_an4/rst1` is
the CPU integration test (a tiny stand-in for the SSL encoder, a few seconds of
audio, every stage up to inference and packing). Start a new recipe with
`egs2/TEMPLATE/rst1/setup.sh egs2/<corpus>/rst1`.
