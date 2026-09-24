# ESPnet restoration (rst): Sidon reproduction

An ESPnet reproduction of **Sidon** ([arXiv:2509.17052](https://arxiv.org/abs/2509.17052)).
The model predicts the clean SSL hidden state from degraded speech (stage 1
of the paper) and vocodes it to 48 kHz with a DAC-style decoder that is
either trained here (paper stages 2-3: pretrain on ground-truth features,
finetune on predicted ones) or taken from the official release.

This directory is `rst1` (restoration) rather than `enh1`: the pipeline is not
an `EnhancementTask` model and does not go through the standard `enh.sh`
driver. The feature predictor is trained by `espnet2.bin.rst_train`
(`RestorationTask`), which scores predicted SSL features rather than
waveforms, and the vocoder by `espnet2.bin.rst_vocoder_train`
(`RestorationVocoderTask`); `run.sh` drives the 11 stages directly because
the vocoder pretrain/finetune stages sit in the middle of the pipeline.

Two SSL backbones are supported (`ssl_encoder` in the config), both 1024-d at
50 Hz so the vocoder stages are identical:

| `ssl_encoder` | backbone | layer | weights | licence |
|---|---|---|---|---|
| `w2v_bert2` (default, paper) | w2v-BERT 2.0 | 8 | `facebook/w2v-bert-2.0` | MIT |
| `xeus` | XEUS (ESPnet E-Branchformer SSL, [Chen et al. 2024](https://arxiv.org/abs/2407.00837)) | block 10 | `espnet/xeus`, loaded with `SSLTask.build_model_from_file` | **CC-BY-NC-SA-4.0** (non-commercial) |

Select XEUS with `--config conf/tuning/train_rst_xeus.yaml` for stage 5 and
pass the same `ssl_encoder` / `ssl_encoder_conf` to the vocoder configs (stages
7-8); inference reads the encoder type from the training config.

## Extra dependencies

`peft` and `pyroomacoustics` are the recipe's own extras: `pip install "espnet[rst]"`
(they are also part of `espnet[all]`). VERSA and NISQA are installed separately:

| Package | Needed by | Install |
|---|---|---|
| `peft` | LoRA adapters on the SSL student (stages 4-5) | `pip install "espnet[rst]"` |
| `pyroomacoustics` | RIR pool generation (stage 3) | `pip install "espnet[rst]"` |
| `versa` | VERSA scoring (stage 11) | `tools/installers/install_versa.sh` |
| NISQA *(optional)* | `local/score.py --nisqa_model` | clone [NISQA](https://github.com/gabrielmittag/NISQA), add to `PYTHONPATH` |

`transformers` supplies the w2v-BERT 2.0 backbone, `WavLMForXVector` for
speaker similarity, and `facebook/mms-1b-all` for WER. All three are fetched
from the Hub on first use, so stages 4-10 need network access (or a pre-warmed
`HF_HOME`) on whichever node runs them.

## Data

Set the paths in `db.sh`. `DATASET_LIBRITTS_R` and `LIBRITTS` are mandatory;
`DATASET_EARS` and `DATASET_VCTK_DEMAND` supply the 48 kHz material. Any
`NOISE_*` variable that points at a real directory is added to the noise pool.

## Stages

| Stage | What |
|---|---|
| 1 | Data preparation (`local/data.sh`) |
| 2 | Resample the feature-predictor sets to 16 kHz |
| 3 | Pre-generate the RIR pool (needs `pyroomacoustics`) |
| 4 | Collect feature-predictor statistics |
| 5 | Train the feature predictor |
| 6 | Collect vocoder statistics (48 kHz sets) |
| 7 | Pretrain the vocoder on ground-truth SSL features of clean speech |
| 8 | Finetune the vocoder on the stage-5 predictor's features of degraded speech |
| 9 | Inference with the stage-8 vocoder, or an externally released one (`--external_vocoder`, e.g. the Sidon v0.1 decoder) |
| 10 | Paper's four metrics: DNSMOS, NISQA, SpkSim, WER (`local/score.py`, dependency-light) |
| 11 | VERSA scoring, reference-free and reference-based (recommended: same metrics plus UTMOS, SQUIM, PESQ, STOI, SDR/SI-SNR and more in one pass) |

```bash
./run.sh --stage 1 --stop_stage 8 --ngpu 4 --nj 64     # predictor + vocoder
./run.sh --stage 9 --stop_stage 11                      # uses exp/rst_vocoder_dac_finetune
# or skip vocoder training and use the official decoder
./run.sh --stage 9 --stop_stage 11 --external_vocoder /path/to/decoder_cuda.pt
```

## Vocoder

Stages 7-8 train the same DAC decoder as the official release (52.4M
parameters, strides 8-5-4-3-2 = 960x, one 20 ms feature frame to 960 samples
at 48 kHz) with ESPnet's `GANTrainer`, the multi-period + multi-band STFT
discriminator from `espnet2.gan_codec` and the mel / adversarial / feature
matching losses from `espnet2.gan_tts.hifigan` (weights 15 / 2 / 1, summed over
sub-discriminators as in DAC). The encoder is frozen in both stages; the
generator and discriminators run on a 1 s excerpt whose features were computed
with 8 s of context (`segment_duration`, `context_duration`).

Four vocoders share the stage-7/8 data path and the inference loader; the
config's `vocoder_type` selects one and `vocoder_conf` configures it.

| `vocoder_type` | Model | Objective | Configs (stage 7 / 8) |
|---|---|---|---|
| `dac` (default) | DAC decoder as in the official release, 52.4M | GAN, two optimizers | `train_rst_vocoder_dac_{pretrain,finetune}.yaml` |
| `hifigan` | ESPnet `HiFiGANGenerator`, 512 channels, 17M | GAN, two optimizers | `train_rst_vocoder_hifigan_{pretrain,finetune}.yaml` |
| `cfm` | WaveNet velocity field (`espnet2.gan_tts.wavenet`) behind a DAC-style 960x conditioner, 5M | conditional flow matching, one optimizer | `train_rst_vocoder_cfm_{pretrain,finetune}.yaml` |
| `periodwave` | PeriodWave (Lee et al., ICLR 2025): multi-period vector-field estimator over periods 1-7 with a ConvNeXt-V2 conditioner, 34.8M | conditional flow matching, one optimizer | `train_rst_vocoder_periodwave_{pretrain,finetune}.yaml` |

One command trains all four: `rst_vocoder_train` reads `vocoder_type` and runs
the GAN types with `GANTrainer` (`RestorationVocoderTask`) and the flow types
with the plain `Trainer` (`RestorationFlowVocoderTask`, same arguments, in
`espnet2/tasks/rst_vocoder.py`). `run.sh` reads the same key to pick the
checkpoint stage 8 starts from and stage 9 decodes with
(`valid.loss_mel.best.pth` for the GAN types, `valid.loss.best.pth` for the
flow types) and to skip the discriminator initialisation.

The flow-matching vocoders train without a discriminator on straight
noise-to-waveform paths (velocity regression, `sigma_min` 1e-4) and synthesise
with a midpoint ODE solver in `num_steps` steps (16 by default, two velocity
evaluations per step; `--vocoder_num_steps` at inference overrides it). `cfm`
is our own WaveNet velocity field. `periodwave` is the published comparison,
the model of https://github.com/sh-lee-prml/PeriodWave (MIT, Copyright (c)
2024 Sang-Hoon Lee) vendored into `espnet2/rst/decoder/periodwave_vocoder.py`,
whose header names every upstream file it derives from and carries the
licence. Its EnCodec variant is the one vendored: that variant conditions on a
latent sequence rather than a mel spectrogram and drops PriorGrad's energy
prior, which cannot be computed from predicted features at inference time. The
one adaptation is resolution: the folded U-Net takes conditioning at a 64th of
the sample rate, so 50 Hz features against 48 kHz must be lifted by 15, done as
5 then 3 (`cond_rates`). Its loss keeps PeriodWave's own source-noise scale
(`noise_scale` 0.25), so its absolute values are not comparable with `cfm`.

All models in this recipe are trained from scratch (the SSL backbone is the
public w2v-BERT 2.0; the LoRA adapter, the vocoder and its discriminator start
from random initialisation, and stage 8 starts from the recipe's own stage-7
checkpoint). The published Sidon weights are never used for training. They can
be run through the same inference and scoring path for comparison:
`local/convert_official_sidon.py` and `local/convert_official_sidon_vocoder.py`
convert the released adapter and the released TorchScript vocoder into
checkpoints that stage 9 loads like a trained one.

### Results

Reference-free and reference-based scores of the from-scratch pipeline (stage-5
w2v-BERT 2.0 predictor at epoch 30 in every row) on 120 clips: the 20
LibriTTS-R test-clean utterances of the DialogueSidon evaluation set under six
conditions (clean, band-limit to 3.6 kHz, clipping, MP3 at 16 kb/s,
reverberation with T60 0.3 s and 1.5 s), scored with VERSA (stage 11) at
48 kHz: UTMOS, DNSMOS, speaker similarity to the clean reference, PESQ and
STOI against the clean reference, and whisper large-v3 word error rate. Means
over the six conditions. Every checkpoint is the last one of its run, not the
loss-selected one (see the note below).

| vocoder | stage | epochs | UTMOS | DNSMOS | SpkSim | PESQ | STOI | WER |
|---|---|---|---|---|---|---|---|---|
| released Sidon decoder on our predictor | - | - | 4.09 | 3.42 | 0.69 | 2.46 | 0.90 | 3.0 % |
| `dac` | 7 | 80 | 3.72 | 3.23 | 0.67 | 2.15 | 0.89 | 3.3 % |
| `dac` | 8 | 48 | 3.71 | 3.19 | 0.67 | 2.12 | 0.88 | 3.0 % |
| `hifigan` | 7 | 38 | 3.57 | 3.22 | 0.64 | 1.95 | 0.88 | 3.3 % |
| `hifigan` | 8 | 13 | 3.42 | 3.08 | 0.63 | 1.95 | 0.87 | 3.3 % |
| `periodwave` | 7 | 38 | 2.80 | 2.86 | 0.45 | 1.62 | 0.80 | 3.7 % |
| `periodwave` | 8 | 12 | 2.01 | 2.58 | 0.41 | 1.41 | 0.76 | 4.8 % |

Budgets: `dac` stage 7 ran 80 epochs on 1-4 A40s (about 1-2 GPU-hours per
epoch), `hifigan` 38 epochs at 1 GPU-hour per epoch, `periodwave` 38 epochs at
about 1.6 GPU-hours per epoch (roughly 120k optimizer steps, an order of
magnitude below the published PeriodWave recipe, and its scores were still
rising at the last checkpoint: UTMOS 2.28 at epoch 14, 2.47 at 18, 2.56 at 27,
2.80 at 38). The `cfm` vocoder was trained for two epochs only and is not
listed.

Two things to know before reading the table:

- Stage 8 selects checkpoints by `valid.loss_mel.best.pth`, but that loss is
  measured against the clean target while the input features are predicted
  from degraded speech, so it jumps at the first finetuning epoch and never
  recovers; the loss-best pin is epoch 1. Pick stage-8 checkpoints by stage 11
  scores, as above. At these budgets stage 8 is not ahead of stage 7 on UTMOS
  or DNSMOS for any vocoder; for the GAN vocoders it trades a little of both
  for PESQ and WER, and for `periodwave` it is behind on every metric after 12
  epochs.
- Sampler steps are not the flow vocoders' bottleneck: `periodwave` at epoch
  18 scored UTMOS 2.47 with 16 steps, 2.54 with 32 and 2.51 with 64.

## Configs

`conf/train.yaml` and `conf/decode.yaml` are the defaults used by `run.sh`;
`conf/train.yaml` is a symlink to the variant under `conf/tuning/`.

## Notes

The RIR pool is generated ahead of training rather than simulated on the fly.
On-the-fly `pyroomacoustics` simulation is CPU-bound and starves the GPUs; with
a pre-generated pool the dataloader keeps 4x A40 at ~98% SM occupancy
(`iter_time` ~1e-4 s per step).

## Acknowledgements

This recipe reproduces **Sidon** (W. Nakata, Y. Saito, Y. Ueda, H. Saruwatari,
"Sidon: Fast and Robust Open-Source Multilingual Speech Restoration for
Large-Scale Dataset Cleansing", arXiv:2509.17052). The code was written for
ESPnet on ESPnet's own task, trainer and GAN infrastructure and copies no file
from the Sidon repository, but four parts are derived from that implementation
(https://github.com/sarulab-speech/Sidon, Copyright (c) 2025 sarulab-speech, MIT License) rather than from the
paper alone, and each file says so in its header:

- the online degradation pipeline in `espnet2/tasks/rst.py`
  (`src/sidon/data/preprocess/degrations.py`, `functional_degrations.py`):
  the six-step order with independent p=0.5 draws, SNR ~ U(-5, 20) dB, the
  band-limit sampling rates, quantile clipping bounds U(0, 0.1) / U(0.9, 1.0)
  and MP3 `qscale` 1-10. Differences: packet loss follows the paper (9 %,
  20-200 ms segments); reverberation uses a pre-generated RIR pool
  (`local/prepare_rir_pool.py`) instead of on-the-fly simulation;
- the room-impulse-response simulation recipe in `local/prepare_rir_pool.py`
  (`functional_degrations.py: convolve_rir_pra`);
- the LoRA adapter configuration (rank 64, alpha 16, dropout 0.1,
  `bias="lora_only"`, target `output_dense`) and the frozen-teacher /
  LoRA-student setup on the first 8 layers of w2v-BERT 2.0 in
  `espnet2/rst/rst_model.py` (`src/sidon/model/sidon/lightning_module.py`);
- the generator loss weighting (mel 15, adversarial 2, feature matching 1) and
  the summed LSGAN / feature-matching aggregation in
  `espnet2/rst/rst_vocoder_model.py` (`src/sidon/model/losses.py`,
  `config/model/sidon_vocoder_pretrain.yaml`); the loss modules are ESPnet's.

`espnet2/rst/decoder/dac_vocoder.py` reproduces the decoder of the Descript
Audio Codec (R. Kumar et al., NeurIPS 2023; `descript-audio-codec`,
Copyright (c) 2023-present Descript, Inc., MIT License) so that the recipe
does not depend on the `dac` package and the released Sidon vocoder can be
loaded into it. `local/convert_official_sidon*.py` only rename keys of the
released weights (`sarulab-speech/sidon_raw_weight`, `sarulab-speech/sidon-v0.1`,
MIT).

MIT License text applying to the derived parts above:

```
Copyright (c) 2025 sarulab-speech
Copyright (c) 2023-present Descript, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
