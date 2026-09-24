import pytest
import torch
from torch import nn

from espnet2.rst.decoder.flow_vocoder import FlowVocoder
from espnet2.rst.decoder.periodwave_vocoder import PeriodWaveVocoder
from espnet2.rst.rst_flow_vocoder_model import ESPnetRestorationFlowVocoderModel


class DummyEncoder(nn.Module):
    def __init__(self, dim=8, input_sr=16000):
        super().__init__()
        self.input_sr = input_sr
        self._dim = dim
        self.proj = nn.Linear(320, dim)

    @property
    def ssl_dim(self):
        return self._dim

    def _wav_to_ssl_inputs(self, wav, lengths):
        frames = wav.size(1) // 320
        x = wav[:, : frames * 320].reshape(wav.size(0), frames, 320)
        mask = (torch.arange(frames)[None] < (lengths // 320)[:, None]).long()
        return {"x": x, "attention_mask": mask}

    def encode(self, inputs, teacher=False):
        feat = self.proj(inputs["x"])
        return (feat.detach() if teacher else feat), inputs["attention_mask"]


def tiny_flow_vocoder():
    return FlowVocoder(
        input_dim=8,
        cond_channels=4,
        cond_hidden=16,
        rates=[8, 5, 4, 3, 2],
        layers=2,
        stacks=1,
        residual_channels=4,
        gate_channels=8,
        skip_channels=4,
        time_channels=8,
        num_steps=2,
    )


def tiny_periodwave():
    return PeriodWaveVocoder(
        input_dim=8,
        periods=(1, 2),
        hidden_dim=16,
        final_dim=1,
        cond_rates=(5, 3),
        num_steps=2,
    )


def make_model(vocoder=None, use_predicted_feat=False):
    return ESPnetRestorationFlowVocoderModel(
        ssl_encoder=DummyEncoder(),
        vocoder=vocoder or tiny_flow_vocoder(),
        use_predicted_feat=use_predicted_feat,
        segment_duration=0.1,
    )


def test_upsample_factor_must_match_output_rate():
    with pytest.raises(ValueError):
        ESPnetRestorationFlowVocoderModel(
            ssl_encoder=DummyEncoder(),
            vocoder=FlowVocoder(
                input_dim=8,
                cond_hidden=16,
                cond_channels=4,
                rates=[2, 2],
                layers=2,
                stacks=1,
            ),
        )


@pytest.mark.parametrize("vocoder", [tiny_flow_vocoder, tiny_periodwave])
def test_flow_matching_step(vocoder):
    model = make_model(vocoder()).train()
    assert not model.ssl_encoder.training  # frozen encoder stays in eval
    speech = torch.randn(2, 24000) * 0.1  # 0.5 s at 48 kHz
    lengths = torch.tensor([24000, 19200])
    loss, stats, weight = model(
        speech_ref1=speech,
        speech_ref1_lengths=lengths,
        vocoder_crop_start=torch.tensor([0, 3]),
    )
    # force_gatherable returns the batch loss as a (1,) tensor
    assert loss.numel() == 1 and torch.isfinite(loss).all()
    assert torch.isfinite(stats["loss"]).all() and int(weight) == 2
    loss.backward()
    assert all(p.grad is not None for p in model.vocoder.parameters())
    assert all(p.grad is None for p in model.ssl_encoder.parameters())


def test_crop_start_defaults_to_zero():
    model = make_model()
    speech = torch.randn(1, 9600) * 0.1
    loss, _, _ = model(speech_ref1=speech, speech_ref1_lengths=torch.tensor([9600]))
    assert torch.isfinite(loss).all()


def test_predicted_features_need_noisy_speech():
    model = make_model(use_predicted_feat=True)
    speech = torch.randn(2, 24000) * 0.1
    lengths = torch.tensor([24000, 24000])
    with pytest.raises(ValueError):
        model(speech_ref1=speech, speech_ref1_lengths=lengths)
    loss, _, _ = model(
        speech_ref1=speech,
        speech_ref1_lengths=lengths,
        noisy_speech=torch.randn(2, 8000) * 0.1,
        noisy_speech_lengths=torch.tensor([8000, 8000]),
    )
    assert torch.isfinite(loss).all()
