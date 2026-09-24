import pytest
import torch

from espnet2.rst.decoder.dac_vocoder import (
    FLOW_VOCODER_TYPES,
    VOCODERS,
    build_vocoder,
)
from espnet2.rst.decoder.flow_vocoder import FlowVocoder
from espnet2.rst.decoder.periodwave_vocoder import PeriodWaveVocoder


def tiny_flow(**kwargs):
    conf = dict(
        input_dim=8,
        cond_channels=4,
        cond_hidden=16,
        rates=[2, 2],
        layers=2,
        stacks=1,
        residual_channels=4,
        gate_channels=8,
        skip_channels=4,
        time_channels=8,
        num_steps=2,
    )
    conf.update(kwargs)
    return FlowVocoder(**conf)


def tiny_periodwave(**kwargs):
    conf = dict(
        input_dim=8,
        periods=(1, 2),
        hidden_dim=16,
        final_dim=1,
        cond_rates=(2, 2),
        num_steps=2,
    )
    conf.update(kwargs)
    return PeriodWaveVocoder(**conf)


def test_flow_vocoder_shapes():
    vocoder = tiny_flow()
    assert vocoder.upsample_factor == 4
    feat = torch.randn(2, 5, 8)
    cond = vocoder.condition(feat)
    assert cond.shape == (2, 4, 20)
    x = torch.randn(2, 1, 20)
    assert vocoder.velocity(x, torch.rand(2), cond).shape == (2, 1, 20)
    wav = vocoder.generate(feat)
    assert wav.shape == (2, 20)
    assert wav.abs().max() <= 1.0
    assert vocoder(feat.transpose(1, 2)).shape == (2, 1, 20)
    assert vocoder.generate(feat, num_steps=1).shape == (2, 20)


def test_flow_vocoder_starts_from_zero_velocity():
    # the last convolution is zero-initialised, so v = 0 before training
    vocoder = tiny_flow()
    feat = torch.randn(1, 3, 8)
    v = vocoder.velocity(torch.randn(1, 1, 12), torch.rand(1), vocoder.condition(feat))
    torch.testing.assert_close(v, torch.zeros_like(v))


def test_flow_vocoder_remove_weight_norm_keeps_velocity():
    vocoder = tiny_flow().eval()
    for p in vocoder.parameters():  # move off the zero init so the check bites
        p.data.normal_(0, 0.1)
    feat, x, t = torch.randn(1, 3, 8), torch.randn(1, 1, 12), torch.rand(1)
    with torch.no_grad():
        before = vocoder.velocity(x, t, vocoder.condition(feat))
        vocoder.remove_weight_norm()
        after = vocoder.velocity(x, t, vocoder.condition(feat))
        vocoder.remove_weight_norm()  # idempotent
        again = vocoder.velocity(x, t, vocoder.condition(feat))
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(after, again)
    assert not any(n.endswith("weight_g") for n, _ in vocoder.named_parameters())


def test_periodwave_shapes_and_loss():
    vocoder = tiny_periodwave()
    assert vocoder.upsample_factor == 64 * 4
    feat = torch.randn(2, 3, 8)
    cond = vocoder.condition(feat)
    assert len(cond) == 2  # one conditioning tensor per period
    n = 3 * vocoder.upsample_factor
    x = torch.randn(2, 1, n)
    assert vocoder.velocity(x, torch.rand(2), cond).shape == (2, 1, n)
    loss = vocoder.flow_loss(x, feat)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in vocoder.parameters() if p.requires_grad)
    wav = vocoder.generate(feat)
    assert wav.shape == (2, n)
    assert wav.abs().max() <= 1.0
    assert vocoder.generate(feat, num_steps=1, solver="euler").shape == (2, n)
    assert vocoder(feat.transpose(1, 2)).shape == (2, 1, n)


def test_periodwave_rejects_bad_geometry():
    with pytest.raises(ValueError):
        tiny_periodwave(final_dim=2)  # must be hidden_dim // 16
    with pytest.raises(ValueError):
        tiny_periodwave(solver="rk4")


def test_periodwave_remove_weight_norm_keeps_velocity():
    vocoder = tiny_periodwave().eval()
    feat = torch.randn(1, 3, 8)
    x = torch.randn(1, 1, 3 * vocoder.upsample_factor)
    t = torch.rand(1)
    with torch.no_grad():
        before = vocoder.velocity(x, t, vocoder.condition(feat))
        vocoder.remove_weight_norm()
        after = vocoder.velocity(x, t, vocoder.condition(feat))
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-4)
    assert not any(n.endswith("weight_g") for n, _ in vocoder.named_parameters())


def test_registry_has_flow_types():
    assert set(VOCODERS) == {"dac", "hifigan", "cfm", "periodwave"}
    assert FLOW_VOCODER_TYPES == {"cfm", "periodwave"}
    cfm = build_vocoder(
        "cfm",
        8,
        {
            "cond_channels": 4,
            "cond_hidden": 16,
            "rates": [2, 2],
            "layers": 2,
            "stacks": 1,
            "residual_channels": 4,
            "gate_channels": 8,
            "skip_channels": 4,
            "time_channels": 8,
        },
    )
    assert isinstance(cfm, FlowVocoder)
    periodwave = build_vocoder(
        "periodwave",
        8,
        {"periods": [1, 2], "hidden_dim": 16, "final_dim": 1, "cond_rates": [2, 2]},
    )
    assert isinstance(periodwave, PeriodWaveVocoder)
