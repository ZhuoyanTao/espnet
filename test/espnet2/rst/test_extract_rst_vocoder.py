"""Test extracting the vocoder weights of an rst vocoder checkpoint."""

import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import yaml

from espnet2.rst.decoder.dac_vocoder import build_vocoder

rst_inference = pytest.importorskip("espnet2.bin.rst_inference")

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "egs2/TEMPLATE/asr1/pyscripts/utils/extract_rst_vocoder.py"
INPUT_DIM = 16
CONFIG = {"vocoder_type": "dac", "vocoder_conf": {"channels": 64, "rates": [8, 5, 4]}}


def _extract(checkpoint, output):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(checkpoint), str(output)],
        capture_output=True,
        text=True,
    )


def _restore(config_path, model_path, features):
    args = Namespace(
        external_vocoder=None,
        vocoder_train_config=str(config_path),
        vocoder_model_file=str(model_path),
    )
    vocoder = rst_inference._load_vocoder(args, INPUT_DIM, "cpu")
    with torch.no_grad():
        return vocoder(features)


def test_extracted_weights_restore_the_same_audio(tmp_path):
    torch.manual_seed(0)
    vocoder = build_vocoder("dac", INPUT_DIM, CONFIG["vocoder_conf"])
    weights = {f"vocoder.{k}": v for k, v in vocoder.state_dict().items()}
    # what a stage 7-8 checkpoint holds besides the vocoder
    others = {
        "ssl_encoder.student.layer.weight": torch.randn(4, 4),
        "discriminator.layer.weight": torch.randn(4, 4),
    }
    full = tmp_path / "valid.loss_mel.best.pth"
    torch.save({**weights, **others}, full)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(CONFIG))

    extracted = tmp_path / "vocoder.pth"
    assert _extract(full, extracted).returncode == 0
    assert set(torch.load(extracted, weights_only=True)) == set(weights)

    features = torch.randn(1, INPUT_DIM, 5)
    torch.testing.assert_close(
        _restore(config, extracted, features), _restore(config, full, features)
    )


def test_checkpoint_without_vocoder(tmp_path):
    checkpoint = tmp_path / "valid.loss.best.pth"
    torch.save({"ssl_encoder.student.layer.weight": torch.randn(4, 4)}, checkpoint)
    result = _extract(checkpoint, tmp_path / "vocoder.pth")
    assert result.returncode != 0
    assert "no 'vocoder.*' tensors" in result.stderr
    assert not (tmp_path / "vocoder.pth").exists()
