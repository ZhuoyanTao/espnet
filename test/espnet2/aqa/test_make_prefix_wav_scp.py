"""Test the make_prefix_wav_scp utility."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile

from espnet2.train.dataset import ESPnetDataset

ROOT = Path(__file__).resolve().parents[3]
UTILS = ROOT / "egs2/TEMPLATE/asr1/pyscripts/utils"
spec = importlib.util.spec_from_file_location(
    "make_prefix_wav_scp", UTILS / "make_prefix_wav_scp.py"
)
prefixes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefixes)


def test_prefix_plan():
    """Keys use six-digit milliseconds; prefixes must undercut the clip."""
    assert prefixes.prefix_key("utt", 2.0) == "utt__p002000"
    assert prefixes.prefix_key("utt", 0.25) == "utt__p000250"
    assert prefixes.plan_prefixes(64016, 16000, [2.0, 4.0]) == [(2.0, 32000)]
    assert prefixes.plan_prefixes(64017, 16000, [4.0]) == [(4.0, 64000)]
    assert prefixes.plan_prefixes(64000, 16000, [4.0], min_margin=0.0) == []


@pytest.mark.parametrize(
    "audio_format, loader", [("wav", "sound"), ("flac.ark", "kaldi_ark")]
)
def test_prefix_audio(tmp_path, audio_format, loader):
    """Sorted prefixes load through the training loaders as clip heads."""
    rate = 1000
    rng = np.random.default_rng(0)
    clips = {
        "a": rng.integers(-2000, 2000, 4500, dtype=np.int16),
        "a-b": rng.integers(-2000, 2000, 2500, dtype=np.int16),
        "c": rng.integers(-2000, 2000, 2000, dtype=np.int16),
    }
    scp = tmp_path / "wav.scp"
    with scp.open("w") as f:
        for key, wave in clips.items():
            soundfile.write(tmp_path / f"{key}.wav", wave, rate)
            f.write(f"{key} {tmp_path / key}.wav\n")
    output = tmp_path / "prefix"
    prefixes.main([str(scp), str(output), "--audio_format", audio_format])

    lengths = (output / "utt2num_samples").read_text()
    assert lengths == "a-b__p002000 2000\na__p002000 2000\na__p004000 4000\n"
    dataset = ESPnetDataset([(str(output / "wav.scp"), "speech", loader)])
    for line in lengths.splitlines():
        key, size = line.split()
        speech = dataset[key][1]["speech"]
        expected = clips[key.split("__p")[0]][: int(size)] / 32768
        np.testing.assert_array_equal(speech, expected)


def test_prefix_key_input(tmp_path):
    """Refuse clips that already carry a prefix key."""
    soundfile.write(tmp_path / "a.wav", np.zeros(3000, dtype=np.int16), 1000)
    scp = tmp_path / "wav.scp"
    scp.write_text(f"a__p002000 {tmp_path / 'a.wav'}\n")
    with pytest.raises(ValueError, match="prefix key"):
        prefixes.main([str(scp), str(tmp_path / "out")])
