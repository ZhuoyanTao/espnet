"""Test the extend_metric_vocab utility."""

import importlib.util
import json
from pathlib import Path
from test.espnet2.aqa.test_metric_tokenizer import token_info

import pytest
import torch

from espnet2.aqa.metric_tokenizer.metric_tokenizer import MetricTokenizer
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder

ROOT = Path(__file__).resolve().parents[3]
UTILS = ROOT / "egs2/TEMPLATE/asr1/pyscripts/utils"
spec = importlib.util.spec_from_file_location(
    "extend_metric_vocab", UTILS / "extend_metric_vocab.py"
)
vocab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vocab)


def make_decoder(size):
    """Build a tiny metric decoder with ``size`` tokens."""
    return TransformerDecoder(size, 8, attention_heads=2, linear_units=16, num_blocks=1)


def test_extend_vocabulary(tmp_path):
    """New metrics append tokens; old IDs and decoder rows stay unchanged."""
    old_token_info = token_info()
    old_info = tmp_path / "tokens.json"
    old_info.write_text(json.dumps(old_token_info))
    metric_scp = tmp_path / "metric.scp"
    metric_scp.write_text(
        "".join(f'u{i} {{"hmos": {i}, "mos": 1.0}}\n' for i in range(1, 6))
    )
    old_size = len(old_token_info["VOCAB"]) + 4
    decoder = make_decoder(old_size)
    torch.save(
        {f"universa.decoder.{k}": v for k, v in decoder.state_dict().items()},
        tmp_path / "model.pth",
    )
    new_info = tmp_path / "tokens_new.json"
    vocab.main(
        [str(old_info), str(new_info), "--numerical", "hmos", "--token_size", "4"]
        + ["--metric_scp", str(metric_scp), "--categorical", "gender=f,m"]
        + ["--checkpoint", str(tmp_path / "model.pth")]
        + ["--output_checkpoint", str(tmp_path / "new.pth")]
    )

    info = json.loads(new_info.read_text())
    assert info["VOCAB"][: old_size - 4] == old_token_info["VOCAB"]
    assert info["tokenizer"]["hmos"] == [1.0, 3.0, 5.0]
    assert info["offset"]["hmos"] == [old_size - 4, 6]
    assert info["offset"]["gender"] == [old_size - 4 + 6, 4]
    old = MetricTokenizer(old_token_info, None)
    new = MetricTokenizer(info, None)
    assert new.metric2token({"mos": 1.5}) == old.metric2token({"mos": 1.5})
    assert new.metric2token({"hmos": 4.0, "gender": "m"}) == {
        "hmos": (old_size, old_size + 3),
        "gender": (old_size + 6, old_size + 9),
    }

    state = torch.load(tmp_path / "new.pth", weights_only=True)
    extended = make_decoder(len(info["VOCAB"]) + 4)
    extended.load_state_dict(
        {k[len("universa.decoder.") :]: v for k, v in state.items()}
    )
    for name, value in decoder.state_dict().items():
        torch.testing.assert_close(extended.state_dict()[name][: len(value)], value)
    assert not extended.output_layer.bias[old_size:].any()


def test_extend_requires_values(tmp_path):
    """Refuse numerical metrics without values and existing metric names."""
    old_token_info = token_info()
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps(old_token_info))
    metric_scp = tmp_path / "metric.scp"
    metric_scp.write_text('u1 {"mos": 1.0}\n')
    arguments = [str(path), str(tmp_path / "out.json"), "--metric_scp", str(metric_scp)]
    with pytest.raises(ValueError, match="No values"):
        vocab.main(arguments + ["--numerical", "hmos"])
    with pytest.raises(ValueError, match="already"):
        vocab.main(arguments + ["--numerical", "mos"])
