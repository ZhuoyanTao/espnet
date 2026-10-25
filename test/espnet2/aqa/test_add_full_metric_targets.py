"""Test the add_full_metric_targets utility."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
UTILS = ROOT / "egs2/TEMPLATE/asr1/pyscripts/utils"
spec = importlib.util.spec_from_file_location(
    "add_full_metric_targets", UTILS / "add_full_metric_targets.py"
)
targets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(targets)


@pytest.fixture
def metric_files(tmp_path):
    """Write prefix and whole-clip metric.scp files and their vocabulary."""
    prefix = tmp_path / "prefix.scp"
    prefix.write_text(
        'a__p002000 {"mos": 2.0}\n'
        'a__p004000 {"mos": 3.0, "lang": "en"}\n'
        'b__p002000 {"mos": 1.0}\n'
        'a {"mos": 3.5}\n'
    )
    whole = tmp_path / "whole.scp"
    whole.write_text('a {"mos": 3.5, "lang": "en", "wer": 0.1}\n')
    (tmp_path / "metric2id").write_text("mos\nlang\nwer\n")
    (tmp_path / "metric2type").write_text(
        "mos numerical\nlang categorical\nwer numerical\n"
    )
    return tmp_path


def test_full_targets(metric_files):
    """Prefix rows gain chosen whole-clip labels; other rows pass through."""
    output = metric_files / "out"
    targets.main(
        [
            str(metric_files / "prefix.scp"),
            str(metric_files / "whole.scp"),
            str(output),
            "--metrics",
            "mos",
            "lang",
        ]
    )
    rows = dict(
        line.split(" ", 1) for line in (output / "metric.scp").read_text().splitlines()
    )
    assert {key: json.loads(value) for key, value in rows.items()} == {
        "a__p002000": {"mos": 2.0, "mos_full": 3.5, "lang_full": "en"},
        "a__p004000": {"mos": 3.0, "lang": "en", "mos_full": 3.5, "lang_full": "en"},
        "b__p002000": {"mos": 1.0},
        "a": {"mos": 3.5},
    }
    assert not (output / "metric2id").exists()


def test_full_vocabulary(metric_files):
    """metric2id and metric2type gain typed _full names once."""
    (metric_files / "metric2id").write_text("mos\nlang\nwer\nmos_full\n")
    output = metric_files / "out"
    options = ["--metric2id", str(metric_files / "metric2id")]
    options += ["--metric2type", str(metric_files / "metric2type")]
    arguments = [str(metric_files / name) for name in ("prefix.scp", "whole.scp")]
    targets.main([*arguments, str(output), "--metrics", "mos", "lang", *options])
    assert (output / "metric2id").read_text() == "mos\nlang\nwer\nmos_full\nlang_full\n"
    assert (output / "metric2type").read_text().splitlines()[-2:] == [
        "mos_full numerical",
        "lang_full categorical",
    ]
    with pytest.raises(ValueError, match="No type"):
        targets.main([*arguments, str(output), "--metrics", "pesq", *options])
    with pytest.raises(ValueError, match="both"):
        targets.main([*arguments, str(output), "--metrics", "mos", *options[:2]])
