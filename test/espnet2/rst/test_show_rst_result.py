"""Test the RESULTS.md summary of the rst recipe."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "egs2/TEMPLATE/asr1/pyscripts/utils/show_rst_result.py"


def _run(expdir):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(expdir)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_tables_per_kind_of_score(tmp_path):
    expdir = tmp_path / "rst_w2v_bert2"
    _write(
        expdir / "score_test-clean/scores.json",
        json.dumps({"spksim_num_scored": 3, "dnsmos_mean": 3.25, "wer": 0.03125}),
    )
    _write(expdir / "score_test-other/scores.json", json.dumps({"dnsmos_mean": 3.0}))
    _write(
        expdir / "inference_test-clean/scoring/versa_eval/avg_result.txt",
        "utmos: 4.5\n"
        "espnet_wer_delete: 2\n"
        "espnet_wer_insert: 1\n"
        "espnet_wer_replace: 3\n"
        "espnet_wer_equal: 94\n"
        "espnet_wer 0.06\n",
    )

    markdown = _run(expdir)

    assert "## rst_w2v_bert2" in markdown
    assert "|dataset|spksim_num_scored|dnsmos_mean|wer|" in markdown
    assert "|test-clean|3|3.2500|0.0312|" in markdown
    # columns a test set lacks stay empty instead of shifting the row
    assert "|test-other||3.0000||" in markdown
    # the edit counts behind the WER are left out; the rate is kept
    assert "|dataset|utmos|espnet_wer|" in markdown
    assert "|test-clean|4.5000|0.0600|" in markdown
    # no reference-based VERSA scores, so no table for them
    assert "clean reference" not in markdown


def test_no_scores_yet(tmp_path):
    expdir = tmp_path / "rst_debug"
    expdir.mkdir()
    markdown = _run(expdir)
    assert "# RESULTS" in markdown
    assert "## rst_debug" in markdown
    assert "|dataset|" not in markdown
