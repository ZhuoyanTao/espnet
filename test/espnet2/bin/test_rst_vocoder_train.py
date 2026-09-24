from argparse import ArgumentParser

import pytest

from espnet2.bin.rst_vocoder_train import get_parser, main, select_task
from espnet2.tasks.rst_vocoder import RestorationFlowVocoderTask, RestorationVocoderTask


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_vocoder_type_choices():
    parser = get_parser()
    args = parser.parse_args(["--vocoder_type", "hifigan", "--output_dir", "x"])
    assert args.vocoder_type == "hifigan"
    args = parser.parse_args(["--vocoder_type", "periodwave", "--output_dir", "x"])
    assert args.vocoder_type == "periodwave"
    with pytest.raises(SystemExit):
        parser.parse_args(["--vocoder_type", "unknown", "--output_dir", "x"])


@pytest.mark.parametrize(
    "vocoder_type, task",
    [
        ("dac", RestorationVocoderTask),
        ("hifigan", RestorationVocoderTask),
        ("cfm", RestorationFlowVocoderTask),
        ("periodwave", RestorationFlowVocoderTask),
    ],
)
def test_select_task_from_command_line(vocoder_type, task):
    assert select_task(["--vocoder_type", vocoder_type, "--output_dir", "x"]) is task


def test_select_task_from_config(tmp_path):
    # The selection must not validate the rest of the config: a GAN config
    # carries optim2, which the flow task's parser does not know, and a flow
    # config carries sigma_min, which the GAN task's parser does not know.
    config = tmp_path / "train.yaml"
    config.write_text("vocoder_type: cfm\nsigma_min: 0.001\n")
    assert select_task(["--config", str(config)]) is RestorationFlowVocoderTask
    config.write_text("vocoder_type: hifigan\noptim2: adamw\n")
    assert select_task(["--config", str(config)]) is RestorationVocoderTask
    config.write_text("optim2: adamw\n")  # no type: the default, dac
    assert select_task(["--config", str(config)]) is RestorationVocoderTask
    # the command line wins over the config
    config.write_text("vocoder_type: dac\n")
    task = select_task(["--config", str(config), "--vocoder_type", "periodwave"])
    assert task is RestorationFlowVocoderTask
    assert select_task(["--output_dir", "x"]) is RestorationVocoderTask
    assert select_task(["--config", "missing.yaml"]) is RestorationVocoderTask


def test_main():
    with pytest.raises(SystemExit):
        main()
