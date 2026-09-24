#!/usr/bin/env python3
"""Train the ESPnet restoration vocoder (recipe stages 7 and 8).

The config's ``vocoder_type`` decides the objective: ``dac`` and ``hifigan``
train adversarially with two optimizers (RestorationVocoderTask), ``cfm`` and
``periodwave`` by conditional flow matching with one
(RestorationFlowVocoderTask). Both tasks take the same task arguments, so one
command line serves every vocoder; the trainer-level options differ (the GAN
task has ``--optim2`` / ``--scheduler2``, the flow task ``--sigma_min``), and
``--help`` shows those of the default type, ``dac``.
"""

import argparse
from pathlib import Path

import yaml

from espnet2.rst.decoder.dac_vocoder import FLOW_VOCODER_TYPES
from espnet2.tasks.rst_vocoder import RestorationFlowVocoderTask, RestorationVocoderTask


def select_task(cmd=None):
    """Return the task class for ``--vocoder_type`` (command line or --config).

    Only these two options are read here; everything else, including the
    validation of the config's keys, is left to the selected task's parser.
    """
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--config")
    peek.add_argument("--vocoder_type")
    args, _ = peek.parse_known_args(cmd)
    vocoder_type = args.vocoder_type
    if vocoder_type is None and args.config and Path(args.config).is_file():
        with open(args.config, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if isinstance(config, dict):
            vocoder_type = config.get("vocoder_type")
    if vocoder_type in FLOW_VOCODER_TYPES:
        return RestorationFlowVocoderTask
    return RestorationVocoderTask


def get_parser():
    parser = RestorationVocoderTask.get_parser()
    return parser


def main(cmd=None):
    select_task(cmd).main(cmd=cmd)


if __name__ == "__main__":
    main()
