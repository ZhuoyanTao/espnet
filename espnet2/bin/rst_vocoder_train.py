#!/usr/bin/env python3
"""Train the ESPnet restoration vocoder (recipe stages 7 and 8).

The config's ``vocoder_type`` decides the objective: ``dac`` and ``hifigan``
train adversarially with two optimizers (RestorationVocoderTask), ``cfm`` and
``periodwave`` by conditional flow matching with one
(RestorationFlowVocoderTask). Both tasks take the same arguments, so one
command line serves every vocoder.
"""

from espnet2.rst.decoder.dac_vocoder import FLOW_VOCODER_TYPES
from espnet2.tasks.rst_vocoder import RestorationFlowVocoderTask, RestorationVocoderTask


def select_task(cmd=None):
    """Return the task class for ``--vocoder_type`` (command line or --config)."""
    # The flow task's parser is the GAN task's plus --sigma_min, so it parses
    # every config; --config is resolved by the parser itself.
    args, _ = RestorationFlowVocoderTask.get_parser().parse_known_args(cmd)
    if args.vocoder_type in FLOW_VOCODER_TYPES:
        return RestorationFlowVocoderTask
    return RestorationVocoderTask


def get_parser():
    parser = RestorationFlowVocoderTask.get_parser()
    return parser


def main(cmd=None):
    select_task(cmd).main(cmd=cmd)


if __name__ == "__main__":
    main()
