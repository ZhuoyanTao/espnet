#!/usr/bin/env python3
"""Keep only the vocoder weights of an rst vocoder checkpoint.

A checkpoint from stages 7-8 of egs2/TEMPLATE/rst1/rst.sh also holds the SSL
encoder and the discriminator, which only training uses. Inference
(espnet2/bin/rst_inference.py) reads the "vocoder.*" tensors alone, so packing
just those leaves out about 1.7 GB with the w2v-BERT 2.0 encoder.
"""

import argparse

import torch


def main(cmd=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", help="e.g. valid.loss_mel.best.pth")
    parser.add_argument("output", help="file to write the vocoder weights to")
    args = parser.parse_args(cmd)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    vocoder = {key: value for key, value in state.items() if key.startswith("vocoder.")}
    if not vocoder:
        raise RuntimeError(f"no 'vocoder.*' tensors in {args.checkpoint}")
    torch.save(vocoder, args.output)


if __name__ == "__main__":
    main()
