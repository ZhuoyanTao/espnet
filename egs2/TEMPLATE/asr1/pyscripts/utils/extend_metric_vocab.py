#!/usr/bin/env python3
"""Add metrics to an ARECHO metric token info and, optionally, a checkpoint.

Existing token IDs do not change: each new metric appends ``<m>@meta_label``
and ``<m>@0`` ... ``<m>@<n>`` to ``VOCAB`` and gets an ``offset`` entry, laid
out as in the published token info files. A numerical metric has
``n = --token_size`` and ``n - 1`` thresholds, the percentiles of its values in
``--metric_scp``. A categorical metric, given as ``NAME=a,b,c``, has one token
per label (label ``i`` is ``<m>@<i + 1>``). With ``--checkpoint``, rows for the
new tokens are appended to the decoder embedding and output layer: random rows
scaled to ``--init_scale`` times the existing rows' standard deviation, and
zero bias.
"""

import argparse
import json

import numpy as np

SPECIAL_TOKENS = 4  # <pad>, <unk>, <eos> and <sos> precede VOCAB.


def add_metric(info, name, values, size):
    """Append one metric's token block to a token info dict in place.

    Args:
        info (dict): Token info with ``tokenizer``, ``VOCAB`` and ``offset``.
        name (str): New metric name.
        values (list): Numerical thresholds or categorical labels.
        size (int): Last value token, ``<name>@<size>``.
    """
    if name in info["tokenizer"]:
        raise ValueError(f"{name} is already in the token info")
    info["tokenizer"][name] = list(values)
    info["offset"][name] = [len(info["VOCAB"]), size + 2]
    info["VOCAB"] += [f"{name}@meta_label"]
    info["VOCAB"] += [f"{name}@{i}" for i in range(size + 1)]


def percentile_thresholds(metric_scp, names, token_size):
    """Compute numerical thresholds from the values in a metric.scp.

    Args:
        metric_scp (str): Labels to take the values from.
        names (list): Numerical metric names.
        token_size (int): One more than the number of thresholds.

    Returns:
        dict: Thresholds per metric name.
    """
    values = {name: [] for name in names}
    with open(metric_scp, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line.split(maxsplit=1)[1])
                for name in names:
                    if name in row:
                        values[name].append(float(row[name]))
    empty = [name for name in names if not values[name]]
    if empty:
        raise ValueError(f"No values for {empty} in {metric_scp}")
    points = np.linspace(0, 100, token_size - 1)
    return {name: np.percentile(v, points).tolist() for name, v in values.items()}


def extend_checkpoint(path, output, old_size, new_size, prefix, scale, seed):
    """Append decoder embedding and output rows for new tokens.

    Args:
        path (str): Input model state dict.
        output (str): Output model state dict.
        old_size (int): Decoder vocabulary size before the new tokens.
        new_size (int): Decoder vocabulary size after the new tokens.
        prefix (str): State-dict prefix of the metric decoder.
        scale (float): New weight rows are N(0, (scale * std(old rows))^2).
        seed (int): Random seed of the new rows.
    """
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    generator = torch.Generator().manual_seed(seed)
    for name in ("embed.0.weight", "output_layer.weight", "output_layer.bias"):
        tensor = state[prefix + name]
        if tensor.shape[0] != old_size:
            raise ValueError(f"{prefix + name} has {tensor.shape[0]} rows")
        shape = (new_size - old_size,) + tuple(tensor.shape[1:])
        if name.endswith("bias"):
            rows = tensor.new_zeros(shape)
        else:
            rows = torch.randn(shape, generator=generator, dtype=tensor.dtype)
            rows = rows * float(tensor.std()) * scale
        state[prefix + name] = torch.cat([tensor, rows])
    torch.save(state, output)


def get_parser():
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("token_info", help="Input metric token info JSON")
    parser.add_argument("output", help="Output metric token info JSON")
    parser.add_argument("--numerical", nargs="+", default=[], help="New metrics")
    parser.add_argument("--metric_scp", help="Values of the numerical metrics")
    parser.add_argument("--token_size", type=int, default=500)
    parser.add_argument(
        "--categorical", nargs="+", default=[], help="New metrics as NAME=a,b,c"
    )
    parser.add_argument("--checkpoint", help="Model state dict to extend")
    parser.add_argument("--output_checkpoint", help="Extended model state dict")
    parser.add_argument("--decoder_prefix", default="universa.decoder.")
    parser.add_argument("--init_scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    """Run the command-line interface.

    Args:
        argv (list): Arguments; None reads ``sys.argv``.
    """
    args = get_parser().parse_args(argv)
    if (args.checkpoint is None) != (args.output_checkpoint is None):
        raise ValueError("Give both --checkpoint and --output_checkpoint, or neither")
    if args.numerical and args.metric_scp is None:
        raise ValueError("--numerical needs --metric_scp")
    with open(args.token_info) as f:
        info = json.load(f)
    old_size = len(info["VOCAB"]) + SPECIAL_TOKENS
    if args.numerical:
        thresholds = percentile_thresholds(
            args.metric_scp, args.numerical, args.token_size
        )
        for name, values in thresholds.items():
            add_metric(info, name, values, args.token_size)
    for spec in args.categorical:
        name, labels = spec.split("=", 1)
        add_metric(info, name, labels.split(","), len(labels.split(",")))
    with open(args.output, "w") as f:
        json.dump(info, f, ensure_ascii=False)
    new_size = len(info["VOCAB"]) + SPECIAL_TOKENS
    if args.checkpoint is not None:
        extend_checkpoint(
            args.checkpoint,
            args.output_checkpoint,
            old_size,
            new_size,
            args.decoder_prefix,
            args.init_scale,
            args.seed,
        )
    print(f"Metric vocabulary size: {old_size} -> {new_size}")


if __name__ == "__main__":
    main()
