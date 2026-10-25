#!/usr/bin/env python3
"""Give prefix rows the labels of their whole clip as ``<metric>_full`` targets.

A prefix row keyed ``<utt>__p<ms>`` (see make_prefix_wav_scp.py) keeps the labels
measured on the prefix and gains ``<metric>_full`` = the label of the whole clip
``<utt>`` for each chosen metric that the clip has. Other rows, such as whole
clips under their own keys, are copied unchanged. With ``--metric2id`` and
``--metric2type``, copies extended with the ``_full`` names (typed like their
base metric) are written next to the output ``metric.scp``.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from espnet2.fileio.metric_scp import MetricReader
from espnet2.fileio.read_text import read_2columns_text

FULL_SUFFIX = "_full"
PREFIX_KEY = re.compile(r"^(?P<utt>.+)__p(?P<ms>\d{6,})$")


def add_full_targets(row, whole, metrics):
    """Return a prefix row extended with whole-clip targets.

    Args:
        row (dict): Labels measured on the prefix.
        whole (dict): Labels of the whole clip, or None if it has none.
        metrics (list): Metric names to copy as ``<metric>_full``.

    Returns:
        dict: The prefix labels followed by the available ``_full`` labels.
    """
    row = dict(row)
    for metric in metrics:
        if whole is not None and metric in whole:
            row[metric + FULL_SUFFIX] = whole[metric]
    return row


def write_metric_scp(prefix_metric_scp, whole_metric_scp, output, metrics):
    """Write a metric.scp whose prefix rows carry ``_full`` targets.

    Args:
        prefix_metric_scp (str): Labels of the prefix rows (and any other rows).
        whole_metric_scp (str): Labels of the whole clips.
        output (Path): Output metric.scp.
        metrics (list): Metric names to copy as ``<metric>_full``.

    Returns:
        dict: Numbers of rows, prefix rows, prefix rows without a whole clip
            and ``_full`` targets written.
    """
    whole_rows = MetricReader(whole_metric_scp)
    counts = dict(rows=0, prefix_rows=0, without_whole=0, full_targets=0)
    with (
        open(prefix_metric_scp, encoding="utf-8") as fin,
        output.open("w", encoding="utf-8") as fout,
    ):
        for line in fin:
            if not line.strip():
                continue
            key, value = line.rstrip("\n").split(maxsplit=1)
            row = json.loads(value)
            counts["rows"] += 1
            match = PREFIX_KEY.match(key)
            if match is not None:
                utt = match.group("utt")
                whole = whole_rows[utt] if utt in whole_rows else None
                counts["prefix_rows"] += 1
                counts["without_whole"] += whole is None
                row = add_full_targets(row, whole, metrics)
                counts["full_targets"] += sum(m + FULL_SUFFIX in row for m in metrics)
            fout.write(f"{key} {json.dumps(row, ensure_ascii=False)}\n")
    return counts


def extend_vocabulary(metric2id, metric2type, out_dir, metrics):
    """Write metric2id and metric2type extended with the ``_full`` names.

    Args:
        metric2id (str): Input metric2id (one name per line).
        metric2type (str): Input metric2type (``<name> <type>`` per line).
        out_dir (Path): Directory for the extended files.
        metrics (list): Base metrics whose ``_full`` names are added.
    """
    names = Path(metric2id).read_text(encoding="utf-8").split()
    types = read_2columns_text(metric2type)
    missing = [m for m in metrics if m not in types]
    if missing:
        raise ValueError(f"No type for {missing} in {metric2type}")
    for metric in metrics:
        types.setdefault(metric + FULL_SUFFIX, types[metric])
        if metric + FULL_SUFFIX not in names:
            names.append(metric + FULL_SUFFIX)
    (out_dir / "metric2id").write_text("".join(f"{n}\n" for n in names))
    (out_dir / "metric2type").write_text(
        "".join(f"{n} {t}\n" for n, t in types.items())
    )


def get_parser():
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("prefix_metric_scp", help="metric.scp of the prefix rows")
    parser.add_argument("whole_metric_scp", help="metric.scp of the whole clips")
    parser.add_argument("out_dir", help="Output directory for metric.scp")
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="Whole-clip metrics to copy as <metric>_full",
    )
    parser.add_argument("--metric2id", default=None, help="metric2id to extend")
    parser.add_argument("--metric2type", default=None, help="metric2type to extend")
    return parser


def main(argv=None):
    """Run the command-line interface.

    Args:
        argv (list): Arguments; None reads ``sys.argv``.
    """
    args = get_parser().parse_args(argv)
    if (args.metric2id is None) != (args.metric2type is None):
        raise ValueError("Give both --metric2id and --metric2type, or neither")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = write_metric_scp(
        args.prefix_metric_scp,
        args.whole_metric_scp,
        out_dir / "metric.scp",
        args.metrics,
    )
    if args.metric2id is not None:
        extend_vocabulary(args.metric2id, args.metric2type, out_dir, args.metrics)
    if counts["without_whole"]:
        logging.warning(f"{counts['without_whole']} prefix rows have no whole clip")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
