#!/usr/bin/env python3
"""Write the leading prefixes of every clip in a wav.scp as new utterances.

Each kept prefix of ``p`` seconds is keyed ``<utt>__p<ms>``, with ``p`` in
milliseconds padded to six digits (``utt1__p002000`` for 2 s). A prefix is kept
only if it is shorter than the clip by more than ``--min_margin`` seconds, so a
clip is never repeated as its own prefix; it keeps its original key elsewhere.

Prefix audio is written as one file per prefix (``wav``/``flac``) or as entries
of one Kaldi ark (``wav.ark``/``flac.ark``), which the ``sound`` and
``kaldi_ark`` training loaders read. Sliced entries such as ``a.ark:9[0:31999]``
are not written because those loaders cannot read them. Lengths come from the
decoded audio, which is also written as ``utt2num_samples``.
"""

import argparse
import re
from pathlib import Path

from omniio import kaldi as kaldi_io

from espnet2.fileio.read_text import read_2columns_text
from espnet2.fileio.sound_scp import SoundScpWriter

PREFIX_KEY = re.compile(r"^(?P<utt>.+)__p(?P<ms>\d{6,})$")


def prefix_key(utt, seconds):
    """Return the key of the ``seconds``-long prefix of ``utt``.

    Args:
        utt (str): Key of the whole clip.
        seconds (float): Prefix length in seconds.

    Returns:
        str: ``<utt>__p<milliseconds, six digits>``.
    """
    return f"{utt}__p{int(round(seconds * 1000)):06d}"


def plan_prefixes(num_samples, rate, seconds, min_margin=0.001):
    """Select the prefix lengths that fit a clip.

    Args:
        num_samples (int): Clip length in samples.
        rate (int): Sampling rate in Hz.
        seconds (list): Candidate prefix lengths in seconds.
        min_margin (float): A prefix must be shorter than the clip by more
            than this many seconds.

    Returns:
        list: ``(seconds, samples)`` for each kept prefix.
    """
    duration = num_samples / rate
    return [(p, int(round(p * rate))) for p in seconds if p < duration - min_margin]


def make_writer(out_dir, audio_format, audio_subtype):
    """Open the writer that stores prefix audio and fills ``wav.scp``.

    Args:
        out_dir (Path): Output data directory.
        audio_format (str): ``wav``, ``flac``, ``wav.ark`` or ``flac.ark``.
        audio_subtype (str): Soundfile subtype, or None for the format default.

    Returns:
        SoundScpWriter or WriteHelper: Writer taking ``writer[key] = (rate, wave)``.
    """
    audio_dir = out_dir / "data"
    if not audio_format.endswith(".ark"):
        return SoundScpWriter(
            audio_dir, out_dir / "wav.scp", format=audio_format, subtype=audio_subtype
        )
    audio_dir.mkdir(parents=True, exist_ok=True)
    return kaldi_io.WriteHelper(
        f"ark,scp:{audio_dir / ('prefix_' + audio_format)},{out_dir / 'wav.scp'}",
        write_function="soundfile",
        write_kwargs={"format": audio_format[: -len(".ark")], "subtype": audio_subtype},
    )


def make_prefixes(
    wav_scp, out_dir, seconds, min_margin, audio_format="wav", audio_subtype=None
):
    """Write prefix audio, ``wav.scp`` and ``utt2num_samples`` for a wav.scp.

    Args:
        wav_scp (str): Input ``wav.scp`` (paths, pipes or ark offsets).
        out_dir (str): Output data directory.
        seconds (list): Prefix lengths in seconds.
        min_margin (float): See :func:`plan_prefixes`.
        audio_format (str): See :func:`make_writer`.
        audio_subtype (str): See :func:`make_writer`.

    Returns:
        int: Number of prefixes written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = make_writer(out_dir, audio_format, audio_subtype)
    handles, count = {}, 0
    try:
        with (out_dir / "utt2num_samples").open("w") as lengths:
            for utt, name in read_2columns_text(wav_scp).items():
                if PREFIX_KEY.match(utt):
                    raise ValueError(f"{utt} already looks like a prefix key")
                value = kaldi_io.load_mat(name, fd_dict=handles)
                if not isinstance(value, tuple):
                    raise ValueError(f"{utt} is not an audio entry: {name}")
                rate, wave = value
                for p, samples in plan_prefixes(len(wave), rate, seconds, min_margin):
                    key = prefix_key(utt, p)
                    writer[key] = (rate, wave[:samples])
                    lengths.write(f"{key} {samples}\n")
                    count += 1
    finally:
        writer.close()
        for handle in handles.values():
            handle.close()
    for name in ("wav.scp", "utt2num_samples"):
        # Keep Kaldi's sort order: "a-b" sorts before "a__p002000".
        lines = (out_dir / name).read_text(encoding="utf-8").splitlines(True)
        (out_dir / name).write_text("".join(sorted(lines)), encoding="utf-8")
    return count


def get_parser():
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("wav_scp", help="Input wav.scp of whole clips")
    parser.add_argument("out_dir", help="Output directory for the prefix rows")
    parser.add_argument(
        "--prefix_seconds",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 6.0, 8.0],
        help="Prefix lengths in seconds",
    )
    parser.add_argument(
        "--min_margin",
        type=float,
        default=0.001,
        help="Keep a prefix only if the clip is longer by more than this (s)",
    )
    parser.add_argument(
        "--audio_format",
        default="wav",
        choices=["wav", "flac", "wav.ark", "flac.ark"],
        help="One file per prefix, or one Kaldi ark with one entry per prefix",
    )
    parser.add_argument(
        "--audio_subtype",
        default=None,
        help="Soundfile subtype, e.g. PCM_16 (default) or FLOAT",
    )
    return parser


def main(argv=None):
    """Run the command-line interface.

    Args:
        argv (list): Arguments; None reads ``sys.argv``.
    """
    args = get_parser().parse_args(argv)
    count = make_prefixes(
        args.wav_scp,
        args.out_dir,
        args.prefix_seconds,
        args.min_margin,
        args.audio_format,
        args.audio_subtype,
    )
    print(f"Wrote {count} prefixes to {args.out_dir}")


if __name__ == "__main__":
    main()
