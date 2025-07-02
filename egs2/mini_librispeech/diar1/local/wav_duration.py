#!/usr/bin/env python3
"""
Compute utterance durations for ESPnet/Kaldi wav.scp entries.

Usage:  wav_duration.py scp:<wav.scp> ark,t:<out.utt2dur>
"""

import sys, re, pathlib, soundfile as sf

def get_path(cmd: str) -> str:
    """Return the real audio filename from a wav.scp field."""
    cmd = cmd.strip()
    if cmd.endswith("|"):                                # handle pipes
        # LibriSpeech & friends: flac -c -d -s /path/file.flac |
        m = re.search(r'\s([\S]+?\.(?:wav|flac))\s*\|$', cmd)
        if m:
            return m.group(1)
        raise RuntimeError(f"Unrecognized piped command: {cmd}")
    return cmd                                           # plain path

def main():
    if len(sys.argv) != 3 or not sys.argv[1].startswith("scp:") \
                          or not sys.argv[2].startswith("ark,t:"):
        sys.exit("usage: wav_duration.py scp:<wav.scp> ark,t:<outfile>")

    scp     = pathlib.Path(sys.argv[1][4:])          # strip "scp:"
    outpath = pathlib.Path(sys.argv[2][6:])          # strip "ark,t:"
    outpath.parent.mkdir(parents=True, exist_ok=True)

    with scp.open() as fin, outpath.open("w") as fout:
        for line in fin:
            if not line.strip(): continue
            utt, wav_field = line.split(maxsplit=1)
            real_path = get_path(wav_field)
            with sf.SoundFile(real_path) as f:
                dur = len(f) / f.samplerate
            fout.write(f"{utt} {dur:.3f}\n")

if __name__ == "__main__":
    main()
