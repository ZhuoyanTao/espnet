#!/usr/bin/env python3
"""
Python re-implementation of Kaldi's wav-reverberate.
"""
import argparse
import sys
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

def read_wav(path_or_cmd, channel=0):
    # path_or_cmd: filename or pipe command (not splitting '|'); assume filename
    if path_or_cmd == '-':
        # read WAV from stdin
        data, fs = sf.read(sys.stdin.buffer, dtype='float32')
    else:
        data, fs = sf.read(path_or_cmd, dtype='float32')
    # data shape: (n_samples,) or (n_samples, n_channels)
    if data.ndim == 1:
        return data.astype(np.float32), fs
    if channel >= data.shape[1]:
        raise ValueError(f"Requested channel {channel} >= available {data.shape[1]}")
    return data[:, channel].astype(np.float32), fs


def compute_early_reverb_energy(rir, fs):
    peak = np.argmax(np.abs(rir))
    before = int(0.001 * fs)
    after = int(0.05 * fs)
    start = max(0, peak - before)
    end = min(len(rir), peak + after)
    early = rir[start:end]
    # assuming unit impulse convolve with signal of ones
    # approximate by convolving early RIR with itself
    out = fftconvolve(early, np.ones_like(early))
    energy = np.dot(out, out) / len(out)
    return energy


def do_reverberation(rir, signal, fs):
    early_energy = compute_early_reverb_energy(rir, fs)
    # full convolution
    out = fftconvolve(signal, rir)
    return out.astype(np.float32), early_energy


def add_noise(signal, noise, snr_db, start_time, fs, signal_power):
    noise_power = np.dot(noise, noise) / len(noise) + np.finfo(np.float32).eps
    scale = np.sqrt((10**(-snr_db/10)) * signal_power / noise_power)
    noise = noise * scale
    offset = int(start_time * fs)
    end = min(len(signal), offset + len(noise))
    signal[offset:end] += noise[:end-offset]
    return signal


def parse_list(arg, cast=float):
    if not arg:
        return []
    return [cast(x) for x in arg.split(',')]


def main():
    p = argparse.ArgumentParser(
        description="Corrupt a WAV with RIR and additive noise (no Kaldi needed)")
    p.add_argument('input_wav')
    p.add_argument('output_wav')
    p.add_argument('--impulse-response', default='', help='RIR WAV file')
    p.add_argument('--additive-signals', default='', help='Comma-separated list of noise WAVs')
    p.add_argument('--snrs', default='', help='Comma-separated list of SNRs (dB)')
    p.add_argument('--start-times', default='', help='Comma-separated list of start times (s)')
    p.add_argument('--duration', type=float, default=0, help='Desired output duration (s)')
    p.add_argument('--volume', type=float, default=0, help='Fixed volume scaling')
    p.add_argument('--normalize-output', action='store_true', help='Normalize to original power')
    p.add_argument('--shift-output', action='store_true', help='Shift by RIR peak')
    p.add_argument('--input-channel', type=int, default=0)
    p.add_argument('--rir-channel', type=int, default=0)
    p.add_argument('--noise-channel', type=int, default=0)
    p.add_argument('--multi-channel-output', action='store_true')
    args = p.parse_args()

    # Read input
    sig, fs = read_wav(args.input_wav, args.input_channel)
    sig = sig.astype(np.float32)
    original_power = np.dot(sig, sig) / len(sig)

    # Read RIR
    if args.impulse_response:
        rir, fs_rir = read_wav(args.impulse_response, args.rir_channel)
        if fs_rir != fs:
           # bring RIR to the same rate as the speech
           from scipy.signal import resample_poly
           rir = resample_poly(rir, fs, fs_rir)
           fs_rir = fs
    else:
        rir = None

    # Pre-reverb
    if rir is not None:
        rev, early_energy = do_reverberation(rir, sig, fs)
        # shift
        if args.shift_output:
            shift_idx = int(np.argmax(np.abs(rir)))
            rev = np.roll(rev, -shift_idx)
        sig = rev
    else:
        early_energy = original_power

    # Add noise
    noises = args.additive_signals.split(',') if args.additive_signals else []
    snrs = parse_list(args.snrs)
    starts = parse_list(args.start_times)
    for noise_path, snr, st in zip(noises, snrs, starts):
        noise, fs_noise = read_wav(noise_path, args.noise_channel)
        if fs_noise != fs:
            raise ValueError("Sampling rates of input and noise differ")
        sig = add_noise(sig, noise, snr, st, fs, early_energy)

    # normalize or volume
    after_power = np.dot(sig, sig) / len(sig)
    if args.volume > 0:
        sig *= args.volume
    elif args.normalize_output:
        sig *= np.sqrt(original_power / (after_power + np.finfo(np.float32).eps))

    # Determine output length
    if args.duration > 0:
        out_len = int(args.duration * fs)
    elif rir is not None and not args.shift_output:
        out_len = len(sig)
    else:
        out_len = len(sig)

    # Trim or pad
    if len(sig) > out_len:
        out = sig[:out_len]
    elif len(sig) < out_len:
        # repeat to fill
        reps = int(np.ceil(out_len / len(sig)))
        out = np.tile(sig, reps)[:out_len]
    else:
        out = sig

    # Write output
    import sys
    if args.output_wav == '-':
        sf.write(sys.stdout.buffer, out, fs,
                 format='WAV', subtype='PCM_16')
    else:
        sf.write(args.output_wav, out, fs)


if __name__ == '__main__':
    main()
