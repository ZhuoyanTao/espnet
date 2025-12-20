import torch

from espnet2.universa.ar_universa.ar_universa_utils import (
    slice_audio,
    chunk_audio,
)

def main():
    fs = 16000
    # 5.1 seconds fake audio
    x = torch.randn(1, int(5.1 * fs))

    # slice 1.0-3.0s
    y = slice_audio(x, fs, 1.0, 3.0)
    print("slice shape:", y.shape)

    # non-overlap 2s chunks
    chunks = chunk_audio(x, fs, chunk_s=2.0, hop_s=2.0)
    print("num chunks:", len(chunks))
    for i, c in enumerate(chunks):
        print(i, c.shape)

if __name__ == "__main__":
    main()
