import torch
from espnet2.universa.ar_universa.ar_universa import ARUniversa

def make_dummy(batch=2, T=64, feat=80, vocab=50, token_len=8):
    torch.manual_seed(0)
    # Fake fbank-like features (B, T, F)
    audio = torch.randn(batch, T, feat)
    audio_lens = torch.full((batch,), T, dtype=torch.long)

    # Fake metric tokens and lengths
    metric_token = torch.randint(0, vocab, (batch, token_len), dtype=torch.long)
    metric_token_lens = torch.full((batch,), token_len, dtype=torch.long)

    metrics = {
        "metric_token": metric_token,
        "metric_token_lengths": metric_token_lens,
    }
    return audio, audio_lens, metrics

def run_case(use_quantizer, mode):
    print(f"\n=== use_hificodec_quantizer={use_quantizer}, mode={mode} ===")
    audio, audio_lens, metrics = make_dummy()

    model = ARUniversa(
        input_size=80,                      # matches our fake features' last dim
        metric2id={"mos": 0},               # minimal map; not used in forward()
        metric_vocab_size=50,
        use_ref_audio=False,
        use_ref_text=False,
        use_hificodec_quantizer=use_quantizer,
        hificodec_mode=mode,
        hificodec_cfg=dict(
            dim=512, n_groups=8, n_codes=1024,
            codebook_loss_lambda=1.0, commitment_loss_lambda=0.25,
            residual_layers=2, weight=0.5
        ),
        metric_token_info={
            "tokenizer": {"type": "default", "vocab_size": 50},
            "VOCAB": [f"tok{i}" for i in range(50)],  # minimal dummy vocab list
        },

    )

    model.eval()
    with torch.no_grad():
        loss, stats, weight = model(
            audio=audio,
            audio_lengths=audio_lens,
            metrics=metrics,
            ref_audio=None, ref_audio_lengths=None,
            ref_text=None, ref_text_lengths=None,
        )

    print("loss:", float(loss))
    print("stats keys:", sorted(stats.keys()))
    # Assertions:
    if use_quantizer:
        assert "loss_q" in stats, "Expected loss_q when quantizer is ON"
    else:
        assert "loss_q" not in stats, "Did not expect loss_q when quantizer is OFF"
    assert "loss_ar_decoder" in stats

if __name__ == "__main__":
    # Baseline (no quantizer): should run, no 'loss_q'
    run_case(use_quantizer=False, mode="replace")

    # Quantizer enabled (replace): should include 'loss_q'
    run_case(use_quantizer=True, mode="replace")

    # Quantizer enabled (concat): should also include 'loss_q'
    run_case(use_quantizer=True, mode="concat")

    print("\nAll smoke tests passed ✅")
