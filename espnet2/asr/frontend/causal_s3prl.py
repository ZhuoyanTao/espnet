"""Causal S3PRL frontend: WavLM-Large with streaming (causal) attention masking.

Applies a lower-triangular causal mask to every transformer layer inside
WavLM-Large so that position i cannot attend to positions > i.  The CNN
feature extractor that precedes the transformer is already causal by design.

The masking is achieved by monkey-patching the WavLM TransformerEncoder's
forward method to inject a causal streaming_mask before each forward pass.
This avoids touching any upstream S3PRL or fairseq source files.
"""

import logging
from typing import Optional, Tuple, Union

import torch

from espnet2.asr.frontend.s3prl import S3prlFrontend


class CausalS3prlFrontend(S3prlFrontend):
    """S3PRL frontend with causal attention masking (streaming WavLM-Large).

    Drop-in replacement for S3prlFrontend.  All constructor arguments are
    forwarded unchanged; after the upstream is loaded the encoder forward is
    patched to inject a causal lower-triangular mask so the model can only
    attend to past/current frames.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._patched = False
        self._patch_encoder_for_causal_masking()

    # ------------------------------------------------------------------
    # Causal-mask patch
    # ------------------------------------------------------------------

    def _patch_encoder_for_causal_masking(self):
        """Monkey-patch WavLM's TransformerEncoder to use a causal mask."""
        try:
            encoder = self.upstream.upstream.model.encoder
        except AttributeError:
            logging.warning(
                "CausalS3prlFrontend: could not access upstream encoder "
                "via upstream.upstream.model.encoder — causal masking NOT applied."
            )
            return

        if not hasattr(encoder, "forward"):
            logging.warning(
                "CausalS3prlFrontend: encoder has no forward() method — "
                "causal masking NOT applied."
            )
            return

        orig_forward = encoder.forward

        def _causal_forward(x, padding_mask=None, streaming_mask=None, layer=None):
            """Wrap encoder forward to inject a causal streaming_mask.

            WavLM's TransformerEncoder expects:
                x            : (B, T, D)  — B batch, T time, D dim
                padding_mask : (B, T)     — True where padding
                streaming_mask: (T, T)    — additive mask (-inf above diagonal)
            """
            if streaming_mask is None:
                T = x.shape[1]  # x is (B, T, D) at encoder entry
                streaming_mask = torch.triu(
                    torch.full(
                        (T, T),
                        float("-inf"),
                        device=x.device,
                        dtype=x.dtype,
                    ),
                    diagonal=1,
                )
            return orig_forward(
                x,
                padding_mask=padding_mask,
                streaming_mask=streaming_mask,
                layer=layer,
            )

        encoder.forward = _causal_forward
        self._patched = True
        logging.info(
            "CausalS3prlFrontend: causal streaming mask successfully patched "
            "onto %s encoder.",
            type(encoder).__name__,
        )

    # ------------------------------------------------------------------
    # Reload hook — re-apply patch after parameter reload
    # ------------------------------------------------------------------

    def reload_pretrained_parameters(self):
        super().reload_pretrained_parameters()
        if not self._patched:
            self._patch_encoder_for_causal_masking()
