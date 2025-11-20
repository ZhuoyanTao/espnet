# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""UniversaBase related modules."""

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from packaging.version import parse as V
from typeguard import typechecked

from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.universa.abs_universa import AbsUniversa
from espnet2.universa.ar_universa.universa_beam_search import ARUniVERSABeamSearch
from espnet2.universa.metric_tokenizer.metric_tokenizer import MetricTokenizer
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask, th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,
    RoPEPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ARUniversa(AbsUniversa):
    def __init__(
        self,
        # Model Backbone
        input_size: int,
        metric2id: Dict[str, int],
        use_ref_audio: bool = True,
        use_ref_text: bool = True,
        embedding_size: int = 512,
        use_normalize: bool = True,
        audio_encoder_type: str = "transformer",
        audio_encoder_params: Dict[str, Union[float, int, bool, str]] = {
            "num_blocks": 3,
            "attention_heads": 4,
            "linear_units": 2048,
            "dropout_rate": 0.1,
            "positional_dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "input_layer": "linear",
            "normalize_before": True,
            "concat_after": False,
            "positionwise_layer_type": "linear",
            "positionwise_conv_kernel_size": 1,
            "layer_drop_rate": 0.0,
            "qk_norm": False,
            "use_flash_attn": False,
        },
        # Metric related
        metric_vocab_size: Optional[int] = None,
        metric_token_info: Optional[Dict[str, Any]] = None,
        metric2type: Optional[Dict[str, str]] = None,
        metric_pad_value: float = -100,
        metric_token_pad_value: int = 0,
        sequential_metrics: bool = True,
        # Text processor
        vocab_size: Optional[int] = None,
        ignore_id: int = -1,
        text_encoder_type: str = "transformer",
        text_encoder_params: Dict[str, Union[float, int, bool, str]] = {
            "num_blocks": 3,
            "attention_heads": 4,
            "linear_units": 2048,
            "dropout_rate": 0.1,
            "positional_dropout_rate": 0.1,
            "attention_dropout_rate": 0.1,
            "input_layer": "linear",
            "normalize_before": True,
            "concat_after": False,
            "positionwise_layer_type": "linear",
            "positionwise_conv_kernel_size": 1,
            "layer_drop_rate": 0.0,
            "qk_norm": False,
            "use_flash_attn": False,
        },
        # Attention modules
        cross_attention_type: str = "multihead",
        cross_attention_params: Dict[str, Union[float, int]] = {
            "n_head": 4,
            "dropout_rate": 0.1,
        },
        # Decoder modules
        metric_decoder_params: Dict[str, Union[float, int]] = {
            "num_blocks": 3,
            "attention_heads": 4,
            "linear_units": 2048,
            "dropout_rate": 0.1,
            "positional_dropout_rate": 0.1,
            "self_attention_dropout_rate": 0.1,
            "src_attention_dropout_rate": 0.1,
            "input_laye": "embed",
            "use_output_layer": True,
            "normalize_before": True,
            "concat_after": False,
            "layer_drop_rate": 0.0,
            "qk_norm": False,
            "use_flash_attn": False,
        },
        use_rope_pos: bool = False,
        # Other parameters
        lsm_weight: float = 0.0,
        # Pretrained HF Tokenizer may needs custom sym_sos and sym_eos
        sym_sos: str = "<sos>",
        sym_eos: str = "<eos>",
        **kwargs,
    ):
        """Initialize UniversaBase module.

        Args:
            input_size (int): Input feature size.
            metric2id (Dict[str, int]): Dictionary mapping metric names to IDs.
            use_ref_audio (bool): Whether to use reference audio.
            use_ref_text (bool): Whether to use reference text.
            embedding_size (int): Embedding size for audio and text encoders.
            use_normalize (bool): Whether to use normalization.
            audio_encoder_type (str): Type of audio encoder.
            audio_encoder_params (Dict[str, Union[float, int, bool, str]]): Parameters for audio encoder.
            metric_vocab_size (Optional[int]): Vocabulary size for metrics.
            metric_token_info (Optional[Dict[str, Any]]): Information about metric tokens.
            metric2type (Optional[Dict[str, str]]): Dictionary mapping metric names to types.
            metric_pad_value (float): Padding value for metrics.
            metric_token_pad_value (int): Padding value for metric tokens.
            sequential_metrics (bool): Whether to use sequential metrics.
            vocab_size (Optional[int]): Vocabulary size for text encoder.
            ignore_id (int): Ignore ID for padding in text encoder.
            text_encoder_type (str): Type of text encoder.
            text_encoder_params (Dict[str, Union[float, int, bool, str]]): Parameters for text encoder.
            cross_attention_type (str): Type of cross attention module.
            cross_attention_params (Dict[str, Union[float, int]]): Parameters for cross attention module.
            metric_decoder_params (Dict[str, Union[float, int]]): Parameters for metric decoder module.
            use_rope_pos (bool): Whether to use RoPE positional encoding.
            lsm_weight (float): Label smoothing weight.
            sym_sos (str): Symbol for start of sequence.
            sym_eos (str): Symbol for end of sequence.
            **kwargs: Additional parameters.

        """
        super().__init__()

        # Precheck parameters
        if not sequential_metrics:
            raise ValueError(
                "sequential_metrics is required for ar-universa, please set it to True."
            )

        # Initialize parameters
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.metric_vocab_size = metric_vocab_size
        self.ignore_id = ignore_id
        self.use_ref_audio = use_ref_audio
        self.use_ref_text = use_ref_text
        self.embedding_size = embedding_size
        decoder_input_dim = embedding_size
        self.use_normalize = use_normalize
        self.search_module = None
        self.save_token_seq = False
        self.sequential_metrics = sequential_metrics

        # Metric information
        # NOTE(jiatong): not useful for ARUniversa, but keep it for future use
        self.metric_size = len(metric2id)
        self.metric2id = metric2id
        self.id2metric = {v: k for k, v in metric2id.items()}
        if metric2type is None:
            self.id2type = {i: "numerical" for i in range(self.metric_size)} #Terry Debug
        else:
            self.id2type = {
                i: metric2type.get(self.id2metric[i], "numerical")
                for i in range(self.metric_size)
            }

        self.metric_pad_value = metric_pad_value
        self.metric_token_pad_value = metric_token_pad_value
        self.metric_tokenizer = MetricTokenizer(
            metric_token_info, tokenize_metric=list(metric2id.keys())
        )

        # NOTE(jiatong): the ID is set in tokenizer for <sos> and <eos>
        # will need to make it more flexible in the future
        # refer to espnet2/unisersa/metric_tokenizer/metric_tokenizer.py
        self.sos = 2
        self.eos = 3

        # Initialize audio encoder
        if audio_encoder_type == "transformer":
            self.audio_encoder = TransformerEncoder(
                input_size=input_size,
                output_size=embedding_size,
                **audio_encoder_params,
            )
        else:
            raise ValueError(f"Not supported: {audio_encoder_type}")
        if self.use_normalize:
            self.normalize = UtteranceMVN(norm_means=True, norm_vars=True)

        # Initialize reference audio encoder
        if self.use_ref_audio:
            if audio_encoder_type == "transformer":
                self.ref_audio_encoder = TransformerEncoder(
                    input_size=input_size,
                    output_size=embedding_size,
                    **audio_encoder_params,
                )
            else:
                raise ValueError(f"Not supported: {audio_encoder_type}")
            decoder_input_dim += embedding_size
            if self.use_normalize:
                self.ref_normalize = UtteranceMVN(norm_means=True, norm_vars=True)

        # Initialize text encoder
        if self.use_ref_text:
            self.text_embedding = torch.nn.Embedding(
                vocab_size,
                embedding_size,
            )
            if text_encoder_type == "transformer":
                self.text_encoder = TransformerEncoder(
                    input_size=embedding_size,
                    output_size=embedding_size,
                    **text_encoder_params,
                )
            else:
                raise ValueError(f"Not supported: {text_encoder_type}")
            decoder_input_dim += embedding_size

        # Initialize cross attention
        if cross_attention_type == "multihead":
            self.cross_attention = MultiHeadedAttention(
                n_feat=embedding_size,
                **cross_attention_params,
            )
        else:
            raise ValueError(f"Not supported: {cross_attention_type}")

        self.decoder = TransformerDecoder(
            vocab_size=metric_vocab_size,
            encoder_output_size=decoder_input_dim,
            pos_enc_class=(
                RoPEPositionalEncoding if use_rope_pos else PositionalEncoding
            ),
            **metric_decoder_params,
        )

        self.ar_criterion = LabelSmoothingLoss(
            size=metric_vocab_size,
            padding_idx=metric_token_pad_value,
            smoothing=lsm_weight,
            normalize_length=True,
        )


    @typechecked
    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        metrics: Dict[str, torch.Tensor],
        ref_audio: Optional[torch.Tensor] = None,
        ref_audio_lengths: Optional[torch.Tensor] = None,
        ref_text: Optional[torch.Tensor] = None,
        ref_text_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate outputs and return the loss tensor.

        Args:
            audio (torch.Tensor): Input audio tensor (B, T).
            audio_lengths (torch.Tensor): Length of audio tensor (B,).
            metrics (torch.Tensor): Metrics tensor Dict[str, tensor (B,)].
            ref_audio (torch.Tensor): Reference audio tensor (B, T).
            ref_audio_lengths (torch.Tensor): Length of reference audio tensor (B,).
            ref_text (torch.Tensor): Reference text tensor (B, U).
            ref_text_lengths (torch.Tensor): Length of reference text tensor (B,).

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
                loss (torch.Tensor): Loss tensor.
                stats (Dict[str, torch.Tensor]): Statistics to be monitored.
                weight (torch.Tensor): Weight tensor.

        """
        assert "metric_token" in metrics, "metric_token is required in metrics"
        assert (
            "metric_token_lengths" in metrics
        ), "metric_token_lengths is required in metrics"
        metric_token, metric_token_lengths = (
            metrics["metric_token"],
            metrics["metric_token_lengths"],
        )

        batch_size = audio.shape[0]
        assert (
            metric_token_lengths.dim() == 1
        ), "metric_token_lengths should be 1D tensor, but received {}".format(
            metric_token_lengths.dim()
        )
        # Check that batch_size is unified
        assert (
            batch_size == audio_lengths.shape[0]
            and batch_size == metric_token.shape[0]
            and batch_size == metric_token_lengths.shape[0]
        ), "mismatch batch size with audio {}, metrics {}, metric_token {}".format(
            audio.shape[0], metrics.shape[0], metric_token.shape[0]
        )

        # for data-parallel
        metric_token = metric_token[:, : metric_token_lengths.max()]
        metric_token[metric_token == -1] = self.metric_token_pad_value
        # 2. Encode audio
        audio_enc, audio_enc_lengths = self.encode(
            audio,
            audio_lengths,
            ref_audio,
            ref_audio_lengths,
            ref_text,
            ref_text_lengths,
        )

        # 3. Metric Decoder
        loss_ar_decoder, acc_ar_decoder, value_ar_decoder = self._calc_decoder_loss(
            audio_enc, audio_enc_lengths, metric_token, metric_token_lengths
        )

        stats = {}
        stats["loss_ar_decoder"] = loss_ar_decoder.detach()
        stats["acc_ar_decoder"] = acc_ar_decoder
        stats["value_ar_decoder"] = value_ar_decoder

        # TODO(jiatong): add nar decoder loss
        # 4. Loss calculation
        loss = loss_ar_decoder

        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((-loss, stats, batch_size), loss.device)
        return loss, stats, weight

    @typechecked
    def encode(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        ref_audio: Optional[torch.Tensor] = None,
        ref_audio_lengths: Optional[torch.Tensor] = None,
        ref_text: Optional[torch.Tensor] = None,
        ref_text_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = audio.shape[0]

        use_ref_audio = self.use_ref_audio and ref_audio is not None
        use_ref_text = self.use_ref_text and ref_text is not None

        if use_ref_text:
            assert (
                ref_text.shape[0] == batch_size
            ), "mismatch batch size with ref_text {}".format(ref_text.shape[0])
            ref_text[ref_text == -1] = self.ignore_id
            # for data-parallel
            ref_text = ref_text[:, : ref_text_lengths.max()]

        # 1. Feats normalization
        if self.use_normalize:
            with autocast(False):
                feats, feats_lengths = self.normalize(audio, audio_lengths)
                if use_ref_audio:
                    ref_feats, ref_feats_lengths = self.ref_normalize(
                        ref_audio, ref_audio_lengths
                    )
                if use_ref_text:
                    ref_text_embed = self.text_embedding(ref_text)

        # 2. Encode audio
        audio_enc, audio_enc_lengths, _ = self.audio_encoder(feats, feats_lengths)
        if use_ref_audio:
            ref_audio_enc, ref_audio_enc_lengths, _ = self.ref_audio_encoder(
                ref_feats, ref_feats_lengths
            )
        if use_ref_text:
            ref_text_enc, ref_text_enc_lengths, _ = self.text_encoder(
                ref_text_embed, ref_text_lengths
            )

        # 3. Cross attention
        
        # 3. Cross attention
        enc_list = [audio_enc]

        if self.use_ref_audio:
            if ref_audio is not None:
                ref_audio_mask = (
                    ~make_pad_mask(ref_audio_enc_lengths).to(audio_enc.device).unsqueeze(1)
                )
                ref_audio_info = self.cross_attention(
                    audio_enc, ref_audio_enc, ref_audio_enc, ref_audio_mask
                )
                enc_list.append(ref_audio_info)
            else:
                # keep decoder_input_dim consistent if ref audio is disabled at inference
                enc_list.append(audio_enc.new_zeros(audio_enc.size(0), audio_enc.size(1), self.embedding_size))

        if self.use_ref_text:
            if ref_text is not None:
                ref_text_mask = (
                    ~make_pad_mask(ref_text_enc_lengths).to(audio_enc.device).unsqueeze(1)
                )
                ref_text_info = self.cross_attention(
                    audio_enc, ref_text_enc, ref_text_enc, ref_text_mask
                )
                enc_list.append(ref_text_info)
            else:
                enc_list.append(audio_enc.new_zeros(audio_enc.size(0), audio_enc.size(1), self.embedding_size))

        audio_enc = torch.cat(enc_list, dim=-1)

        
        
        # enc_list = [audio_enc]
        # if use_ref_audio:
        #     ref_audio_mask = (
        #         ~make_pad_mask(ref_audio_enc_lengths).to(audio_enc.device).unsqueeze(1)
        #     )
        #     ref_audio_info = self.cross_attention(
        #         audio_enc, ref_audio_enc, ref_audio_enc, ref_audio_mask
        #     )
        #     enc_list.append(ref_audio_info)
        # if use_ref_text:
        #     ref_text_mask = (
        #         ~make_pad_mask(ref_text_enc_lengths).to(audio_enc.device).unsqueeze(1)
        #     )
        #     ref_text_info = self.cross_attention(
        #         audio_enc, ref_text_enc, ref_text_enc, ref_text_mask
        #     )
        #     enc_list.append(ref_text_info)
        # audio_enc = torch.cat(enc_list, dim=-1)

        return audio_enc, audio_enc_lengths

    def _calc_decoder_loss(
        self,
        audio_enc: torch.Tensor,
        audio_enc_lengths: torch.Tensor,
        metric_token: torch.Tensor,
        metric_token_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate decoder loss.

        Args:
            audio_enc (torch.Tensor): Encoded audio tensor (B, T, D).
            audio_enc_lengths (torch.Tensor): Length of encoded audio tensor (B,).
            metric_token (torch.Tensor): Metric tokens tensor (B, U).
            metric_token_lengths (torch.Tensor): Length of metric tokens tensor (B,).

        Returns:
            loss_ar_decoder (torch.Tensor): Loss tensor for AR decoder.
            acc_ar_decoder (torch.Tensor): Accuracy tensor for AR decoder.
            value_ar_decoder (torch.Tensor): Value tensor for AR decoder.
        """

        ys_in_pad, ys_out_pad = add_sos_eos(
            metric_token, self.sos, self.eos, self.metric_token_pad_value
        )
        ys_in_lens = metric_token_lengths + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            audio_enc, audio_enc_lengths, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_ar_decoder = self.ar_criterion(decoder_out, ys_out_pad)
        acc_ar_decoder = th_accuracy(
            decoder_out.view(-1, self.metric_vocab_size),
            ys_out_pad,
            ignore_label=self.metric_token_pad_value,
        )
        acc_value_ar_decoder = th_accuracy(
            decoder_out[:, 1::2].reshape(-1, self.metric_vocab_size),
            ys_out_pad[:, 1::2],
            ignore_label=self.metric_token_pad_value,
        )

        return loss_ar_decoder, acc_ar_decoder, acc_value_ar_decoder

    @typechecked
    def set_inference(
        self,
        beam_size: int,
        metric_list: List[str],
        skip_meta_label_score: bool,
        save_token_seq: bool = False,
    ) -> None:
        """Set inference mode.

        Args:
            beam_size (int): Beam size for beam search.
            metric_list (List[str]): List of metrics to predict.
            skip_meta_label_score (bool): Whether to skip meta label score.
            save_token_seq (bool): Whether to save token sequence.
        """
        self.eval()
        scorers = {
            "metric_decoder": self.decoder,
        }
        weights = {"metric_decoder": 1.0}

        # NOTE(jiatong): add the metric token offset for beam search, this masking is used for pre-beam pruning
        beam_masking = {}
        for metric_name in self.metric_tokenizer.metric_offset.keys():
            metric_token = self.metric_tokenizer.vocab_indices[
                "{}@meta_label".format(metric_name)
            ]
            start_idx, num_idx = self.metric_tokenizer.metric_offset[metric_name]
            # +2 to skip the meta label token and padding token
            start_idx = start_idx + self.metric_tokenizer.overall_offset + 2
            end_idx = start_idx + (num_idx - 2)
            beam_masking[metric_token] = (start_idx, end_idx)

        self.save_token_seq = save_token_seq

        self.search_module = ARUniVERSABeamSearch(
            scorers=scorers,
            weights=weights,
            beam_size=beam_size,
            vocab_size=self.metric_vocab_size,
            sos=self.sos,
            eos=self.eos,
            meta_label_for_search=[
                self.metric_tokenizer.get_metric_meta_label(metric)
                for metric in metric_list
            ],
            token_list=self.metric_tokenizer.get_token_list(),
            skip_meta_label_score=skip_meta_label_score,
            beam_masking=beam_masking,
        )

    @typechecked
    def inference(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        ref_audio: Optional[torch.Tensor] = None,
        ref_audio_lengths: Optional[torch.Tensor] = None,
        ref_text: Optional[torch.Tensor] = None,
        ref_text_lengths: Optional[torch.Tensor] = None,
        # add to inference() signature (end of args)
        chunk_seconds: float = 3.0,
        segment_mode: str = "sliding",       # "single" or "sliding"
        segment_strategy: str = "center",   # "center"|"random"|"begin"|"end"
        assumed_sr: int = 16000,            # used only if you pass seconds
        # ADD in inference() signature (after existing segment args)
        overlap: float = 0.5,            # 50% hop by default
        return_segment_outputs: bool = False,  # optionally return per-segment metrics
        **kwargs,
    ) -> Dict[str, Union[np.array, torch.Tensor]]:
        """Return predicted output as a dict.

        Args:
            audio (torch.Tensor): Input audio tensor (B, T).
            audio_lengths (torch.Tensor): Length of audio tensor (B,).
            ref_audio (torch.Tensor): Reference audio tensor (B, T).
            ref_audio_lengths (torch.Tensor): Length of reference audio tensor (B,).
            ref_text (torch.Tensor): Reference text tensor (B, U).
            ref_text_lengths (torch.Tensor): Length of reference text tensor (B,).
            metric_list (Optional[List[str]]): List of metrics to predict.
            **kwargs: Additional parameters.

        Returns:
            Dict[str, torch.Tensor]: Predicted output.

        """

        if self.search_module is None:
            self.set_inference(
                beam_size=1,
                metric_list=list(self.metric2id.keys()),
                skip_meta_label_score=False,
            )
        # --- segment-only inference (no concat) ---
        if chunk_seconds is not None and segment_mode == "single":
            # convert seconds to samples (assume 16k unless you pass assumed_sr)
            L = int(chunk_seconds * assumed_sr)
            # audio shape: (B=1, T). We only support B=1 in inference.
            assert audio.size(0) == 1, "segment-only inference expects batch size 1"
            T = audio.size(1)
            L = min(L, T)
            # pick start index
            if segment_strategy == "center":
                s = max(0, (T - L) // 2)
            elif segment_strategy == "begin":
                s = 0
            elif segment_strategy == "end":
                s = max(0, T - L)
            elif segment_strategy == "random":
                s = torch.randint(0, max(1, T - L + 1), (1,), device=audio.device).item()
            else:
                raise ValueError(f"Unknown segment_strategy: {segment_strategy}")
            # crop the audio to the chosen segment
            audio = audio[:, s:s+L]
            audio_lengths = torch.tensor([L], device=audio.device)
            # make sure we don't accidentally try to use references
            ref_audio = None
            ref_audio_lengths = None
            ref_text = None
            ref_text_lengths = None
        # --- end segment-only block ---
        
        
        # --- sliding-window inference (multiple segments, 50% overlap by default) ---
        if chunk_seconds is not None and segment_mode == "sliding":
            assert audio.size(0) == 1, "sliding-window inference expects batch size 1"
            assert chunk_seconds > 0.0 and 0.0 < overlap < 1.0, "bad chunk_seconds/overlap"

            sr = assumed_sr
            L = int(chunk_seconds * sr)
            T = audio.size(1)
            if L <= 0 or T <= 0:
                raise ValueError(f"Invalid lengths: L={L}, T={T}")

            hop = max(1, int(L * (1.0 - overlap)))
            # generate start indices covering [0, T), include a final tail-aligned window
            starts = list(range(0, max(1, T - L + 1), hop))
            if len(starts) == 0 or starts[-1] != max(0, T - L):
                starts.append(max(0, T - L))

            seg_metrics_list = []
            seg_tokenseq_list = []
            seg_encoded_feats = []

            # Disable refs for segment tests
            ref_audio = None
            ref_audio_lengths = None
            ref_text = None
            ref_text_lengths = None

            for s in starts:
                e = s + L
                seg_audio = audio[:, s:e]
                seg_len = seg_audio.size(1)

                # Guard against very short tail (<1 sample)
                if seg_len <= 0:
                    continue

                seg_audio_lengths = torch.tensor([seg_len], device=audio.device)

                # Encode this segment
                seg_enc, _ = self.encode(
                    seg_audio, seg_audio_lengths,
                    ref_audio, ref_audio_lengths,
                    ref_text, ref_text_lengths,
                )

                # Beam search on this segment
                if self.search_module is None:
                    raise ValueError("Inference module is not set. Please call set_inference() first.")
                nbest = self.search_module.forward(seg_enc[0])
                assert len(nbest) > 0 and len(nbest[0].yseq) > 0
                yseq = nbest[0].yseq

                # Token -> metric dict for this segment
                seg_pred = self.metric_tokenizer.tokenseq2metric(yseq, return_dict=True)
                if self.save_token_seq:
                    seg_pred["token_seq"] = [int(t) for t in yseq]  # 1D for this segment

                # Collect
                seg_metrics_list.append(seg_pred)
                if self.save_token_seq:
                    seg_tokenseq_list.append([int(t) for t in yseq])
                seg_encoded_feats.append(seg_enc)

            # Aggregate across segments
            from collections import Counter
            agg_pred = {}
            keys = set().union(*[d.keys() for d in seg_metrics_list]) if seg_metrics_list else set()
            for k in keys:
                vals = [d[k] for d in seg_metrics_list if k in d]
                # Try numeric aggregation
                num_list = []
                for v in vals:
                    if isinstance(v, (int, float, np.number)):
                        num_list.append(float(v))
                    elif torch.is_tensor(v) and v.dim() == 0:
                        num_list.append(float(v.item()))
                if len(num_list) == len(vals) and len(num_list) > 0:
                    agg_pred[k] = float(np.mean(num_list))
                else:
                    # Fall back: majority vote for non-numeric / mixed
                    try:
                        agg_pred[k] = Counter(vals).most_common(1)[0][0]
                    except Exception:
                        agg_pred[k] = vals[0] if vals else None

            # Optionally attach per-segment details
            if return_segment_outputs:
                agg_pred["_segment_starts"] = starts
                agg_pred["_segment_seconds"] = chunk_seconds
                agg_pred["_segment_overlap"] = overlap
                agg_pred["_per_segment"] = seg_metrics_list
                if self.save_token_seq:
                    agg_pred["_token_seqs"] = seg_tokenseq_list

            # Keep flags consistent with your previous outputs
            agg_pred["use_tokenizer_metrics"] = True
            agg_pred["sequential_metrics"] = True
            # You can skip encoded_feat here (multiple segments). If you want them:
            # agg_pred["encoded_feat"] = seg_encoded_feats

            return agg_pred
        # --- end sliding-window block ---

        

        # 1. Encode audio
        audio_enc, _ = self.encode(
            audio,
            audio_lengths,
            ref_audio,
            ref_audio_lengths,
            ref_text,
            ref_text_lengths,
        )

        assert audio_enc.size(0) == 1, "Inference only supports batch size of 1."

        # 2. Inference
        if self.search_module is None:
            raise ValueError(
                "Inference module is not set. Please call set_inference() first."
            )
        nbest_hyps = self.search_module.forward(audio_enc[0])

        # NOTE(jiatong): get the top one hypothesis
        assert len(nbest_hyps) > 0, "nbest_hyps should not be empty"
        assert len(nbest_hyps[0].yseq) > 0, "nbest_hyps[0].yseq should not be empty"
        pred_metrics = nbest_hyps[0].yseq

        # 3. Decorate the predicted metrics
        pred_metrics = self.metric_tokenizer.tokenseq2metric(
            pred_metrics, return_dict=True
        )

        if self.save_token_seq:
            pred_metrics["token_seq"] = [[int(token) for token in nbest_hyps[0].yseq]]

        pred_metrics["use_tokenizer_metrics"] = True
        pred_metrics["sequential_metrics"] = True
        pred_metrics["encoded_feat"] = audio_enc
        return pred_metrics
