from typing import Optional, Tuple

import torch
from typeguard import typechecked

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class LabelAggregate(torch.nn.Module):
    @typechecked
    def __init__(
        self,
        win_length: int = 512,
        hop_length: int = 128,
        center: bool = True,
        num_spk: Optional[int] = None,     # NEW
        reduce_mode: str = "topk",         # NEW: "topk" or "first"
        **kwargs,                          # NEW: ignore unknown YAML keys
    ):
        super().__init__()

        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.num_spk = num_spk
        assert reduce_mode in ("topk", "first")
        self.reduce_mode = reduce_mode

    def extra_repr(self):
        return (
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
            f"num_spk={self.num_spk}, "
            f"reduce_mode={self.reduce_mode}"
        )

    def _clamp_num_spk(self, out: torch.Tensor) -> torch.Tensor:
        """Ensure out has last-dim == num_spk by padding or reducing.

        out: [B, T, S_found]
        returns: [B, T, num_spk] if self.num_spk is set, else unchanged
        """
        if self.num_spk is None:
            return out
        B, T, S = out.shape
        if S == self.num_spk:
            return out
        if S < self.num_spk:
            pad = out.new_zeros(B, T, self.num_spk - S)
            return torch.cat([out, pad], dim=2)

        # S > num_spk: keep either the most-active speakers or first K
        if self.reduce_mode == "first":
            return out[:, :, : self.num_spk]

        # "topk": choose speakers with largest total activity over time per item
        # activity: [B, S]
        activity = out.sum(dim=1)
        # indices: [B, num_spk]
        topk_idx = torch.topk(activity, k=self.num_spk, dim=1).indices
        # gather needs [B, T, num_spk]
        gather_idx = topk_idx.unsqueeze(1).expand(B, T, self.num_spk)
        return torch.gather(out, dim=2, index=gather_idx)

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """LabelAggregate forward function.

        Args:
            input: (Batch, Nsamples, Label_dim)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Label_dim)

        """
        bs = input.size(0)
        max_length = input.size(1)
        label_dim = input.size(2)

        # NOTE(jiatong):
        #   The default behaviour of label aggregation is compatible with
        #   torch.stft about framing and padding.

        # Step1: center padding
        if self.center:
            pad = self.win_length // 2
            max_length = max_length + 2 * pad
            input = torch.nn.functional.pad(input, (0, 0, pad, pad), "constant", 0)
            input[:, :pad, :] = input[:, pad : (2 * pad), :]
            input[:, (max_length - pad) : max_length, :] = input[
                :, (max_length - 2 * pad) : (max_length - pad), :
            ]
            nframe = (
                torch.div(
                    max_length - self.win_length, self.hop_length, rounding_mode="trunc"
                )
                + 1
            )

        # Step2: framing
        output = input.as_strided(
            (bs, nframe, self.win_length, label_dim),
            (max_length * label_dim, self.hop_length * label_dim, label_dim, 1),
        )

        # Step3: aggregate label
        output = torch.gt(output.sum(dim=2, keepdim=False), self.win_length // 2)
        output = output.float()

        output = self._clamp_num_spk(output)
        
        # Step4: process lengths
        if ilens is not None:
            if self.center:
                pad = self.win_length // 2
                ilens = ilens + 2 * pad

            olens = (
                torch.div(
                    ilens - self.win_length, self.hop_length, rounding_mode="trunc"
                )
                + 1
            )
            output.masked_fill_(make_pad_mask(olens, output, 1), 0.0)
        else:
            olens = None

        return output, olens
