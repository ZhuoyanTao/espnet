# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Universa abstrast class."""

from abc import ABC, abstractmethod
from typing import Dict, Tuple
from typing import Any, Dict, List, Optional, Tuple
import torch


class AbsUniversa(torch.nn.Module, ABC):
    """Universa abstract class."""

    @abstractmethod
    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        metrics: torch.Tensor,
        ref_audio: torch.Tensor,
        ref_audio_lengths: torch.Tensor,
        ref_text: torch.Tensor,
        ref_text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate outputs and return the loss tensor."""
        raise NotImplementedError

    @abstractmethod
    def inference(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Return predicted output as a dict."""
        raise NotImplementedError
    
    def inference_with_metric_list(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        metric_list: Optional[List[str]] = None,
        **kwargs,
    ):
        """Inference with an explicit metric_list (wrapper around set_inference+inference)."""
        if metric_list is None:
            metric_list = list(self.metric2id.keys())

        # Reset search module for this metric list
        self.set_inference(
            beam_size=1,
            metric_list=metric_list,
            skip_meta_label_score=False,
            save_token_seq=False,
        )
        return self.inference(audio, audio_lengths, **kwargs)


    @property
    def require_raw_audio(self):
        """Return whether or not raw_audio is required."""
        return False

    @property
    def require_raw_text(self):
        """Return whether or not raw_text is required."""
        return False
