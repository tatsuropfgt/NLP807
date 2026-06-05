"""Linear probe heads over a frozen audio encoder.

For SUPERB-style frozen evaluation: the encoder's weights are not updated;
only a single linear classifier is trained on top of the encoder features.
Two variants are provided:
- :class:`FrozenLinearProbe` — per-frame linear (phoneme cls, boundary, F0).
- :class:`FrozenUtteranceProbe` — masked mean-pool over time, then linear
  (KS, SID, ER — any utterance-level classification).
"""

from __future__ import annotations

import torch
from torch import nn

from src.data.alignments import N_PHONES
from src.models.encoder import MelTransformerEncoder


class FrozenLinearProbe(nn.Module):
    """Frozen ``encoder`` + a single linear classification head.

    The encoder is set to ``eval()`` and its parameters have ``requires_grad
    = False``. A no-grad block wraps the encoder forward so activations are
    not retained for backprop. Only ``head`` is trainable.
    """

    def __init__(
        self,
        encoder: MelTransformerEncoder,
        n_classes: int = N_PHONES,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.head = nn.Linear(encoder.d_model, n_classes)

    def train(self, mode: bool = True):
        # Always keep the encoder in eval mode (no dropout / fixed running
        # stats). Only the head toggles between train/eval.
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def features(self, wav: torch.Tensor) -> torch.Tensor:
        return self.encoder(wav)  # (B, T, d_model)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        feats = self.features(wav)  # (B, T, d_model)
        return self.head(feats)  # (B, T, n_classes)


class FrozenUtteranceProbe(nn.Module):
    """Frozen encoder + masked mean-pool + linear classification head.

    Forward expects a wav batch ``(B, T_wav)`` plus ``wav_lens (B,)`` giving
    the valid sample count per item; padded frames are masked out of the mean
    pool. ``wav_lens`` is required: SUPERB-style utterance tasks pad to the
    longest in the batch and silently averaging the zero pad would skew
    short utterances.
    """

    def __init__(
        self,
        encoder: MelTransformerEncoder,
        n_classes: int,
        hop_length: int = 160,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()
        self.head = nn.Linear(encoder.d_model, n_classes)
        self.hop_length = hop_length

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def features(self, wav: torch.Tensor) -> torch.Tensor:
        return self.encoder(wav)  # (B, T_frames, d_model)

    def forward(self, wav: torch.Tensor, wav_lens: torch.Tensor) -> torch.Tensor:
        feats = self.features(wav)  # (B, T, d)
        T = feats.shape[1]
        # MelSpectrogram(center=True): frame_count = floor(n_samples/hop) + 1
        frame_lens = (wav_lens // self.hop_length + 1).clamp(min=1, max=T)
        idx = torch.arange(T, device=feats.device).unsqueeze(0)
        mask = (idx < frame_lens.unsqueeze(1)).to(feats.dtype).unsqueeze(-1)  # (B, T, 1)
        summed = (feats * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom  # (B, d)
        return self.head(pooled)  # (B, n_classes)


def load_encoder(
    state_dict_path: str | None,
    sample_rate: int = 16000,
    n_mels: int = 80,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 6,
    ffn_dim: int = 1024,
    dropout: float = 0.0,
) -> MelTransformerEncoder:
    """Build an encoder and optionally load pretrained weights.

    Pass ``state_dict_path=None`` for the **random init** baseline (condition F).
    Note: dropout defaults to 0 here because the encoder is frozen at probe time.
    """
    encoder = MelTransformerEncoder(
        sample_rate=sample_rate,
        n_mels=n_mels,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
    )
    if state_dict_path is not None:
        sd = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        missing, unexpected = encoder.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"load_encoder: missing={missing}, unexpected={unexpected}")
    return encoder
