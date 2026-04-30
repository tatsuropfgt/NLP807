"""Linear probe head over a frozen audio encoder.

For SUPERB-style frozen evaluation: the encoder's weights are not updated;
only a single linear classifier is trained on top of the per-frame features.
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
