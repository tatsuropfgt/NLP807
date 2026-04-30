"""Masked Spectrogram Modeling.

Self-supervised pretraining objective:
- Compute log-mel from waveform.
- Standardize per-utterance to make MSE well-conditioned.
- Project mel to model dim.
- Sample mask spans (wav2vec 2.0 style: each timestep is a span start with
  probability `mask_prob`; spans of length `mask_length` may overlap).
- Replace masked positions with a learnable mask token.
- Run the Transformer; predict standardized mel at masked frames.
- MSE on masked frames only.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.encoder import MelTransformerEncoder


def compute_mask_indices(
    shape: tuple[int, int],
    mask_prob: float,
    mask_length: int,
    device: torch.device | str,
    min_masks: int = 1,
) -> torch.Tensor:
    """Wav2vec2-style static masking. Returns bool mask of shape (B, T)."""
    B, T = shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    if T < mask_length:
        return mask

    valid_starts = T - mask_length + 1
    for b in range(B):
        n_starts = max(min_masks, int(round(valid_starts * mask_prob)))
        # Sample without replacement among valid start positions.
        starts = torch.randperm(valid_starts, device=device)[:n_starts]
        for s in starts.tolist():
            mask[b, s : s + mask_length] = True
    return mask


class MaskedSpecModel(nn.Module):
    """Wraps an encoder with a mask-and-reconstruct objective."""

    def __init__(
        self,
        encoder: MelTransformerEncoder,
        mask_prob: float = 0.065,
        mask_length: int = 10,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_token = nn.Parameter(torch.zeros(encoder.d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.head = nn.Linear(encoder.d_model, encoder.n_mels)

    @staticmethod
    def _standardize(mel: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        # Per-utterance, across (T, n_mels)
        mean = mel.mean(dim=(1, 2), keepdim=True)
        std = mel.std(dim=(1, 2), keepdim=True)
        return (mel - mean) / (std + eps)

    def forward(self, wav: torch.Tensor) -> dict[str, torch.Tensor]:
        mel = self.encoder.compute_mel(wav)  # (B, T, n_mels)
        mel_n = self._standardize(mel)
        target = mel_n.detach()

        x = self.encoder.project(mel_n)  # (B, T, d)
        B, T, _ = x.shape
        mask = compute_mask_indices(
            (B, T), self.mask_prob, self.mask_length, x.device
        )  # (B, T) bool
        # Substitute mask token at masked positions
        x = torch.where(mask.unsqueeze(-1), self.mask_token.expand_as(x), x)

        h = self.encoder.contextualize(x)  # (B, T, d)
        pred = self.head(h)  # (B, T, n_mels)

        if mask.any():
            loss = ((pred - target) ** 2)[mask].mean()
        else:
            loss = torch.zeros((), device=x.device, requires_grad=True)

        return {
            "loss": loss,
            "mask_ratio": mask.float().mean(),
            "n_masked": mask.sum(),
            "n_total": torch.tensor(mask.numel(), device=mask.device),
        }
