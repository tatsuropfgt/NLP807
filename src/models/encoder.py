"""Mel-spectrogram + Transformer encoder.

Pipeline: waveform → log-mel → linear projection → +PE → Transformer → frame
features. Designed so that masking can be inserted between the projection and
the Transformer for self-supervised pretraining (see msm.py).

Frame rate at default settings (16 kHz, hop=160) is 100 Hz.
"""

from __future__ import annotations

import math

import torch
import torchaudio.transforms as taT
from torch import nn


class LogMelSpec(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        log_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.mel = taT.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.log_eps = log_eps
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, T_wav)  ->  mel: (B, T_frames, n_mels)
        m = self.mel(wav)
        m = torch.log(m + self.log_eps)
        return m.transpose(1, 2).contiguous()


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MelTransformerEncoder(nn.Module):
    """Reusable encoder. Used both for pretraining and as a frozen feature
    extractor for downstream probes.

    Forward pass yields frame-level features at the mel-spec frame rate.
    For pretraining, the steps `compute_mel`, `project`, `contextualize` are
    exposed so a mask token can be substituted between projection and the
    Transformer.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        max_frames: int = 2048,
    ) -> None:
        super().__init__()
        self.melspec = LogMelSpec(sample_rate=sample_rate, n_mels=n_mels)
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_frames)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.n_mels = n_mels

    def compute_mel(self, wav: torch.Tensor) -> torch.Tensor:
        return self.melspec(wav)

    def project(self, mel: torch.Tensor) -> torch.Tensor:
        return self.input_proj(mel)

    def contextualize(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.norm(x)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.contextualize(self.project(self.compute_mel(wav)))
