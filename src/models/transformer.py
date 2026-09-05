"""
Transformer Encoder architecture for temporal sequence modeling of gym exercise landmarks.
Uses Positional Encoding + Multi-Head Self-Attention layers (4 layers, 8 heads).
"""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TransformerModel(nn.Module):
    """
    Calibrated Transformer Encoder Classifier (~355K params).
    - Input LayerNorm on feature dimension
    - Linear projection to d_model
    - Sinusoidal Positional Encoding
    - 3 layers of TransformerEncoderLayer (8 attention heads, ff_dim=192)
    - Global Average Pooling over time + Classification Head
    """
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 192,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(feat_dim)
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x_norm = self.in_norm(x)
        h = self.input_proj(x_norm)
        h = self.pos_encoder(h)
        encoded = self.transformer_encoder(h)  # (B, T, d_model)
        encoded = self.norm(encoded)
        pooled = encoded.mean(dim=1)  # (B, d_model)
        return self.fc(pooled)

class BranchConcatTransformer(nn.Module):
    """
    Dual-branch Transformer model for processing (Rel_Landmarks, Angles).
    Both branches pass through independent Transformer Encoders before fusion.
    """
    def __init__(
        self,
        dim1: int,
        dim2: int,
        num_classes: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        # Branch 1
        self.proj1 = nn.Linear(dim1, d_model)
        self.pos1 = PositionalEncoding(d_model)
        layer1 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True)
        self.enc1 = nn.TransformerEncoder(layer1, num_layers=num_layers)

        # Branch 2
        self.proj2 = nn.Linear(dim2, d_model)
        self.pos2 = PositionalEncoding(d_model)
        layer2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True)
        self.enc2 = nn.TransformerEncoder(layer2, num_layers=num_layers)

        # Merge
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = x
        h1 = self.enc1(self.pos1(self.proj1(x1))).mean(dim=1)
        h2 = self.enc2(self.pos2(self.proj2(x2))).mean(dim=1)
        merged = torch.cat([h1, h2], dim=1)
        return self.classifier(merged)
