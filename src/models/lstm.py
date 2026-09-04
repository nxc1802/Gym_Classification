"""
LSTM and Bidirectional LSTM architectures for temporal landmark sequence classification.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    """
    Standard LSTM Classifier for temporal feature sequences.
    Input: (B, T, D)
    Output: (B, num_classes)
    """
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (B, T, hidden_dim)
        last_step = out[:, -1, :]  # (B, hidden_dim)
        return self.fc(last_step)

class BiLSTMModel(nn.Module):
    """
    Bidirectional LSTM Classifier for temporal feature sequences.
    Input: (B, T, D)
    Output: (B, num_classes)
    """
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.bilstm(x)  # (B, T, hidden_dim * 2)
        last_step = out[:, -1, :]
        return self.fc(last_step)

class BranchConcatModel(nn.Module):
    """
    Dual-branch LSTM model:
    Branch 1 processes relative coordinates (dim1, e.g. 49)
    Branch 2 processes joint angles (dim2, e.g. 286)
    Their final hidden representations are concatenated and fed to a classifier.
    """
    def __init__(
        self,
        dim1: int,
        dim2: int,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        # Branch 1
        self.b1_fc = nn.Linear(dim1, hidden_dim)
        self.b1_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Branch 2
        self.b2_fc = nn.Linear(dim2, hidden_dim)
        self.b2_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

        # Merge classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x1, x2 = x
        h1 = F.relu(self.b1_fc(x1))
        out1, _ = self.b1_lstm(h1)
        r1 = out1[:, -1, :]

        h2 = F.relu(self.b2_fc(x2))
        out2, _ = self.b2_lstm(h2)
        r2 = out2[:, -1, :]

        merged = torch.cat([r1, r2], dim=1)
        return self.classifier(merged)
