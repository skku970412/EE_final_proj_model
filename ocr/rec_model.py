"""CRNN recognition model definition for license plate OCR."""

from __future__ import annotations

import torch
import torch.nn as nn


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for sequence recognition."""

    def __init__(self, nclass: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 1/2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 1/4
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 1/8 on height
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 1/16 on height
        )
        self.rnn = nn.LSTM(
            input_size=512, hidden_size=256, num_layers=2, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(512, nclass)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)  # B x 512 x 1 x W'
        batch, channels, height, width = feat.size()
        if height != 1:
            raise ValueError(f"Expected feature height == 1 after CNN, got {height}")
        seq = feat.squeeze(2).permute(0, 2, 1)  # B x W' x 512
        seq, _ = self.rnn(seq)
        logits = self.fc(seq)
        return logits
