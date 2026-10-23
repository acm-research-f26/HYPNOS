"""Compact 1D CNN. Input: [batch, 3, 300] 30 seconds at 10 Hz."""
import torch.nn as nn


class ActigraphyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        return self.classifier(self.features(x).squeeze(-1))
