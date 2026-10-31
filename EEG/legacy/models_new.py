"""Architectures B-D and F. A is SleepApneaCNN from model.py. Every model maps x [B,C,3750] -> one logit [B].
C=1 for single-window models, C=3 (prev, cur, next window) for the context model F."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SleepApneaCNN


class CNNA(nn.Module):
    """Barnes et al. replica (2 logits) collapsed to a single logit so every arch shares one BCE loss."""

    def __init__(self):
        super().__init__()
        self.net = SleepApneaCNN()

    def forward(self, x):
        o = self.net(x)
        return o[:, 1] - o[:, 0]


class Block(nn.Module):
    def __init__(self, cin, cout, stride, k=7):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = None if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(h + (x if self.skip is None else self.skip(x)))


class ResEncoder(nn.Module):
    """Small 1D ResNet trunk -> [B, 128] (global avg + max pooled over time, then projected)."""

    def __init__(self, width=(16, 32, 64, 128), dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, width[0], 15, 2, 7, bias=False), nn.BatchNorm1d(width[0]), nn.ReLU(),
                                  nn.MaxPool1d(2))
        blocks, c = [], width[0]
        for w in width:
            blocks += [Block(c, w, 2 if w != width[0] else 1), Block(w, w, 1)]
            c = w
        self.blocks = nn.Sequential(*blocks)
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c, 128), nn.ReLU())
        self.out_dim = 128

    def forward(self, x):
        h = self.blocks(self.stem(x))
        return self.proj(torch.cat([h.mean(-1), h.amax(-1)], 1))


class ResNet1D(nn.Module):  # B
    def __init__(self):
        super().__init__()
        self.enc = ResEncoder()
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        return self.head(self.enc(x)).squeeze(-1)


class CNNBiGRU(nn.Module):  # C
    def __init__(self, hidden=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 11, 2, 5, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 7, 1, 3, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 64, 5, 1, 2, bias=False), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))  # 3750 -> 58 steps
        self.gru = nn.GRU(64, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(4 * hidden, 1))

    def forward(self, x):
        h, _ = self.gru(self.cnn(x).transpose(1, 2))
        return self.head(torch.cat([h.mean(1), h.amax(1)], 1)).squeeze(-1)


class SpecCNN(nn.Module):  # D: fixed-DFT conv "STFT" (MPS-safe) + 2D CNN
    def __init__(self, n_fft=250, hop=50, fs=125, fmax=42):
        super().__init__()
        self.nbins = int(fmax * n_fft / fs) + 1  # 2 s Hann window, 0.5 Hz bins, up to 42 Hz
        k = torch.arange(n_fft)
        f = torch.arange(self.nbins)[:, None]
        win = torch.hann_window(n_fft)
        ang = 2 * math.pi * f * k / n_fft
        basis = torch.cat([torch.cos(ang) * win, -torch.sin(ang) * win], 0)[:, None, :]  # [2*nbins,1,n_fft]
        self.register_buffer("basis", basis)
        self.hop = hop

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.cnn = nn.Sequential(blk(1, 16), blk(16, 32), blk(32, 64), blk(64, 64))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        s = F.conv1d(x, self.basis, stride=self.hop)  # [B, 2*nbins, frames]
        re, im = s[:, :self.nbins], s[:, self.nbins:]
        spec = torch.log1p(torch.sqrt(re * re + im * im + 1e-8))[:, None]  # [B,1,freq,time]
        spec = (spec - spec.mean((2, 3), keepdim=True)) / (spec.std((2, 3), keepdim=True) + 1e-5)
        return self.head(self.cnn(spec).mean((2, 3))).squeeze(-1)


class ContextNet(nn.Module):  # F: shared ResNet encoder on prev/cur/next (90 s), concat -> MLP
    def __init__(self):
        super().__init__()
        self.enc = ResEncoder()
        self.head = nn.Sequential(nn.Linear(3 * 128, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):  # x [B,3,T]
        b = x.shape[0]
        h = self.enc(x.reshape(b * 3, 1, -1)).reshape(b, -1)
        return self.head(h).squeeze(-1)


ARCHS = {"A": CNNA, "B": ResNet1D, "C": CNNBiGRU, "D": SpecCNN, "F": ContextNet}
CONTEXT = {"F"}


def make_model(arch):
    return ARCHS[arch]()
