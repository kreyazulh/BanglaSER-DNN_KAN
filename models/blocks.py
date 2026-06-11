"""Factorized DNN block for the spectral stream (eq 2 in the paper).

Note: the camera-ready adds the projection shortcut explicitly. when
in_dim != out_dim the residual goes through a linear projection, otherwise
the addition wouldn't typecheck. d_mid = sqrt(in*out), rounded.
"""

import math

import torch.nn as nn


class FactorizedDNNBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        mid = int(round(math.sqrt(in_dim * out_dim)))  # ~181 for 128x256

        self.w1 = nn.Linear(in_dim, mid)
        self.bn1 = nn.BatchNorm1d(mid)
        self.w2 = nn.Linear(mid, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.act1 = nn.ReLU()
        self.act2 = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # projection shortcut when dims don't match
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.act1(self.bn1(self.w1(x)))
        h = self.act2(self.bn2(self.w2(h)))
        return self.drop(h) + self.shortcut(x)
