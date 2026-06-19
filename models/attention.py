"""Early stage (128d, 4 heads): spectral queries attend to prosodic and vice
versa, locating where pitch events sit in the mfcc stream. Late stage
(256d, 8 heads): roles reverse for high level integration. 
"""

import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, dim, n_heads, residual_weight=0.5):
        super().__init__()
        # one mha reused for both directions, keeps params near the 1.1M budget
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.w = residual_weight

    def forward(self, a, b):
        # treat each vector as a length-1 sequence, mha wants (B, T, D)
        a_ = a.unsqueeze(1)
        b_ = b.unsqueeze(1)

        a2b, _ = self.attn(a_, b_, b_)  # a queries b
        b2a, _ = self.attn(b_, a_, a_)  # b queries a

        a_out = a + self.w * a2b.squeeze(1)
        b_out = b + self.w * b2a.squeeze(1)
        return a_out, b_out
