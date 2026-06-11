"""The full dual-stream model (figure 1).

spectral (80d mfcc stats) -> proj 128 -> DNN block -> [early x-attn]
prosodic (51d)            -> proj 128 -> KAN stack -> [early x-attn]
then both 128->256, [late x-attn], emotion-adaptive gating, concat 512,
bottleneck 512-384-512, classifier 512-256-128-C.

gating (eq 4): g = sigmoid(Wg relu(Wr [s;p])) in R^{C x 2}, then weighted
by the model's OWN predicted emotion probabilities from an aux head,
g_final = sum_e p_e * g_e. trained end to end, no warmup, no teacher
forcing (it just works, early noise washes out).
"""

import torch
import torch.nn as nn

from .blocks import FactorizedDNNBlock
from .kan import KANStack
from .attention import CrossAttention


class DualStreamDNNKAN(nn.Module):
    def __init__(self, n_classes, spectral_dim=80, prosodic_dim=51, dropout=0.3):
        super().__init__()
        self.n_classes = n_classes

        # independent input projections to 128
        self.proj_s = nn.Sequential(nn.Linear(spectral_dim, 128), nn.LayerNorm(128))
        self.proj_p = nn.Sequential(nn.Linear(prosodic_dim, 128), nn.LayerNorm(128))

        # layer 1: specialized processing at 128
        self.dnn1 = FactorizedDNNBlock(128, 128, dropout)
        self.kan1 = KANStack(128, 128, dropout)

        self.early_attn = CrossAttention(dim=128, n_heads=4)

        # layer 2: widen to 256
        self.dnn2 = FactorizedDNNBlock(128, 256, dropout)
        self.kan2 = KANStack(128, 256, dropout)

        self.late_attn = CrossAttention(dim=256, n_heads=8)

        # emotion-adaptive gating, eq 4. [s;p] is 512d
        # Wr: 512 -> 256, Wg: 256 -> 2*C, reshaped to (C, 2)
        self.gate_r = nn.Linear(512, 256)
        self.gate_g = nn.Linear(256, 2 * n_classes)

        # aux head produces the emotion probabilities that weight the gates
        self.aux_head = nn.Linear(512, n_classes)

        # bottleneck fusion + classifier
        self.fusion = nn.Sequential(
            nn.Linear(512, 384), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(384, 512), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x_spec, x_pros, return_aux=False):
        s = self.proj_s(x_spec)
        p = self.proj_p(x_pros)

        s = self.dnn1(s)
        p = self.kan1(p)

        s, p = self.early_attn(s, p)

        s = self.dnn2(s)
        p = self.kan2(p)

        s, p = self.late_attn(s, p)

        sp = torch.cat([s, p], dim=-1)  # (B, 512)

        # gates per emotion: (B, C, 2)
        g = torch.sigmoid(self.gate_g(torch.relu(self.gate_r(sp))))
        g = g.view(-1, self.n_classes, 2)

        # weight emotion-specific gates by predicted probs
        aux_logits = self.aux_head(sp)
        probs = torch.softmax(aux_logits, dim=-1)            # (B, C)
        g_final = (probs.unsqueeze(-1) * g).sum(dim=1)       # (B, 2)

        s = s * g_final[:, 0:1]
        p = p * g_final[:, 1:2]

        fused = self.fusion(torch.cat([s, p], dim=-1))
        logits = self.classifier(fused)

        if return_aux:
            return logits, aux_logits
        return logits


def count_params(model):
    return sum(t.numel() for t in model.parameters() if t.requires_grad)


if __name__ == "__main__":
    # quick smoke test: python -m models.network
    m = DualStreamDNNKAN(n_classes=7)
    x1 = torch.randn(4, 80)
    x2 = torch.rand(4, 51) * 2 - 1
    out = m(x1, x2)
    print("logits:", out.shape, "| params:", f"{count_params(m)/1e6:.2f}M")
