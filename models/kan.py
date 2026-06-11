"""Shared-basis KAN layer (eq 3).

The trick vs vanilla KAN: instead of a separate spline per (input, output)
pair we share one set of cubic b-spline basis functions across all inputs
and only learn control points per output neuron. drops first-layer spline
params from 51*128*3 = 19584 to 128*3 = 384.

inputs are expected in [-1, 1] (we minmax-normalize prosodic features
upstream). knots are uniform on that range, grid=5, order=3.
"""

import torch
import torch.nn as nn


def _cubic_bspline(u):
    # standard cubic b-spline kernel, support on [-2, 2]
    au = u.abs()
    out = torch.zeros_like(u)
    m1 = au < 1
    m2 = (au >= 1) & (au < 2)
    out[m1] = (4 - 6 * au[m1] ** 2 + 3 * au[m1] ** 3) / 6
    out[m2] = ((2 - au[m2]) ** 3) / 6
    return out


class SharedSplineKAN(nn.Module):
    def __init__(self, in_dim, out_dim, n_basis=3, grid=5, grid_range=(-1.0, 1.0)):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_basis = n_basis

        # the w_i . x residual part of eq 3, plain linear
        self.base = nn.Linear(in_dim, out_dim)

        # control points c_ij, unique per OUTPUT neuron, shared over inputs
        self.coef = nn.Parameter(torch.empty(out_dim, n_basis))
        nn.init.normal_(self.coef, std=0.1)

        # uniform knot centers over the grid range
        lo, hi = grid_range
        centers = torch.linspace(lo, hi, n_basis)
        width = (hi - lo) / max(grid - 1, 1)
        self.register_buffer("centers", centers)
        self.register_buffer("width", torch.tensor(width))

    def forward(self, x):
        # x: (B, in_dim), should already live in [-1,1] but clamp just in case
        x = x.clamp(-1.0, 1.0)

        # evaluate shared basis on every input element: (B, in_dim, n_basis)
        u = (x.unsqueeze(-1) - self.centers) / self.width
        B = _cubic_bspline(u)

        # share across inputs by averaging, then mix with per-output coefs
        pooled = B.mean(dim=1)              # (B, n_basis)
        spline = pooled @ self.coef.t()     # (B, out_dim)

        return self.base(x) + spline


class KANStack(nn.Module):
    """two shared-spline KAN layers with a norm in between, used per stream layer"""

    def __init__(self, in_dim, out_dim, dropout=0.3):
        super().__init__()
        self.kan = SharedSplineKAN(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.norm(self.kan(x)))
