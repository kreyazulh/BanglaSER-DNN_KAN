"""Evaluate a trained checkpoint: overall + per-class accuracy, confusion
matrix png, and optionally a McNemar significance test against another
model's predictions.

usage:
    python evaluate.py --dataset subesco --ckpt checkpoints/subesco_si.pt
    python evaluate.py --dataset subesco --ckpt checkpoints/subesco_si.pt \
        --compare other_preds.npy
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from models.network import DualStreamDNNKAN
from train import load_cache, Normalizer, set_seed
from utils.splits import speaker_independent_split


def mcnemar(correct_a, correct_b):
    """exact-ish mcnemar with continuity correction. inputs are boolean
    arrays of per-utterance correctness for two models on the SAME test set."""
    b = int(np.sum(correct_a & ~correct_b))   # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))   # a wrong, b right
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2 as chi2_dist
    p = float(chi2_dist.sf(chi2, df=1))
    return chi2, p


def confusion(y_true, y_pred, n):
    M = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", default="features_cache")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare", default=None, help=".npy of 0/1 correctness from another model")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = config.DATASETS[args.dataset]
    emotions = cfg["emotions"]
    n_classes = len(emotions)

    items = load_cache(args.dataset, args.cache_dir)
    s = cfg["si_split"]
    train, _, test = speaker_independent_split(items, s["n_test_spk"], s.get("n_val_spk", 0), args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = DualStreamDNNKAN(n_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # rebuild the normalizer from saved stats
    norm = Normalizer()
    for k, v in ckpt["normalizer"].items():
        setattr(norm, k, v)

    xs, xp, y = norm(test)
    with torch.no_grad():
        preds = model(xs, xp).argmax(1).numpy()
    y = y.numpy()

    acc = (preds == y).mean()
    print(f"\noverall SI accuracy: {acc*100:.2f}%\n")

    print(f"{'class':<12}{'n':>6}{'acc':>8}")
    for c in range(n_classes):
        mask = y == c
        if mask.sum() == 0:
            continue
        print(f"{emotions[c]:<12}{mask.sum():>6}{(preds[mask]==c).mean()*100:>7.1f}%")

    M = confusion(y, preds, n_classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(n_classes), emotions, rotation=45, ha="right")
    ax.set_yticks(range(n_classes), emotions)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, M[i, j], ha="center", va="center",
                    color="white" if M[i, j] > M.max() / 2 else "black", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{args.dataset} SI confusion matrix ({acc*100:.2f}%)")
    plt.tight_layout()
    out = f"confusion_{args.dataset}.png"
    plt.savefig(out, dpi=150)
    print(f"\nsaved {out}")

    # save our correctness vector so other runs can mcnemar against it
    np.save(f"correct_{args.dataset}.npy", (preds == y).astype(np.uint8))

    if args.compare:
        other = np.load(args.compare).astype(bool)
        assert len(other) == len(y), "compare file must be on the same test set / same seed"
        chi2, p = mcnemar(preds == y, other)
        print(f"\nMcNemar vs {args.compare}: chi2={chi2:.3f}, p={p:.2e}")
        print("significant at p<0.01" if p < 0.01 else "NOT significant at p<0.01")


if __name__ == "__main__":
    main()
