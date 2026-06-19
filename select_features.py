"""Eta-squared feature selection 
Runs one-way ANOVA per candidate feature against emotion labels, computes
eta^2 = SS_between / SS_total, keeps features above the threshold
"""

import argparse
import random

import numpy as np
import librosa
from scipy import stats
from tqdm import tqdm

import config
from utils.data import list_dataset
from utils.splits import speaker_independent_split
from utils.audio import candidate_pool


def eta_squared(values, labels):
    """eq 1. values: (N,) one feature across utterances, labels: (N,)"""
    grand = values.mean()
    ss_total = ((values - grand) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for c in np.unique(labels):
        grp = values[labels == c]
        ss_between += len(grp) * (grp.mean() - grand) ** 2
    return float(ss_between / ss_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="subesco")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--threshold", type=float, default=config.FEAT["eta2_threshold"])
    ap.add_argument("--fraction", type=float, default=config.FEAT["selection_fraction"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = config.DATASETS[args.dataset]
    items = list_dataset(args.dataset, args.data_dir)

    # select only from the SI train portion, test speakers must stay unseen
    train, _, _ = speaker_independent_split(
        items, cfg["si_split"]["n_test_spk"], cfg["si_split"].get("n_val_spk", 0), seed=args.seed
    )

    rng = random.Random(args.seed)
    n_sel = max(int(len(train) * args.fraction), 50)
    sample = rng.sample(train, min(n_sel, len(train)))
    print(f"running selection on {len(sample)} utterances ({args.fraction:.0%} of train)")

    names = None
    rows, labels = [], []
    for it in tqdm(sample):
        try:
            y, sr = librosa.load(it["path"], sr=config.FEAT["sr"])
            n, v = candidate_pool(y, sr)
            names = n
            rows.append(v)
            labels.append(it["label"])
        except Exception:
            continue

    X = np.stack(rows)
    y = np.array(labels)

    print(f"\n{'feature':<22}{'eta^2':>8}{'p-value':>14}")
    print("-" * 44)
    results = []
    for j, name in enumerate(names):
        e2 = eta_squared(X[:, j], y)
        groups = [X[y == c, j] for c in np.unique(y)]
        # one-way ANOVA F-test for the p-value
        try:
            _, p = stats.f_oneway(*groups)
        except Exception:
            p = 1.0
        results.append((name, e2, p))

    results.sort(key=lambda r: -r[1])
    selected = []
    for name, e2, p in results:
        mark = " *" if (e2 > args.threshold and p < 1e-15) else ""
        if mark:
            selected.append(name)
        print(f"{name:<22}{e2:>8.3f}{p:>14.2e}{mark}")

    print(f"\n{len(selected)} features pass eta^2 > {args.threshold} and p < 1e-15:")
    print(selected)
    print("\npaste these into SELECTED_FEATURES in config.py if you're adapting to a new corpus")


if __name__ == "__main__":
    main()
