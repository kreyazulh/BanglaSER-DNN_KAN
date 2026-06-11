"""Split logic. This is the part of the paper we care most about getting
right: speaker-independent means NO speaker overlap between train/val/test.

SD splits are random stratified and only exist for comparison against prior
work, don't report them as your headline number.
"""

import random
from collections import defaultdict

import numpy as np


def speaker_independent_split(items, n_test_spk, n_val_spk=0, seed=42):
    """split by speaker. test speakers never appear in train or val."""
    rng = random.Random(seed)
    speakers = sorted(set(i["speaker"] for i in items))
    rng.shuffle(speakers)

    test_spk = set(speakers[:n_test_spk])
    val_spk = set(speakers[n_test_spk:n_test_spk + n_val_spk])
    train_spk = set(speakers[n_test_spk + n_val_spk:])

    train = [i for i in items if i["speaker"] in train_spk]
    val = [i for i in items if i["speaker"] in val_spk]
    test = [i for i in items if i["speaker"] in test_spk]

    # paranoia check, this bug has burned people before
    assert not (train_spk & test_spk) and not (val_spk & test_spk)

    print(f"SI split -> train {len(train)} ({len(train_spk)} spk) | "
          f"val {len(val)} ({len(val_spk)} spk) | test {len(test)} ({len(test_spk)} spk)")
    return train, val, test


def speaker_dependent_split(items, test_frac=0.2, val_frac=0.1, seed=42):
    """random stratified split, same speaker CAN appear on both sides.
    inflated numbers, reference only."""
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for i in items:
        by_class[i["label"]].append(i)

    train, val, test = [], [], []
    for label, group in by_class.items():
        group = group[:]
        rng.shuffle(group)
        n_test = int(len(group) * test_frac)
        n_val = int(len(group) * val_frac)
        test += group[:n_test]
        val += group[n_test:n_test + n_val]
        train += group[n_test + n_val:]

    print(f"SD split -> train {len(train)} | val {len(val)} | test {len(test)} (speaker overlap allowed!)")
    return train, val, test


def speaker_kfold(items, k=5, seed=42):
    """speaker-wise k-fold. yields (train, test) per fold.
    used for the 5-fold CV stability check in the paper (92.07 +/- 0.91)."""
    rng = random.Random(seed)
    speakers = sorted(set(i["speaker"] for i in items))
    rng.shuffle(speakers)
    folds = np.array_split(speakers, k)

    for f, fold_spk in enumerate(folds):
        fold_spk = set(fold_spk)
        test = [i for i in items if i["speaker"] in fold_spk]
        train = [i for i in items if i["speaker"] not in fold_spk]
        print(f"fold {f}: train {len(train)}, test {len(test)} ({sorted(fold_spk)})")
        yield train, test
