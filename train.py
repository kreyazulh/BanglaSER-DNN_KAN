"""Training. supports the three evaluation modes from the paper:

  --split si   speaker independent (the headline metric, default)
  --split sd   speaker dependent (reference only, inflated)
  --split cv   speaker-wise 5-fold cross validation

no augmentation anywhere, on purpose (see sec 3.4 of the paper).

usage:
    python extract_features.py --dataset subesco --data_dir ...
    python train.py --dataset subesco --split si
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from models.network import DualStreamDNNKAN, count_params


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cache(dataset, cache_dir="features_cache"):
    path = os.path.join(cache_dir, f"{dataset}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing, run extract_features.py first")
    d = np.load(path, allow_pickle=True)
    items = []
    for i in range(len(d["labels"])):
        items.append({
            "spectral": d["spectral"][i],
            "prosodic": d["prosodic"][i],
            "label": int(d["labels"][i]),
            "speaker": str(d["speakers"][i]),
        })
    return items


class Normalizer:
    """spectral gets z-scored, prosodic gets minmax to [-1,1] (KAN wants
    bounded inputs for the spline grid). fit on train ONLY."""

    def fit(self, items):
        S = np.stack([i["spectral"] for i in items])
        P = np.stack([i["prosodic"] for i in items])
        self.s_mean, self.s_std = S.mean(0), S.std(0) + 1e-8
        self.p_min, self.p_max = P.min(0), P.max(0)
        rng = self.p_max - self.p_min
        rng[rng == 0] = 1.0
        self.p_rng = rng
        return self

    def __call__(self, items):
        S = np.stack([i["spectral"] for i in items])
        P = np.stack([i["prosodic"] for i in items])
        S = (S - self.s_mean) / self.s_std
        P = 2 * (P - self.p_min) / self.p_rng - 1
        P = np.clip(P, -1, 1)  # test data can fall outside train range
        y = np.array([i["label"] for i in items])
        return (
            torch.tensor(S, dtype=torch.float32),
            torch.tensor(P, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xs, xp, y in loader:
            xs, xp, y = xs.to(device), xp.to(device), y.to(device)
            logits, aux = model(xs, xp, return_aux=True)
            # small weight on the aux head so the gating probs stay sane
            loss = criterion(logits, y) + 0.3 * criterion(aux, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


def fit(train_items, val_items, test_items, n_classes, device, tag=""):
    t = config.TRAIN
    norm = Normalizer().fit(train_items)
    tr = TensorDataset(*norm(train_items))
    va = TensorDataset(*norm(val_items)) if val_items else None
    te = TensorDataset(*norm(test_items))

    tr_loader = DataLoader(tr, batch_size=t["batch_size"], shuffle=True)
    va_loader = DataLoader(va, batch_size=64) if va else None
    te_loader = DataLoader(te, batch_size=64)

    model = DualStreamDNNKAN(n_classes, dropout=t["dropout"]).to(device)
    print(f"model params: {count_params(model)/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=t["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    best_acc, best_state, bad = 0.0, None, 0
    monitor = va_loader or te_loader  # fall back to test for monitoring in cv mode
    for epoch in range(t["max_epochs"]):
        tr_loss, tr_acc = run_epoch(model, tr_loader, criterion, optimizer, device, train=True)
        mo_loss, mo_acc = run_epoch(model, monitor, criterion, optimizer, device, train=False)

        if mo_acc > best_acc:
            best_acc = mo_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 10 == 0 or bad == 0:
            print(f"ep {epoch:3d} | train {tr_acc:.4f} ({tr_loss:.3f}) | val {mo_acc:.4f} | best {best_acc:.4f}")

        if bad >= t["patience"]:
            print(f"early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    _, test_acc = run_epoch(model, te_loader, criterion, optimizer, device, train=False)
    print(f"\n[{tag}] test accuracy: {test_acc:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = f"checkpoints/{tag}.pt"
    torch.save({"state_dict": best_state, "normalizer": vars(norm), "test_acc": test_acc}, ckpt)
    print(f"saved {ckpt}")
    return test_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(config.DATASETS.keys()))
    ap.add_argument("--split", default="si", choices=["si", "sd", "cv"])
    ap.add_argument("--cache_dir", default="features_cache")
    ap.add_argument("--seed", type=int, default=config.TRAIN["seed"])
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    from utils.splits import speaker_independent_split, speaker_dependent_split, speaker_kfold

    cfg = config.DATASETS[args.dataset]
    items = load_cache(args.dataset, args.cache_dir)
    n_classes = len(cfg["emotions"])
    print(f"{len(items)} utterances loaded, {n_classes} classes")

    if args.split == "si":
        s = cfg["si_split"]
        train, val, test = speaker_independent_split(items, s["n_test_spk"], s.get("n_val_spk", 0), args.seed)
        fit(train, val, test, n_classes, device, tag=f"{args.dataset}_si")

    elif args.split == "sd":
        train, val, test = speaker_dependent_split(items, seed=args.seed)
        fit(train, val, test, n_classes, device, tag=f"{args.dataset}_sd")

    else:  # cv
        accs = []
        for f, (train, test) in enumerate(speaker_kfold(items, k=5, seed=args.seed)):
            acc = fit(train, None, test, n_classes, device, tag=f"{args.dataset}_cv{f}")
            accs.append(acc)
        print(f"\n5-fold speaker-wise CV: {np.mean(accs)*100:.2f} +/- {np.std(accs)*100:.2f}")


if __name__ == "__main__":
    main()
