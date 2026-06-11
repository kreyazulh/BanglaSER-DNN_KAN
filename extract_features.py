"""Extract spectral (80d) + prosodic (51d) features for a whole dataset and
cache them as one .npz. pyin is the slow part, expect ~1-2s per clip on cpu.
"""

import argparse
import os

import numpy as np
import librosa
from tqdm import tqdm

import config
from utils.data import list_dataset
from utils.audio import spectral_vector, prosodic_vector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(config.DATASETS.keys()))
    ap.add_argument("--data_dir", default=None, help="override path from config.py")
    ap.add_argument("--out_dir", default="features_cache")
    args = ap.parse_args()

    items = list_dataset(args.dataset, args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.dataset}.npz")

    if os.path.exists(out_path):
        print(f"{out_path} already exists, delete it if you want to re-extract")
        return

    spec, pros, labels, speakers, paths = [], [], [], [], []
    failed = 0
    for it in tqdm(items, desc=f"extracting {args.dataset}"):
        try:
            y, sr = librosa.load(it["path"], sr=config.FEAT["sr"])
            if len(y) < sr // 10:  # skip clips under 100ms, usually corrupt
                failed += 1
                continue
            spec.append(spectral_vector(y, sr))
            pros.append(prosodic_vector(y, sr))
            labels.append(it["label"])
            speakers.append(it["speaker"])
            paths.append(it["path"])
        except Exception as e:
            # some kaggle mirrors have broken wavs, just skip and count
            failed += 1

    if failed:
        print(f"warning: {failed} files failed to load/extract")

    np.savez(
        out_path,
        spectral=np.stack(spec),
        prosodic=np.stack(pros),
        labels=np.array(labels),
        speakers=np.array(speakers),
        paths=np.array(paths),
    )
    print(f"saved {len(labels)} utterances -> {out_path}")
    print(f"spectral {np.stack(spec).shape}, prosodic {np.stack(pros).shape}")


if __name__ == "__main__":
    main()
