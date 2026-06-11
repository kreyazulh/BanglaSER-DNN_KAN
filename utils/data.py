"""Dataset file listing + label/speaker parsing.

Every corpus encodes labels differently (token in filename, numeric code,
single letter...) so each gets a small parser. If your downloaded copy has
different naming, fix it here, everything downstream just sees
(path, label_idx, speaker_id) tuples.
"""

import os
from pathlib import Path

import config


def _parse_subesco(fname, cfg):
    toks = Path(fname).stem.split("_")
    emo = toks[cfg["emotion_token"]].upper()
    spk = "_".join(toks[i] for i in cfg["speaker_tokens"])
    return emo, spk


def _parse_code(fname, cfg):
    toks = Path(fname).stem.split("-")
    emo = cfg["code_map"].get(toks[cfg["code_field"]])
    spk = toks[cfg["speaker_field"]]
    return emo, spk


def _parse_emodb(fname, cfg):
    stem = Path(fname).stem
    spk = stem[:2]
    emo = cfg["letter_map"].get(stem[5].upper())
    return emo, spk


def _parse_emovo(fname, cfg):
    toks = Path(fname).stem.split("-")
    emo = cfg["prefix_map"].get(toks[0].lower())
    spk = toks[1]
    return emo, spk


PARSERS = {
    "filename": _parse_subesco,
    "code": _parse_code,
    "emodb": _parse_emodb,
    "emovo": _parse_emovo,
}


def list_dataset(name, data_dir=None):
    """walk the dataset dir, return list of dicts with path/label/speaker.

    skips files it can't parse instead of crashing, but prints a count
    so you notice when something is off.
    """
    cfg = config.DATASETS[name]
    root = data_dir or cfg["data_dir"]
    emotions = cfg["emotions"]
    parser = PARSERS[cfg["label_mode"]]

    items, skipped = [], 0
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if not f.lower().endswith((".wav", ".flac", ".mp3")):
                continue
            try:
                emo, spk = parser(f, cfg)
            except (IndexError, KeyError):
                skipped += 1
                continue
            if emo is None or emo not in emotions:
                skipped += 1
                continue
            items.append({
                "path": os.path.join(dirpath, f),
                "label": emotions.index(emo),
                "emotion": emo,
                "speaker": spk,
            })

    if skipped:
        print(f"[{name}] skipped {skipped} files that didn't parse, double check naming if this looks high")
    if not items:
        raise RuntimeError(f"no audio found under {root}, did you set the path in config.py?")

    print(f"[{name}] {len(items)} utterances, {len(set(i['speaker'] for i in items))} speakers, "
          f"{len(emotions)} classes")
    return items
