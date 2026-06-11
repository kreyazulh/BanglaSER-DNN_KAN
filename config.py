"""Central config. Edit the data_dir paths to point at wherever you unzipped
the datasets. Emotion lists and filename parsing differ per corpus, so each
dataset gets its own block here instead of hardcoding things all over.
"""

# NOTE: kaggle mirrors sometimes rename files / reorganize folders.
# If label parsing breaks, fix the parser in utils/data.py for your copy.

DATASETS = {
    "subesco": {
        "data_dir": "data/SUBESCO",  # <- change me
        "emotions": ["ANGRY", "DISGUST", "FEAR", "HAPPY", "NEUTRAL", "SAD", "SURPRISE"],
        # SUBESCO filenames look like: F_02_MONIKA_S_1_SURPRISE_5.wav
        # speaker id = gender + number (F_02), emotion is the 6th token
        "label_mode": "filename",
        "emotion_token": 5,
        "speaker_tokens": (0, 1),
        "si_split": {"n_train_spk": 16, "n_test_spk": 4, "n_val_spk": 2},
    },
    "banglaser": {
        "data_dir": "data/BanglaSER",  # <- change me
        "emotions": ["ANGRY", "HAPPY", "NEUTRAL", "SAD", "SURPRISE", "DISGUST"],
        # banglaser uses RAVDESS-style numeric codes in filenames,
        # e.g. 03-01-05-02-02-02-05.wav -> 3rd field is emotion code
        "label_mode": "code",
        "code_field": 2,
        "code_map": {"01": "NEUTRAL", "02": "HAPPY", "03": "SAD",
                     "04": "ANGRY", "05": "SURPRISE", "06": "DISGUST"},
        "speaker_field": 6,  # last field is actor id
        "si_split": {"n_train_spk": 25, "n_test_spk": 9, "n_val_spk": 3},
    },
    "ravdess": {
        "data_dir": "data/RAVDESS",
        "emotions": ["NEUTRAL", "CALM", "HAPPY", "SAD", "ANGRY", "FEAR", "DISGUST", "SURPRISE"],
        "label_mode": "code",
        "code_field": 2,
        "code_map": {"01": "NEUTRAL", "02": "CALM", "03": "HAPPY", "04": "SAD",
                     "05": "ANGRY", "06": "FEAR", "07": "DISGUST", "08": "SURPRISE"},
        "speaker_field": 6,
        "si_split": {"n_train_spk": 19, "n_test_spk": 5, "n_val_spk": 2},
    },
    "emodb": {
        "data_dir": "data/EmoDB",
        "emotions": ["ANGRY", "BORED", "DISGUST", "FEAR", "HAPPY", "SAD", "NEUTRAL"],
        # emodb: 03a01Fa.wav -> first 2 chars speaker, 6th char emotion letter
        "label_mode": "emodb",
        "letter_map": {"W": "ANGRY", "L": "BORED", "E": "DISGUST", "A": "FEAR",
                       "F": "HAPPY", "T": "SAD", "N": "NEUTRAL"},
        "si_split": {"n_train_spk": 8, "n_test_spk": 2, "n_val_spk": 1},
    },
    "emovo": {
        "data_dir": "data/EMOVO",
        "emotions": ["DISGUST", "FEAR", "ANGRY", "HAPPY", "SURPRISE", "SAD", "NEUTRAL"],
        # emovo: dis-f1-b1.wav -> emotion prefix, speaker is 2nd token
        "label_mode": "emovo",
        "prefix_map": {"dis": "DISGUST", "pau": "FEAR", "rab": "ANGRY",
                       "gio": "HAPPY", "sor": "SURPRISE", "tri": "SAD", "neu": "NEUTRAL"},
        "si_split": {"n_train_spk": 4, "n_test_spk": 2, "n_val_spk": 1},
    },
}

# training hyperparams, found via grid search (see paper sec 3.4)
TRAIN = {
    "lr": 5e-4,
    "weight_decay": 0.015,
    "dropout": 0.3,
    "batch_size": 16,
    "max_epochs": 200,
    "patience": 50,
    "label_smoothing": 0.1,
    "seed": 42,
}

# feature settings
FEAT = {
    "sr": 16000,
    "n_mfcc": 13,
    "spectral_dim": 80,     # 40 LLDs x (mean, std)
    "prosodic_dim": 51,     # 15 feats x 3 scales x 3 stats + 6 contour descriptors
    "scales_ms": [10, 25, 50],
    "eta2_threshold": 0.94,
    "selection_fraction": 0.10,  # only 10% of train data used for selection
}

# the 15 features that survived eta-squared selection on SUBESCO.
# select_features.py can recompute these from scratch on any corpus.
SELECTED_FEATURES = [
    "f0_mean", "f0_std", "f0_p25", "f0_p75", "f0_range",
    "f0_rising_ratio", "f0_falling_ratio",
    "flux_std", "rms_mean", "rms_std",
    "hnr", "voiced_ratio", "zcr_mean", "rolloff_mean", "mfcc1_mean",
]
