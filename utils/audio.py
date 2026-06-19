"""Low level feature computation. librosa does most of the work here.
The paper used OpenSMILE for the candidate pool, this is a librosa port that
gets very close. HNR is approximated with an HPSS energy ratio, if you want
the exact praat-style HNR install parselmouth and swap it in (left a hook).
"""

import numpy as np
import librosa

import config


def _frame_params(sr, win_ms):
    n = int(sr * win_ms / 1000)
    return n, n // 2  # 50% hop


def _safe(x, fallback=0.0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return fallback
    return x


def _hnr_proxy(y):
    # cheap harmonic-to-noise estimate via hpss. good enough for ranking,
    # not identical to praat. TODO maybe switch to parselmouth someday
    h, p = librosa.effects.hpss(y)
    eh = np.sum(h ** 2) + 1e-10
    ep = np.sum(p ** 2) + 1e-10
    return float(10.0 * np.log10(eh / ep))


def base_tracks(y, sr, win_ms):
    """frame-level tracks at one window scale. returns dict of 1d arrays"""
    n_fft, hop = _frame_params(sr, win_ms)
    n_fft = max(n_fft, 64)  # pyin gets cranky with tiny windows

    f0, voiced, _ = librosa.pyin(
        y, fmin=65, fmax=500, sr=sr, frame_length=max(n_fft, 1024), hop_length=hop
    )

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    rms = librosa.feature.rms(S=S, frame_length=n_fft)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    mfcc1 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2, n_fft=max(n_fft, 256), hop_length=hop)[1]

    return {
        "f0": f0, "voiced": voiced, "rms": rms, "zcr": zcr,
        "rolloff": rolloff, "flux": flux, "mfcc1": mfcc1,
    }


def feature_values(tracks, y):
    """compute the 15 selected features (scalars) from frame tracks"""
    f0 = _safe(tracks["f0"])
    if f0.size < 2:
        f0 = np.array([0.0, 0.0])
    df0 = np.diff(f0)

    voiced = tracks["voiced"]
    voiced_ratio = float(np.nanmean(voiced)) if voiced is not None else 0.0

    vals = {
        "f0_mean": float(np.mean(f0)),
        "f0_std": float(np.std(f0)),
        "f0_p25": float(np.percentile(f0, 25)),
        "f0_p75": float(np.percentile(f0, 75)),
        "f0_range": float(np.max(f0) - np.min(f0)),
        "f0_rising_ratio": float(np.mean(df0 > 0)) if df0.size else 0.0,
        "f0_falling_ratio": float(np.mean(df0 < 0)) if df0.size else 0.0,
        "flux_std": float(np.std(_safe(tracks["flux"]))),
        "rms_mean": float(np.mean(_safe(tracks["rms"]))),
        "rms_std": float(np.std(_safe(tracks["rms"]))),
        "hnr": _hnr_proxy(y),
        "voiced_ratio": voiced_ratio,
        "zcr_mean": float(np.mean(_safe(tracks["zcr"]))),
        "rolloff_mean": float(np.mean(_safe(tracks["rolloff"]))),
        "mfcc1_mean": float(np.mean(_safe(tracks["mfcc1"]))),
    }
    return vals


def prosodic_vector(y, sr):
    """the 51-d prosodic descriptor from the paper.

    15 features x 3 scales x 3 stats (value, windowed std, windowed delta) = 45
    plus 6 global f0 contour descriptors = 51
    """
    parts = []
    for win_ms in config.FEAT["scales_ms"]:
        tracks = base_tracks(y, sr, win_ms)
        vals = feature_values(tracks, y)

        for name in config.SELECTED_FEATURES:
            v = vals[name]
            # per-scale stats: the value itself, plus dispersion + delta of the
            # underlying track when one exists, else zeros (scalar features)
            track_key = name.split("_")[0]
            if track_key in tracks and tracks[track_key] is not None:
                t = _safe(tracks[track_key])
                std = float(np.std(t)) if t.size else 0.0
                delta = float(np.mean(np.abs(np.diff(t)))) if t.size > 1 else 0.0
            else:
                std, delta = 0.0, 0.0
            parts.extend([v, std, delta])

    # 6 global contour descriptors over the full f0 track (25ms scale)
    tracks = base_tracks(y, sr, 25)
    f0 = _safe(tracks["f0"])
    if f0.size < 2:
        f0 = np.array([0.0, 0.0])
    df0 = np.diff(f0)
    slope = float(np.polyfit(np.arange(f0.size), f0, 1)[0]) if f0.size > 2 else 0.0
    contour = [
        float(np.mean(df0 > 0)),
        float(np.mean(df0 < 0)),
        slope,
        float(np.max(f0) - np.min(f0)),
        float(np.percentile(f0, 25)),
        float(np.percentile(f0, 75)),
    ]

    # parts holds (value, std, delta) triples per feature per scale.
    # collapse the triple into one number per feature per scale -> 45,
    # then append the 6 contour descriptors -> 51 total
    triples = np.array(parts, dtype=np.float32).reshape(3, 15, 3)
    per_scale = triples.mean(axis=2).reshape(-1)  # (45,)
    out = np.concatenate([per_scale, np.array(contour, dtype=np.float32)])
    assert out.shape[0] == 51
    return out


def spectral_vector(y, sr):
    """80-d spectral side: 40 LLDs (13 mfcc + 13 d + 13 dd + log energy)
    aggregated with mean and std -> 80"""
    n_mfcc = config.FEAT["n_mfcc"]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    energy = librosa.feature.rms(y=y)
    loge = np.log(energy + 1e-10)

    lld = np.vstack([mfcc, d1, d2, loge])  # (40, T)
    feat = np.concatenate([lld.mean(axis=1), lld.std(axis=1)])
    return feat.astype(np.float32)  # (80,)


def candidate_pool(y, sr):
    """the full 80-candidate pool used by eta-squared selection.
    returns (names, values). this is what select_features.py iterates over."""
    names, vals = [], []

    tracks = base_tracks(y, sr, 25)
    fv = feature_values(tracks, y)
    for k, v in fv.items():
        names.append(k)
        vals.append(v)

    # pad the pool with mfcc / spectral stats up to 80 candidates
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    for i in range(13):
        names += [f"mfcc{i}_mean", f"mfcc{i}_std"]
        vals += [float(mfcc[i].mean()), float(mfcc[i].std())]
    for i in range(13):
        names += [f"dmfcc{i}_mean"]
        vals += [float(d1[i].mean())]
    for i in range(13):
        names += [f"ddmfcc{i}_mean"]
        vals += [float(d2[i].mean())]

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    loge = np.log(librosa.feature.rms(y=y)[0] + 1e-10)
    names += ["centroid_mean", "centroid_std", "bandwidth_mean", "bandwidth_std",
              "rolloff_std", "zcr_std", "flux_mean", "duration",
              "f0_median", "f0_iqr", "rms_max", "rms_min", "loge_mean"]
    f0 = _safe(tracks["f0"], 0.0)
    f0 = f0 if np.ndim(f0) else np.array([f0])
    rms = _safe(tracks["rms"])
    vals += [float(cent.mean()), float(cent.std()), float(bw.mean()), float(bw.std()),
             float(np.std(_safe(tracks["rolloff"]))), float(np.std(_safe(tracks["zcr"]))),
             float(np.mean(_safe(tracks["flux"]))), float(len(y) / sr),
             float(np.median(f0)), float(np.percentile(f0, 75) - np.percentile(f0, 25)),
             float(rms.max()) if rms.size else 0.0, float(rms.min()) if rms.size else 0.0,
             float(loge.mean())]

    assert len(names) == 80, f"pool drifted to {len(names)}, fix the padding above"
    return names, np.array(vals, dtype=np.float32)
