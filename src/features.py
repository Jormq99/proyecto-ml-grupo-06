"""Transformación de EEG y espectrogramas en imágenes para CNNs, y features clásicas.

Por cada muestra se generan 8 "tiles" de TILE_H x TILE_W:
  * 4 espectrogramas de Kaggle (10 min, regiones LL/LP/RP/RL)
  * 4 espectrogramas calculados a partir del EEG crudo (50 s, cadenas LL/LP/RP/RR)
"""
import numpy as np
import pandas as pd
from scipy import signal

from .config import (CHAINS, DATA_DIR, EEG_FS, EEG_WINDOW_S, KAGGLE_REGIONS,
                     SPEC_WINDOW_S, TILE_H, TILE_W)

BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 20)}

# Filtro pasa-banda 0.5-40 Hz para el EEG crudo
_SOS = signal.butter(4, [0.5, 40], btype="bandpass", fs=EEG_FS, output="sos")


def _resize(img: np.ndarray, h: int = TILE_H, w: int = TILE_W) -> np.ndarray:
    """Re-muestreo bilineal simple (sin dependencias externas)."""
    H, W = img.shape
    yi = np.linspace(0, H - 1, h)
    xi = np.linspace(0, W - 1, w)
    y0 = np.floor(yi).astype(int); y1 = np.minimum(y0 + 1, H - 1); wy = (yi - y0)[:, None]
    x0 = np.floor(xi).astype(int); x1 = np.minimum(x0 + 1, W - 1); wx = (xi - x0)[None, :]
    a = img[y0][:, x0]; b = img[y0][:, x1]; c = img[y1][:, x0]; d = img[y1][:, x1]
    return (a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + c * (1 - wx) * wy + d * wx * wy)


def _standardize(x: np.ndarray) -> np.ndarray:
    m, s = np.nanmean(x), np.nanstd(x)
    x = (x - m) / (s + 1e-6)
    return np.nan_to_num(x, nan=0.0).astype(np.float32)


# ----------------------------------------------------------------------------
# Espectrogramas de Kaggle
# ----------------------------------------------------------------------------
def read_kaggle_spectrogram(spec_id) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "train_spectrograms" / f"{spec_id}.parquet")


def kaggle_spec_window(spec: pd.DataFrame, offset: float) -> pd.DataFrame:
    t = spec["time"].to_numpy()
    m = (t >= offset) & (t < offset + SPEC_WINDOW_S)
    return spec.loc[m]


def kaggle_spec_tiles(spec_win: pd.DataFrame) -> np.ndarray:
    """(4, TILE_H, TILE_W): log-potencia, frecuencia baja abajo, normalizada por tile."""
    tiles = []
    for reg in KAGGLE_REGIONS:
        cols = [c for c in spec_win.columns if c.startswith(reg + "_")]
        x = spec_win[cols].to_numpy(dtype=np.float32).T          # (100 freq, ~300 t)
        x = np.log(np.clip(np.nan_to_num(x, nan=np.exp(-4)), np.exp(-4), np.exp(8)))
        x = _resize(x[::-1])                                    # alta frecuencia arriba
        tiles.append(_standardize(x))
    return np.stack(tiles)


# ----------------------------------------------------------------------------
# EEG crudo -> espectrogramas por cadena bipolar
# ----------------------------------------------------------------------------
def read_eeg(eeg_id) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "train_eegs" / f"{eeg_id}.parquet")


def eeg_window(eeg: pd.DataFrame, offset: float) -> pd.DataFrame:
    start = int(offset * EEG_FS)
    return eeg.iloc[start:start + EEG_WINDOW_S * EEG_FS]


def clean_eeg(win: pd.DataFrame) -> pd.DataFrame:
    """Imputa NaN con la media del canal (o 0), filtra 0.5-40 Hz y recorta amplitudes extremas."""
    x = win.copy()
    x = x.fillna(x.mean()).fillna(0.0)
    arr = signal.sosfiltfilt(_SOS, x.to_numpy(dtype=np.float64), axis=0)
    arr = np.clip(arr, -1024, 1024)
    return pd.DataFrame(arr, columns=x.columns)


def bipolar_montage(win: pd.DataFrame) -> dict:
    return {ch: np.stack([win[a].to_numpy() - win[b].to_numpy() for a, b in pairs])
            for ch, pairs in CHAINS.items()}


def eeg_spec_tiles(win: pd.DataFrame) -> np.ndarray:
    """(4, TILE_H, TILE_W): STFT de cada derivación (0.5-20 Hz), promedio por cadena."""
    win = clean_eeg(win)
    tiles = []
    for ch, sig in bipolar_montage(win).items():
        f, t, S = signal.spectrogram(sig, fs=EEG_FS, nperseg=256, noverlap=256 - 40,
                                     nfft=1024, axis=-1)
        keep = (f >= 0.5) & (f <= 20)
        S = np.log(S[:, keep, :].mean(0) + 1e-6)              # (freq, time)
        tiles.append(_standardize(_resize(S[::-1])))
    return np.stack(tiles)


# ----------------------------------------------------------------------------
# Muestra completa
# ----------------------------------------------------------------------------
def build_sample(row) -> np.ndarray:
    """(8, TILE_H, TILE_W) float16 para una fila del dataframe consolidado."""
    spec = kaggle_spec_window(read_kaggle_spectrogram(row.spectrogram_id), row.spec_offset)
    eeg = eeg_window(read_eeg(row.eeg_id), row.eeg_offset)
    return np.concatenate([kaggle_spec_tiles(spec), eeg_spec_tiles(eeg)]).astype(np.float16)


def band_power_features(row) -> dict:
    """Features clásicas para el baseline no profundo: potencia relativa por banda,
    región y estadísticas de amplitud del EEG."""
    feats = {}
    spec = kaggle_spec_window(read_kaggle_spectrogram(row.spectrogram_id), row.spec_offset)
    for reg in KAGGLE_REGIONS:
        cols = [c for c in spec.columns if c.startswith(reg + "_")]
        freqs = np.array([float(c.split("_")[1]) for c in cols])
        x = np.nan_to_num(spec[cols].to_numpy(dtype=np.float64))
        tot = x.sum(1) + 1e-9
        for b, (lo, hi) in BANDS.items():
            rel = x[:, (freqs >= lo) & (freqs < hi)].sum(1) / tot
            feats[f"k_{reg}_{b}_mean"] = rel.mean()
            feats[f"k_{reg}_{b}_std"] = rel.std()
        feats[f"k_{reg}_logpow"] = np.log(tot.mean())
    eeg = clean_eeg(eeg_window(read_eeg(row.eeg_id), row.eeg_offset))
    for ch, sig in bipolar_montage(eeg).items():
        f, P = signal.welch(sig, fs=EEG_FS, nperseg=512, axis=-1)
        P = P.mean(0)
        tot = P[(f >= 0.5) & (f < 20)].sum() + 1e-9
        for b, (lo, hi) in BANDS.items():
            feats[f"e_{ch}_{b}_rel"] = P[(f >= lo) & (f < hi)].sum() / tot
        feats[f"e_{ch}_logpow"] = np.log(tot)
        feats[f"e_{ch}_std"] = sig.std()
        feats[f"e_{ch}_linelen"] = np.abs(np.diff(sig, axis=-1)).mean()
    feats["e_lr_asym"] = (feats["e_LL_logpow"] + feats["e_LP_logpow"]
                          - feats["e_RR_logpow"] - feats["e_RP_logpow"])
    return feats
