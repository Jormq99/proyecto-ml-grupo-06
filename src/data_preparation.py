"""Carga de train.csv, consolidación por eeg_id y partición en folds por paciente."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import CLASSES, DATA_DIR, N_FOLDS, SEED, TARGETS


def load_train(path=None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_DIR / "train.csv")


def consolidate_by_eeg(df: pd.DataFrame) -> pd.DataFrame:
    """Una muestra por eeg_id.

    Las sub-ventanas (eeg_sub_id) de un mismo EEG se solapan mucho y casi siempre
    comparten las mismas etiquetas; usarlas todas infla el dataset con muestras
    casi duplicadas. Se toma la sub-ventana central como representativa (para
    recortar la señal) y como objetivo la media de las distribuciones de votos
    normalizadas de todas sus sub-ventanas.
    """
    df = df.copy()
    votes = df[TARGETS].to_numpy(dtype=np.float64)
    df["total_votes"] = votes.sum(1)
    df[TARGETS] = votes / df["total_votes"].to_numpy()[:, None]

    rows = []
    for eeg_id, g in df.groupby("eeg_id", sort=False):
        mid = g.iloc[len(g) // 2]
        rec = {
            "eeg_id": eeg_id,
            "spectrogram_id": mid.spectrogram_id,
            "patient_id": mid.patient_id,
            "eeg_offset": mid.eeg_label_offset_seconds,
            "spec_offset": mid.spectrogram_label_offset_seconds,
            "n_subwindows": len(g),
            "total_votes": g["total_votes"].mean(),
        }
        rec.update(g[TARGETS].mean().to_dict())
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["target"] = out[TARGETS].to_numpy().argmax(1)
    out["expert_consensus"] = np.array(CLASSES)[out["target"]]
    return out.reset_index(drop=True)


def add_folds(df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    """Folds agrupados por paciente (evita fuga entre train/valid) y estratificados por clase."""
    df = df.copy()
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for f, (_, va) in enumerate(sgkf.split(df, df["target"], df["patient_id"])):
        df.loc[va, "fold"] = f
    return df


def vote_quality(total_votes) -> np.ndarray:
    """Clasifica las muestras según cantidad de anotadores (las de <10 votos son más ruidosas)."""
    return np.where(np.asarray(total_votes) >= 10, "alta (>=10 votos)", "baja (<10 votos)")
