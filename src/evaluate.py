"""Métricas: KL-divergence (métrica oficial de la competencia) y métricas clínicas."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)

from .config import CLASSES

EPS = 1e-15


def kl_divergence(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """KL(y_true || y_pred) promedio por fila, como en el score de Kaggle."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), EPS, 1 - EPS)
    y_pred = y_pred / y_pred.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0)
    return float(terms.sum(1).mean())


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray, total_votes=None) -> dict:
    yt = y_true.argmax(1)
    yp = y_pred.argmax(1)
    out = {
        "kl": kl_divergence(y_true, y_pred),
        "accuracy": accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, average="macro"),
        "macro_auc": roc_auc_score(np.eye(len(CLASSES))[yt], y_pred, average="macro"),
    }
    if total_votes is not None:
        hq = np.asarray(total_votes) >= 10
        out["kl_hq"] = kl_divergence(y_true[hq], y_pred[hq])
        out["accuracy_hq"] = accuracy_score(yt[hq], yp[hq])
    return out


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Sensibilidad, especificidad, F1 y AUC por clase (uno-contra-resto)."""
    yt, yp = y_true.argmax(1), y_pred.argmax(1)
    cm = confusion_matrix(yt, yp, labels=range(len(CLASSES)))
    rows = []
    for k, c in enumerate(CLASSES):
        tp = cm[k, k]; fn = cm[k].sum() - tp; fp = cm[:, k].sum() - tp
        tn = cm.sum() - tp - fn - fp
        rows.append({
            "clase": c, "soporte": int(cm[k].sum()),
            "sensibilidad": tp / max(tp + fn, 1),
            "especificidad": tn / max(tn + fp, 1),
            "precision": tp / max(tp + fp, 1),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "auc": roc_auc_score(yt == k, y_pred[:, k]),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Métricas robustas al desbalance de clases
# ----------------------------------------------------------------------------
def kl_per_row(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), EPS, 1 - EPS)
    y_pred = y_pred / y_pred.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(y_true > 0, y_true * np.log(y_true / y_pred), 0.0).sum(1)


def balanced_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Métricas en las que cada clase pesa igual, sin importar su frecuencia.

    - balanced_accuracy: promedio de la sensibilidad (recall) de cada clase.
    - macro_kl: KL media calculada por separado para cada clase de consenso y luego promediada.
    - kappa / mcc: acuerdo corregido por azar (0 = no mejor que el azar dado el desbalance).
    - min_recall: la sensibilidad de la peor clase.
    - share_pred_other / share_true_other: fracción de muestras predichas / reales como Other.
    """
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef, recall_score
    yt, yp = y_true.argmax(1), y_pred.argmax(1)
    k = len(CLASSES)
    klr = kl_per_row(y_true, y_pred)
    per_class_kl = [klr[yt == c].mean() for c in range(k) if (yt == c).any()]
    rec = recall_score(yt, yp, labels=range(k), average=None, zero_division=0)
    present = np.bincount(yt, minlength=k) > 0
    other = CLASSES.index("Other")
    return {
        "kl": klr.mean(),
        "macro_kl": float(np.mean(per_class_kl)),
        "accuracy": accuracy_score(yt, yp),
        "balanced_accuracy": balanced_accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, average="macro", labels=range(k), zero_division=0),
        "kappa": cohen_kappa_score(yt, yp),
        "mcc": matthews_corrcoef(yt, yp),
        "macro_auc": roc_auc_score(np.eye(k)[yt][:, present], y_pred[:, present], average="macro"),
        "min_recall": rec[present].min(),
        "share_pred_other": (yp == other).mean(),
        "share_true_other": (yt == other).mean(),
    }
