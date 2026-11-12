"""Baselines no profundos evaluados con los mismos folds por paciente.

    python -m src.baselines
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CACHE_DIR, OOF_DIR, RESULTS_DIR, TARGETS
from .evaluate import all_metrics


def soft_fit_predict(make_model, Xtr, Ytr, Xva):
    """Entrena con etiquetas suaves replicando cada muestra por clase con peso = probabilidad."""
    k = Ytr.shape[1]
    Xr = np.repeat(Xtr, k, axis=0)
    yr = np.tile(np.arange(k), len(Xtr))
    wr = Ytr.reshape(-1)
    keep = wr > 0
    m = make_model()
    if hasattr(m, "steps"):
        m.fit(Xr[keep], yr[keep], **{f"{m.steps[-1][0]}__sample_weight": wr[keep]})
    else:
        m.fit(Xr[keep], yr[keep], sample_weight=wr[keep])
    return m.predict_proba(Xva)


def main():
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    feats = pd.read_csv(CACHE_DIR / "band_features.csv")
    ok = meta.cache_ok.to_numpy()
    meta, feats = meta[ok].reset_index(drop=True), feats[ok].reset_index(drop=True)
    X = feats.to_numpy(np.float64)
    Y = meta[TARGETS].to_numpy()
    votes = meta.total_votes.to_numpy()

    models = {
        "prior_uniforme": None,
        "prior_clases": None,
        "logreg_bandas": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
        "hgb_bandas": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                              max_leaf_nodes=31, l2_regularization=1.0,
                                                              random_state=0),
    }
    rows = []
    for name, mk in models.items():
        P = np.zeros_like(Y)
        for f in sorted(meta.fold.unique()):
            tr, va = (meta.fold != f).to_numpy(), (meta.fold == f).to_numpy()
            if name == "prior_uniforme":
                P[va] = 1 / Y.shape[1]
            elif name == "prior_clases":
                P[va] = Y[tr].mean(0)
            else:
                P[va] = soft_fit_predict(mk, X[tr], Y[tr], X[va])
        m = all_metrics(Y, P, votes)
        print(name, {k: round(v, 4) for k, v in m.items()})
        df = meta[["eeg_id", "patient_id", "fold", "total_votes"] + TARGETS].copy()
        df[[f"pred_{t}" for t in TARGETS]] = P
        df.to_csv(OOF_DIR / f"{name}.csv", index=False)
        rows.append({"experiment": name, "model": name, "pretrained": False, "input": "features",
                     "folds": "all", "stage": 0, "epochs": 0, "n_valid": len(Y), "train_min": 0, **m})
    mpath = RESULTS_DIR / "metrics.csv"
    res = pd.DataFrame(rows)
    if mpath.exists():
        old = pd.read_csv(mpath)
        res = pd.concat([old[~old.experiment.isin(res.experiment)], res])
    res.to_csv(mpath, index=False)


if __name__ == "__main__":
    main()
