"""Auditoría de sesgo por desbalance de clases.

Compara cada modelo con referencias triviales (predecir siempre la clase mayoritaria o la distribución
media de clases) usando métricas en las que todas las clases pesan igual.

    python -m src.imbalance_audit
Genera results/metrics_balance.csv y results/metrics_balance_por_clase.csv
"""
import numpy as np
import pandas as pd

from .config import CLASSES, OOF_DIR, RESULTS_DIR, TARGETS
from .evaluate import balanced_metrics, kl_per_row

PRED = [f"pred_{t}" for t in TARGETS]

MODELS = [
    ("Gradient boosting (bandas)", "hgb_bandas", "todos"),
    ("EfficientNet-B0 (fase 1)", "cv_effb0", "todos"),
    ("ConvNeXt-T (fase 1)", "cv_convnext_t", "todos"),
    ("Ensamble 50/50 (fase 1)", "ensemble", "todos"),
    ("EfficientNet-B0 sin optimizar", "hq_effb0_default", ">=10 votos"),
    ("EfficientNet-B0 optimizado", "hq_effb0", ">=10 votos"),
    ("ConvNeXt-T optimizado", "hq_convnext_t", ">=10 votos"),
    ("Ensamble optimizado", "hq_ensemble", ">=10 votos"),
    ("Ensamble optimizado + BBSE", "hq_ensemble:bbse", ">=10 votos"),
]


def subsets(df):
    hq = (df.total_votes >= 10).to_numpy()
    return {"todas": np.ones(len(df), bool), "alta calidad (>=10)": hq, "baja calidad (<10)": ~hq}


def main():
    ref = pd.read_csv(OOF_DIR / "ensemble.csv").sort_values("eeg_id").reset_index(drop=True)
    Y = ref[TARGETS].to_numpy()
    rows, per_class = [], []
    for sub, m in subsets(ref).items():
        k = Y.shape[1]
        other = np.zeros((m.sum(), k)); other[:, CLASSES.index("Other")] = 1
        # "siempre Other" con un poco de masa en el resto para que la KL sea finita
        other = other * 0.95 + 0.05 / k
        for name, P in [("Trivial: siempre Other", other),
                        ("Trivial: distribución media", np.tile(Y[m].mean(0), (m.sum(), 1)))]:
            rows.append({"modelo": name, "entrenado_con": "-", "subconjunto": sub, **balanced_metrics(Y[m], P)})

    for name, f, trained in MODELS:
        file, _, variant = f.partition(":")
        path = OOF_DIR / f"{file}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path).set_index("eeg_id").loc[ref.eeg_id].reset_index()
        cols = [f"{variant}_{t}" for t in TARGETS] if variant else PRED
        P = d[cols].to_numpy()
        for sub, m in subsets(ref).items():
            rows.append({"modelo": name, "entrenado_con": trained, "subconjunto": sub, **balanced_metrics(Y[m], P[m])})
            yt, yp = Y[m].argmax(1), P[m].argmax(1)
            klr = kl_per_row(Y[m], P[m])
            for c, cname in enumerate(CLASSES):
                sel = yt == c
                per_class.append({"modelo": name, "subconjunto": sub, "clase": cname, "n": int(sel.sum()),
                                  "recall": (yp[sel] == c).mean() if sel.any() else np.nan,
                                  "precision": (yt[yp == c] == c).mean() if (yp == c).any() else np.nan,
                                  "kl_media": klr[sel].mean() if sel.any() else np.nan,
                                  "share_real": sel.mean(), "share_predicha": (yp == c).mean()})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "metrics_balance.csv", index=False)
    pd.DataFrame(per_class).to_csv(RESULTS_DIR / "metrics_balance_por_clase.csv", index=False)
    cols = ["modelo", "subconjunto", "kl", "macro_kl", "accuracy", "balanced_accuracy", "macro_f1",
            "kappa", "macro_auc", "min_recall", "share_pred_other", "share_true_other"]
    with pd.option_context("display.width", 250):
        print(res[cols].round(3).to_string())


if __name__ == "__main__":
    main()
