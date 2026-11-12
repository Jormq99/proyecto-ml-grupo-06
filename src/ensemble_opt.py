"""Optimización bayesiana del ensamble EfficientNet-B0 + ConvNeXt-T (entrenados solo con >=10 votos)
y análisis del test de baja calidad (<10 votos).

Parámetros del ensamble (Optuna/TPE):
Objetivo: KL macro (KL promediada por clase de consenso), para no premiar el sesgo hacia la clase mayoritaria.

    w     peso de EfficientNet-B0 (1-w para ConvNeXt-T)
    modo  'arit' (promedio de probabilidades) o 'geom' (promedio de log-probabilidades)
    T     temperatura aplicada a las log-probabilidades combinadas (calibración)

Evaluación anidada (honesta): los parámetros aplicados al fold f se optimizan SOLO con las predicciones
de validación de alta calidad de los otros 4 folds. Las muestras de baja calidad nunca se usan para optimizar.

Además, sobre el test de baja calidad:
    * ajuste de priors por EM (Saerens et al., 2002): no supervisado, usa solo las predicciones del test;
    * ajuste "oráculo" con la distribución real de clases del test (cota superior, solo como referencia).

    python -m src.ensemble_opt
"""
import json

import numpy as np
import optuna
import pandas as pd

from .config import OOF_DIR, RESULTS_DIR, SEED, TARGETS
from .evaluate import all_metrics, balanced_metrics, kl_divergence
from .hq import macro_kl

MEMBERS = ["hq_effb0", "hq_convnext_t"]
PRED = [f"pred_{t}" for t in TARGETS]
EPS = 1e-7


def combine(P1, P2, w, mode, T):
    if mode == "arit":
        logp = np.log(np.clip(w * P1 + (1 - w) * P2, EPS, 1))
    else:
        logp = w * np.log(np.clip(P1, EPS, 1)) + (1 - w) * np.log(np.clip(P2, EPS, 1))
    z = logp / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def optimize(P1, P2, Y, n_trials=80):
    def obj(trial):
        w = trial.suggest_float("w", 0.0, 1.0)
        mode = trial.suggest_categorical("modo", ["arit", "geom"])
        T = trial.suggest_float("T", 0.5, 2.0, log=True)
        return macro_kl(Y, combine(P1, P2, w, mode, T))   # objetivo balanceado por clase
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.enqueue_trial({"w": 0.5, "modo": "arit", "T": 1.0})
    st.optimize(obj, n_trials=n_trials)
    return st.best_params, st.best_value, st


def em_prior_shift(P, prior_train, iters=100, tol=1e-6):
    """Re-estima la distribución de clases del test sin etiquetas y ajusta las probabilidades."""
    prior = prior_train.copy()
    for _ in range(iters):
        Q = P * (prior / prior_train)
        Q /= Q.sum(1, keepdims=True)
        new = Q.mean(0)
        if np.abs(new - prior).max() < tol:
            break
        prior = new
    return Q, prior


def bbse_prior(P_val, Y_val, P_test):
    """Black Box Shift Estimation (Lipton et al., 2018), versión suave: mu_test = A q, con
    A[i, j] = E_val[p_i | y = j]. Se resuelve q >= 0 por mínimos cuadrados y se normaliza."""
    from scipy.optimize import nnls
    A = (P_val.T @ Y_val) / Y_val.sum(0, keepdims=True)
    q, _ = nnls(A, P_test.mean(0))
    q = np.clip(q, 1e-4, None)
    return q / q.sum()


def prior_adjust(P, prior_train, prior_test):
    Q = P * (prior_test / prior_train)
    return Q / Q.sum(1, keepdims=True)


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    d = [pd.read_csv(OOF_DIR / f"{m}.csv").sort_values("eeg_id").reset_index(drop=True) for m in MEMBERS]
    assert (d[0].eeg_id.values == d[1].eeg_id.values).all()
    base = d[0][["eeg_id", "patient_id", "fold", "total_votes", "split"] + TARGETS].copy()
    Y = base[TARGETS].to_numpy()
    P1, P2 = d[0][PRED].to_numpy(), d[1][PRED].to_numpy()
    hq = (base.split == "valid_hq").to_numpy()
    folds = base.fold.to_numpy()

    # --- Optimización anidada por fold -------------------------------------------------------
    E = np.zeros_like(Y)
    rows_params = []
    for f in sorted(np.unique(folds)):
        opt_rows = hq & (folds != f)
        params, val, _ = optimize(P1[opt_rows], P2[opt_rows], Y[opt_rows])
        rows_params.append({"fold": f, **params, "kl_opt_otros_folds": val})
        apply = folds == f
        E[apply] = combine(P1[apply], P2[apply], params["w"], params["modo"], params["T"])
    params_df = pd.DataFrame(rows_params)
    print(params_df.round(4))

    # --- Parámetros finales (todas las predicciones de alta calidad) para despliegue -----------
    final, final_val, study = optimize(P1[hq], P2[hq], Y[hq], n_trials=150)
    study.trials_dataframe().to_csv(RESULTS_DIR / "optuna" / "ensemble_trials.csv", index=False)
    json.dump({"params": final, "kl_hq_in_sample": final_val, "por_fold": rows_params},
              open(RESULTS_DIR / "optuna" / "ensemble_best.json", "w"), indent=2, default=float)

    # --- Ajuste de priors en el test de baja calidad -------------------------------------------
    E_em = E.copy(); E_bb = E.copy(); E_or = E.copy(); est_em, est_bb = [], []
    for f in sorted(np.unique(folds)):
        trv = hq & (folds != f)
        tr_prior = Y[trv].mean(0)                        # prior del entrenamiento de ese fold
        te = (~hq) & (folds == f)
        E_em[te], pr = em_prior_shift(E[te], tr_prior)
        q = bbse_prior(E[trv], Y[trv], E[te])            # A estimada con validación de alta calidad de otros folds
        E_bb[te] = prior_adjust(E[te], tr_prior, q)
        E_or[te] = prior_adjust(E[te], tr_prior, Y[te].mean(0))
        est_em.append(pr); est_bb.append(q)
    priors = pd.DataFrame({"entrenamiento (>=10 votos)": Y[hq].mean(0), "EM en test": np.mean(est_em, 0),
                           "BBSE en test": np.mean(est_bb, 0), "real del test (<10 votos)": Y[~hq].mean(0)},
                          index=TARGETS)
    priors.to_csv(RESULTS_DIR / "optuna" / "priors_test.csv")
    print(priors.round(3))

    out = base.copy()
    out[PRED] = E
    out[[f"em_{t}" for t in TARGETS]] = E_em
    out[[f"bbse_{t}" for t in TARGETS]] = E_bb
    out.to_csv(OOF_DIR / "hq_ensemble.csv", index=False)

    # --- Tabla comparativa ----------------------------------------------------------------------
    rows = []

    def add(name, P, entreno, note=""):
        r = {"modelo": name, "entrenado_con": entreno, "nota": note}
        for split, m in [("valid_hq", hq), ("valid_hq_f1-4", hq & (folds != 0)), ("test_lq", ~hq)]:
            mm = {**all_metrics(Y[m], P[m]), **balanced_metrics(Y[m], P[m])}
            r.update({f"{split}_{k}": v for k, v in mm.items()})
        rows.append(r)

    old = {}
    for name in ["cv_effb0", "cv_convnext_t", "ensemble"]:
        o = pd.read_csv(OOF_DIR / f"{name}.csv").set_index("eeg_id").loc[base.eeg_id]
        old[name] = o[PRED].to_numpy()
    add("EfficientNet-B0 (fase 1)", old["cv_effb0"], "todos (2 etapas)")
    add("ConvNeXt-T (fase 1)", old["cv_convnext_t"], "todos (2 etapas)")
    add("Ensamble 50/50 (fase 1)", old["ensemble"], "todos (2 etapas)")
    if (OOF_DIR / "hq_effb0_default.csv").exists():
        c = pd.read_csv(OOF_DIR / "hq_effb0_default.csv").set_index("eeg_id").loc[base.eeg_id]
        add("EfficientNet-B0 sin optimizar", c[PRED].to_numpy(), ">=10 votos")
    add("EfficientNet-B0 optimizado", P1, ">=10 votos")
    add("ConvNeXt-T optimizado", P2, ">=10 votos")
    add("Ensamble 50/50", combine(P1, P2, 0.5, "arit", 1.0), ">=10 votos")
    add("Ensamble optimizado (anidado)", E, ">=10 votos", "w, modo, T por fold")
    add("Ensamble optimizado + EM priors", E_em, ">=10 votos", "ajuste no supervisado en test")
    add("Ensamble optimizado + BBSE priors", E_bb, ">=10 votos", "ajuste no supervisado en test")
    add("Ensamble optimizado + prior real", E_or, ">=10 votos", "oráculo, solo referencia")
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "metrics_hq_comparacion.csv", index=False)
    cols = ["modelo", "entrenado_con", "valid_hq_kl", "valid_hq_macro_kl", "valid_hq_balanced_accuracy",
            "valid_hq_macro_auc", "test_lq_kl", "test_lq_macro_kl", "test_lq_balanced_accuracy", "test_lq_macro_auc"]
    print(res[cols].round(4).to_string())
    print("Parámetros finales del ensamble:", final, "KL macro alta calidad (in-sample):", round(final_val, 4))


if __name__ == "__main__":
    main()
