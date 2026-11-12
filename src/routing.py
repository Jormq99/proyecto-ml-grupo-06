"""Iteración 3: ensambles y enrutamiento entre especialistas de alta y baja calidad.

Todas las decisiones (pesos de ensamble, umbrales, compuerta) se ajustan de forma ANIDADA: lo que se aplica
al fold f se optimiza solo con las predicciones fuera de fold de los otros 4 folds.

Objetivo "general" = promedio de la KL macro en alta calidad (>=10 votos) y en baja calidad (<10 votos).

Estrategias comparadas
  A   modelo único (todos los datos)            ensamble B0 + ConvNeXt
  A'  modelo de dos cabezas + compuerta          (B0, y ensamble si existe la versión ConvNeXt)
  B   especialistas H (solo alta) y L (solo baja), cada uno ensamble B0 + ConvNeXt, combinados por:
        oráculo (sabe cuántos votos tiene el EEG; NO desplegable, cota superior)
        solo H / solo L
        mezcla fija (ensamble de especialistas)
        cascada por confianza H->L y L->H (umbral tau)
        compuerta aprendida (regresión logística que predice "baja calidad" a partir de las salidas de H y L)

    python -m src.routing
"""
import json

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CLASSES, OOF_DIR, RESULTS_DIR, SEED, TARGETS
from .ensemble_opt import combine
from .evaluate import all_metrics, balanced_metrics
from .hq import macro_kl

PRED = [f"pred_{t}" for t in TARGETS]
EPS = 1e-7
OUT = RESULTS_DIR / "mixed"
OUT.mkdir(exist_ok=True)


def load(name, cols=PRED):
    p = OOF_DIR / f"{name}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p).sort_values("eeg_id").reset_index(drop=True)
    return d if cols is None else d[cols].to_numpy()


def general(Y, P, lq):
    return 0.5 * (macro_kl(Y[~lq], P[~lq]) + macro_kl(Y[lq], P[lq]))


def objective_fn(kind):
    if kind == "general":
        return general
    if kind == "hq":
        return lambda Y, P, lq: macro_kl(Y[~lq], P[~lq])
    return lambda Y, P, lq: macro_kl(Y[lq], P[lq])


def optimize_pair(P1, P2, Y, lq, kind, n_trials=60):
    f = objective_fn(kind)
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    st.enqueue_trial({"w": 0.5, "modo": "arit", "T": 1.0})
    st.optimize(lambda t: f(Y, combine(P1, P2, t.suggest_float("w", 0, 1),
                                       t.suggest_categorical("modo", ["arit", "geom"]),
                                       t.suggest_float("T", 0.5, 2.0, log=True)), lq), n_trials=n_trials)
    return st.best_params


def nested_pair(P1, P2, Y, lq, folds, kind, rows=None):
    """Ensamble de dos modelos con (w, modo, T) optimizados en los otros folds."""
    rows = np.ones(len(Y), bool) if rows is None else rows
    E, params = np.zeros_like(P1), []
    for f in np.unique(folds):
        m = rows & (folds != f)
        p = optimize_pair(P1[m], P2[m], Y[m], lq[m], kind)
        params.append({"fold": int(f), **p})
        a = folds == f
        E[a] = combine(P1[a], P2[a], p["w"], p["modo"], p["T"])
    return E, params


def entropy(P):
    return -(P * np.log(np.clip(P, EPS, 1))).sum(1)


def gate_features(H, L, extra=None):
    M = 0.5 * (H + L)
    js = entropy(M) - 0.5 * (entropy(H) + entropy(L))
    F = [H, L, H.max(1, keepdims=True), L.max(1, keepdims=True), entropy(H)[:, None], entropy(L)[:, None],
         js[:, None], (H.argmax(1) == L.argmax(1))[:, None].astype(float)]
    if extra is not None:
        F.append(extra.reshape(len(H), -1))
    return np.hstack(F)


def nested_gate(F, lq, folds):
    g = np.zeros(len(lq))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(F[tr], lq[tr])
        g[te] = clf.predict_proba(F[te])[:, 1]
    return g


def mix(H, L, g):
    return g[:, None] * L + (1 - g[:, None]) * H


def cascade(A, B, tau):
    use_a = A.max(1) >= tau
    return np.where(use_a[:, None], A, B), use_a


def nested_cascade(A, B, Y, lq, folds, grid=np.round(np.arange(0.20, 1.001, 0.025), 3)):
    E, use_a, taus = np.zeros_like(A), np.zeros(len(A), bool), []
    for f in np.unique(folds):
        m = folds != f
        scores = [general(Y[m], cascade(A[m], B[m], t)[0], lq[m]) for t in grid]
        t = float(grid[int(np.argmin(scores))])
        taus.append(t)
        a = folds == f
        E[a], use_a[a] = cascade(A[a], B[a], t)
    return E, use_a, taus


def nested_gate_threshold(H, L, g, Y, lq, folds, grid=np.round(np.arange(0.05, 0.951, 0.05), 2)):
    """Enrutamiento duro: usa L si g >= umbral (umbral optimizado en los otros folds)."""
    E, taus = np.zeros_like(H), []
    for f in np.unique(folds):
        m = folds != f
        scores = [general(Y[m], np.where((g[m] >= t)[:, None], L[m], H[m]), lq[m]) for t in grid]
        t = float(grid[int(np.argmin(scores))])
        taus.append(t)
        a = folds == f
        E[a] = np.where((g[a] >= t)[:, None], L[a], H[a])
    return E, taus


def ece(Y, P, bins=10):
    conf, ok = P.max(1), (P.argmax(1) == Y.argmax(1))
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(ok[s].mean() - conf[s].mean())
    return e


def confidence_rows(name, Y, P, lq):
    rows = []
    for sub, m in [("alta (>=10)", ~lq), ("baja (<10)", lq)]:
        conf, ok = P[m].max(1), P[m].argmax(1) == Y[m].argmax(1)
        r = {"modelo": name, "subconjunto": sub, "accuracy": ok.mean(), "conf_media": conf.mean(), "ece": ece(Y[m], P[m])}
        for t in (0.6, 0.8, 0.9):
            s = conf >= t
            r[f"cobertura_conf>={t}"] = s.mean()
            r[f"error_conf>={t}"] = 1 - ok[s].mean() if s.any() else np.nan
        rows.append(r)
    return rows


def main():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = load("hq_effb0", None)
    Y = base[TARGETS].to_numpy()
    lq = (base.total_votes < 10).to_numpy()
    folds = base.fold.to_numpy()
    eeg = base.eeg_id.to_numpy()
    for n in ["mix_single_effb0", "mix_lq_effb0"]:
        d = load(n, None)
        assert d is not None and (d.eeg_id.values == eeg).all(), n

    S = {}           # estrategia -> (P, nota)
    params = {}
    extra_cols = {}  # estrategia -> fracción enrutada a L

    # --- referencias de iteraciones anteriores -------------------------------------------------
    S["Fase 1: ensamble 50/50 (todos, 2 etapas)"] = (load("ensemble"), "iteración 1, sin optimizar")
    H = load("hq_ensemble")
    S["Iter. 2: solo alta calidad (ensamble)"] = (H, "= especialista H")

    # --- A: modelo único ---------------------------------------------------------------------------
    u1, u2 = load("mix_single_effb0"), load("mix_single_convnext_t")
    S["A: único B0"] = (u1, "todos los datos, lq_weight optimizado")
    if u2 is not None:
        S["A: único ConvNeXt-T"] = (u2, "")
        U, params["A_ensamble"] = nested_pair(u1, u2, Y, lq, folds, "general")
        S["A: único, ensamble optimizado"] = (U, "w, modo, T anidados (objetivo general)")
    else:
        U = u1

    # --- A': dos cabezas + compuerta --------------------------------------------------------------
    d1, d2 = load("mix_dual_effb0", None), load("mix_dual_convnext_t", None)
    hqc, lqc = [f"hq_{t}" for t in TARGETS], [f"lq_{t}" for t in TARGETS]
    dual_gate = None
    if d1 is not None:
        S["A': dos cabezas B0 (compuerta interna)"] = (d1[PRED].to_numpy(), "g*P_baja + (1-g)*P_alta")
        Dh, Dl, dual_gate = d1[hqc].to_numpy(), d1[lqc].to_numpy(), d1.gate.to_numpy()
        if d2 is not None:
            S["A': dos cabezas ConvNeXt-T"] = (d2[PRED].to_numpy(), "")
            Dh, params["Ad_h"] = nested_pair(Dh, d2[hqc].to_numpy(), Y, lq, folds, "hq", rows=~lq)
            Dl, params["Ad_l"] = nested_pair(Dl, d2[lqc].to_numpy(), Y, lq, folds, "lq", rows=lq)
            dual_gate = 0.5 * (dual_gate + d2.gate.to_numpy())
            S["A': dos cabezas, ensamble (compuerta interna)"] = (mix(Dh, Dl, dual_gate), "")
        S["A': dos cabezas, cabeza según votos reales (oráculo)"] = (np.where(lq[:, None], Dl, Dh), "NO desplegable")

    # --- B: especialistas ---------------------------------------------------------------------------
    l1, l2 = load("mix_lq_effb0"), load("mix_lq_convnext_t")
    if l2 is not None:
        L, params["B_L"] = nested_pair(l1, l2, Y, lq, folds, "lq", rows=lq)
    else:
        L = l1
    S["B: solo especialista L (baja)"] = (L, "entrenado solo con <10 votos")
    S["B: oráculo (H si >=10 votos, L si <10)"] = (np.where(lq[:, None], L, H), "NO desplegable: usa los votos")
    M, params["B_mezcla"] = nested_pair(H, L, Y, lq, folds, "general")
    S["B: mezcla fija H+L"] = (M, "ensamble de especialistas")
    C, use_h, params["B_cascada_HL"] = nested_cascade(H, L, Y, lq, folds)
    S["B: cascada H -> L por confianza"] = (C, "usa H si max(P_H) >= tau")
    extra_cols["B: cascada H -> L por confianza"] = 1 - use_h.mean()
    C2, use_l, params["B_cascada_LH"] = nested_cascade(L, H, Y, lq, folds)
    S["B: cascada L -> H por confianza"] = (C2, "usa L si max(P_L) >= tau")
    extra_cols["B: cascada L -> H por confianza"] = use_l.mean()

    F = gate_features(H, L)
    g = nested_gate(F, lq, folds)
    gate_auc = {"compuerta (salidas de H y L)": roc_auc_score(lq, g)}
    S["B: compuerta aprendida, mezcla suave"] = (mix(H, L, g), "g = P(baja calidad | salidas de H, L)")
    Gh, params["B_compuerta_umbral"] = nested_gate_threshold(H, L, g, Y, lq, folds)
    S["B: compuerta aprendida, enrutamiento duro"] = (Gh, "L si g >= umbral")
    if dual_gate is not None:
        gate_auc["compuerta interna del modelo de dos cabezas"] = roc_auc_score(lq, dual_gate)
        g2 = nested_gate(gate_features(H, L, dual_gate), lq, folds)
        gate_auc["compuerta (salidas de H, L + compuerta CNN)"] = roc_auc_score(lq, g2)
        S["B: compuerta (H, L + CNN), mezcla suave"] = (mix(H, L, g2), "")
        Gh2, params["B_compuerta2_umbral"] = nested_gate_threshold(H, L, g2, Y, lq, folds)
        S["B: compuerta (H, L + CNN), enrutamiento duro"] = (Gh2, "")
        fr = np.where(g2 >= np.array(params["B_compuerta2_umbral"])[folds], 1, 0)
        extra_cols["B: compuerta (H, L + CNN), enrutamiento duro"] = fr.mean()
    extra_cols["B: compuerta aprendida, enrutamiento duro"] = \
        (g >= np.array(params["B_compuerta_umbral"])[folds]).mean()
    extra_cols["B: oráculo (H si >=10 votos, L si <10)"] = lq.mean()

    # --- tabla comparativa ---------------------------------------------------------------------------
    rows, conf_rows, cls_rows = [], [], []
    for name, (P, note) in S.items():
        if P is None:
            continue
        r = {"estrategia": name, "nota": note, "general": general(Y, P, lq), "frac_a_L": extra_cols.get(name, np.nan)}
        for sub, m in [("alta", ~lq), ("baja", lq), ("todas", np.ones(len(Y), bool))]:
            mm = {**all_metrics(Y[m], P[m]), **balanced_metrics(Y[m], P[m])}
            r.update({f"{sub}_{k}": v for k, v in mm.items()})
        rows.append(r)
        conf_rows += confidence_rows(name, Y, P, lq)
        yt, yp = Y.argmax(1), P.argmax(1)
        for sub, m in [("alta", ~lq), ("baja", lq)]:
            for c, cn in enumerate(CLASSES):
                s = m & (yt == c)
                cls_rows.append({"estrategia": name, "subconjunto": sub, "clase": cn, "n": int(s.sum()),
                                 "recall": (yp[s] == c).mean() if s.any() else np.nan})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "metrics_mixed_comparacion.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(OUT / "confianza.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(OUT / "recall_por_clase.csv", index=False)
    json.dump({"params": params, "gate_auc": gate_auc}, open(OUT / "routing_params.json", "w"), indent=2, default=float)

    oof = base[["eeg_id", "patient_id", "fold", "total_votes"] + TARGETS].copy()
    oof["g_compuerta"] = g
    for tag, P in [("H", H), ("L", L), ("U", U), ("mezcla", M)]:
        oof[[f"{tag}_{t}" for t in TARGETS]] = P
    if dual_gate is not None:
        oof["g_cnn"] = dual_gate
    oof.to_csv(OOF_DIR / "mixed_routing.csv", index=False)

    cols = ["estrategia", "general", "alta_macro_kl", "baja_macro_kl", "alta_kl", "baja_kl", "todas_kl",
            "alta_balanced_accuracy", "baja_balanced_accuracy", "todas_macro_auc", "frac_a_L"]
    with pd.option_context("display.width", 260, "display.max_colwidth", 52):
        print(res[cols].round(4).to_string())
        print("\nAUC de la compuerta (¿se distingue alta de baja calidad?):", {k: round(v, 4) for k, v in gate_auc.items()})
        cf = pd.DataFrame(conf_rows)
        print(cf[["modelo", "subconjunto", "accuracy", "ece", "cobertura_conf>=0.8", "error_conf>=0.8"]].round(3).to_string())


if __name__ == "__main__":
    main()
