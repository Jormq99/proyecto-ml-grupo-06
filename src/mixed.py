"""Iteración 3: aprender de EEG de alta calidad (>=10 votos) y de baja calidad (<10 votos).

Tres modos de entrenamiento (mismos 5 folds por paciente; cada modelo del fold f predice TODO el fold f,
alta y baja calidad, con pacientes nunca vistos):

  single  Enfoque A, modelo único: entrena con todos los EEG. La pérdida de los EEG de baja calidad se
          multiplica por lq_weight (hiperparámetro), así el optimizador decide cuánto pesan.
  dual    Enfoque A', un solo modelo con dos "lecturas": backbone compartido + cabeza para alta calidad
          + cabeza para baja calidad + compuerta que predice si el EEG es de baja calidad.
          Predicción final = g * P_baja + (1 - g) * P_alta, con g la probabilidad de la compuerta.
  lq      Enfoque B, especialista: entrena SOLO con baja calidad (el especialista de alta calidad es hq_*
          de la iteración 2).

Objetivo de la optimización bayesiana (fold 0):
  single/dual: "general" = promedio de la KL macro en validación de alta y en validación de baja calidad.
  lq:          KL macro en validación de baja calidad.

    python -m src.mixed tune --arch effb0 --mode single --trials 16
    python -m src.mixed cv   --arch effb0 --mode dual --params mix_single_effb0
"""
import argparse
import json
import math
import time

import numpy as np
import optuna
import pandas as pd
import timm
import torch
import torch.nn.functional as F

from .config import MODELS_DIR, OOF_DIR, RESULTS_DIR, SEED, TARGETS
from .evaluate import all_metrics, balanced_metrics
from .hq import ARCHS, HQ_MIN_VOTES, OPT_DIR, augment, load_data, macro_kl
from .train import assemble, seed_all

DEVICE = "cuda"
K = len(TARGETS)
PRED = [f"pred_{t}" for t in TARGETS]

# Punto de partida: mejores hiperparámetros de la iteración 2 (solo alta calidad)
BASE = {a: json.load(open(OPT_DIR / f"{a}_best.json"))["params"] for a in ARCHS}


def general_score(Y, P, lq):
    """Devuelve (general, macro_kl alta, macro_kl baja)."""
    h, l = macro_kl(Y[~lq], P[~lq]), macro_kl(Y[lq], P[lq])
    return 0.5 * (h + l), h, l


# ----------------------------------------------------------------------------
def predict_full(model, X, idx, mode, bs=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            xb = torch.from_numpy(X[idx[i:i + bs]]).to(DEVICE).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(assemble(xb, "both").contiguous(memory_format=torch.channels_last)).float().cpu())
    z = torch.cat(out)
    if mode != "dual":
        return {"pred": torch.softmax(z, 1).numpy()}
    ph, pl = torch.softmax(z[:, :K], 1).numpy(), torch.softmax(z[:, K:2 * K], 1).numpy()
    g = torch.sigmoid(z[:, 2 * K]).numpy()
    return {"pred": g[:, None] * pl + (1 - g[:, None]) * ph, "hq": ph, "lq": pl, "gate": g}


def fit(arch, p, X, Y, LQ, tr_idx, va_idx, mode, report=None, log=None):
    seed_all(SEED)
    a = ARCHS[arch]
    n_out = 2 * K + 1 if mode == "dual" else K
    model = timm.create_model(a["timm"], pretrained=True, num_classes=n_out, in_chans=1,
                              drop_rate=p["drop_rate"], drop_path_rate=p["drop_path_rate"])
    model = model.to(DEVICE).to(memory_format=torch.channels_last)
    bs = p["batch_size"]
    accum = max(1, bs // a["micro_bs"])
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    steps_ep = len(tr_idx) // bs
    steps = p["epochs"] * steps_ep
    warm = max(1, int(p["warmup"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm)))) + 1e-3)

    # peso por muestra: (1/frecuencia de clase)^class_weight  x  lq_weight si es de baja calidad; media 1
    cons = Y[tr_idx].argmax(1)
    freq = np.bincount(cons, minlength=K) / len(cons)
    w = np.ones(len(Y), np.float32)
    w[tr_idx] = (1.0 / np.clip(freq[cons], 1e-3, None)) ** p.get("class_weight", 0.0)
    if mode in ("single", "dual"):
        w[tr_idx] *= np.where(LQ[tr_idx], p.get("lq_weight", 1.0), 1.0)
    w[tr_idx] /= w[tr_idx].mean()
    Yt, Wt, Qt = torch.from_numpy(Y).float(), torch.from_numpy(w), torch.from_numpy(LQ.astype(np.float32))
    beta = torch.distributions.Beta(p["mixup"], p["mixup"]) if p["mixup"] > 0 else None
    gate_w = p.get("gate_weight", 0.3)

    for ep in range(p["epochs"]):
        model.train()
        perm = np.random.permutation(tr_idx)
        tot, n, t0 = 0.0, 0, time.time()
        for i in range(steps_ep):
            b = np.sort(perm[i * bs:(i + 1) * bs])
            opt.zero_grad(set_to_none=True)
            for mb in np.array_split(b, accum):
                xb = torch.from_numpy(X[mb]).to(DEVICE, non_blocking=True).float()
                yb, wb, qb = Yt[mb].to(DEVICE), Wt[mb].to(DEVICE), Qt[mb].to(DEVICE)
                xb = augment(xb, p["swap_p"], p["n_masks"], p["mask_frac"])
                if beta is not None and np.random.rand() < 0.5:
                    lam = beta.sample().item()
                    j = torch.randperm(len(mb), device=DEVICE)
                    xb = lam * xb + (1 - lam) * xb[j]
                    yb = lam * yb + (1 - lam) * yb[j]
                    wb = lam * wb + (1 - lam) * wb[j]
                    qb = lam * qb + (1 - lam) * qb[j]
                xb = assemble(xb, "both").contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z = model(xb)
                z = z.float()
                ent = (yb * torch.log(yb.clamp_min(1e-8))).sum(1)
                if mode == "dual":
                    kl_h = ent - (yb * torch.log_softmax(z[:, :K], 1)).sum(1)
                    kl_l = ent - (yb * torch.log_softmax(z[:, K:2 * K], 1)).sum(1)
                    per = qb * kl_l + (1 - qb) * kl_h
                    loss = (per * wb).mean() + gate_w * F.binary_cross_entropy_with_logits(z[:, 2 * K], qb)
                else:
                    per = ent - (yb * torch.log_softmax(z, 1)).sum(1)
                    loss = (per * wb).mean()
                (loss * len(mb) / len(b)).backward()
                tot += loss.item() * len(mb); n += len(mb)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        out = predict_full(model, X, va_idx, mode)
        lq_va, Yv, Pv = LQ[va_idx], Y[va_idx], out["pred"]
        h = macro_kl(Yv[~lq_va], Pv[~lq_va]) if (~lq_va).any() else float("nan")
        l = macro_kl(Yv[lq_va], Pv[lq_va]) if lq_va.any() else float("nan")
        obj = l if mode == "lq" else 0.5 * (h + l)
        if log is not None:
            log.append({"epoch": ep + 1, "train_loss": tot / n, "valid_macro_kl_hq": h, "valid_macro_kl_lq": l,
                        "objetivo": obj, "sec": time.time() - t0})
        print(f"  ep {ep + 1}/{p['epochs']} loss {tot / n:.4f} macroKL alta {h:.4f} baja {l:.4f} "
              f"obj {obj:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if report is not None:
            report(int(round(10 * (ep + 1) / p["epochs"])), obj)
    return model


# ----------------------------------------------------------------------------
def suggest(trial, arch, mode):
    lo, hi = ARCHS[arch]["lr"]
    p = dict(BASE[arch])
    p.update(
        lr=trial.suggest_float("lr", lo, hi, log=True),
        epochs=trial.suggest_int("epochs", 5, 14),
        drop_rate=trial.suggest_float("drop_rate", 0.0, 0.5),
        n_masks=trial.suggest_int("n_masks", 1, 3),
        mask_frac=trial.suggest_float("mask_frac", 0.1, 0.3),
        mixup=trial.suggest_float("mixup", 0.0, 0.3),
        class_weight=trial.suggest_float("class_weight", 0.0, 1.0),
    )
    if mode in ("single", "dual"):
        p["lq_weight"] = trial.suggest_float("lq_weight", 0.05, 1.0, log=True)
    return p


def splits(meta, LQ, f, mode):
    fold = meta.fold.to_numpy()
    tr = (fold != f) & LQ if mode == "lq" else (fold != f)
    return np.where(tr)[0], np.where(fold == f)[0]


def tune(args):
    meta, X, Y = load_data(only_hq=False)
    LQ = meta.total_votes.to_numpy() < HQ_MIN_VOTES
    tr, va = splits(meta, LQ, args.fold, args.mode)
    if args.mode == "lq":
        va = va[LQ[va]]                      # el especialista se optimiza solo en su dominio
    name = f"mix_{args.mode}_{args.arch}"
    print(f"Optimización {name}: train {len(tr)} / valid {len(va)} (fold {args.fold})", flush=True)
    study = optuna.create_study(
        study_name=name, storage=f"sqlite:///{(OPT_DIR / f'{name}.db').as_posix()}", direction="minimize",
        load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=6, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=5))
    keys = ["lr", "epochs", "drop_rate", "n_masks", "mask_frac", "mixup", "class_weight"]
    if len(study.trials) == 0:
        start = {k: BASE[args.arch][k] for k in keys}
        start["mask_frac"] = float(np.clip(start["mask_frac"], 0.1, 0.3))
        start["n_masks"] = max(1, start["n_masks"])
        if args.mode == "lq":
            study.enqueue_trial(start)
        else:   # todos los datos con igual peso, y baja calidad a 0.3
            study.enqueue_trial({**start, "lq_weight": 1.0})
            study.enqueue_trial({**start, "lq_weight": 0.3})

    def objective(trial):
        p = suggest(trial, args.arch, args.mode)
        print(f"trial {trial.number}: {json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in p.items()})}",
              flush=True)
        t0, last = time.time(), {}

        def report(step, v):
            last["v"] = v
            trial.report(v, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        try:
            del_model = fit(args.arch, p, X, Y, LQ, tr, va, args.mode, report=report)
            del del_model
        finally:
            torch.cuda.empty_cache()
            trial.set_user_attr("minutes", round((time.time() - t0) / 60, 2))
        return last["v"]

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True,
                   callbacks=[lambda s, t: print(f"trial {t.number} {t.state.name} obj={t.value} "
                                                 f"best={s.best_value:.4f}", flush=True)])
    study.trials_dataframe().to_csv(OPT_DIR / f"{name}_trials.csv", index=False)
    best = {**BASE[args.arch], **study.best_params}
    json.dump({"value": study.best_value, "trial": study.best_trial.number, "params": best},
              open(OPT_DIR / f"{name}_best.json", "w"), indent=2)
    print("MEJOR", study.best_value, best, flush=True)


def cv(args):
    if args.params == "base":                 # mejores de la iteración 2, sin optimizar de nuevo
        p = dict(BASE[args.arch])
    elif args.params.startswith("{"):
        p = json.loads(args.params)
    else:
        p = json.load(open(OPT_DIR / f"{args.params}_best.json"))["params"]
    p.update(json.loads(args.override or "{}"))
    tag = args.tag or f"mix_{args.mode}_{args.arch}"
    print(tag, args.mode, p, flush=True)
    meta, X, Y = load_data(only_hq=False)
    LQ = meta.total_votes.to_numpy() < HQ_MIN_VOTES
    out = {"pred": np.zeros_like(Y)}
    if args.mode == "dual":
        out.update(hq=np.zeros_like(Y), lq=np.zeros_like(Y), gate=np.zeros(len(Y), np.float32))
    log, t0 = [], time.time()
    for f in sorted(meta.fold.unique()):
        tr, va = splits(meta, LQ, f, args.mode)
        flog = []
        model = fit(args.arch, p, X, Y, LQ, tr, va, args.mode, log=flog)
        o = predict_full(model, X, va, args.mode)
        for k in out:
            out[k][va] = o[k]
        g, h, l = general_score(Y[va], o["pred"], LQ[va])
        print(f"fold {f}: general {g:.4f} | macroKL alta {h:.4f} | baja {l:.4f}", flush=True)
        log += [{**r, "fold": f} for r in flog]
        torch.save(model.state_dict(), MODELS_DIR / f"{tag}_fold{f}.pt")
        del model; torch.cuda.empty_cache()
    pd.DataFrame(log).to_csv(RESULTS_DIR / "history" / f"{tag}.csv", index=False)
    df = meta[["eeg_id", "patient_id", "fold", "total_votes"] + TARGETS].copy()
    df["split"] = np.where(LQ, "lq", "hq")
    df[PRED] = out["pred"]
    if args.mode == "dual":
        df[[f"hq_{t}" for t in TARGETS]] = out["hq"]
        df[[f"lq_{t}" for t in TARGETS]] = out["lq"]
        df["gate"] = out["gate"]
    df.to_csv(OOF_DIR / f"{tag}.csv", index=False)
    res = {"experiment": tag, "arch": args.arch, "mode": args.mode, "params": json.dumps(p),
           "minutes": round((time.time() - t0) / 60, 1),
           "general": general_score(Y, out["pred"], LQ)[0]}
    for name, m in [("hq", ~LQ), ("lq", LQ), ("todas", np.ones(len(Y), bool))]:
        for k, v in {**all_metrics(Y[m], out["pred"][m]), **balanced_metrics(Y[m], out["pred"][m])}.items():
            res[f"{name}_{k}"] = v
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    path = RESULTS_DIR / "metrics_mixed.csv"
    new = pd.DataFrame([res])
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old.experiment != tag], new])
    new.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "cv"])
    ap.add_argument("--arch", choices=list(ARCHS), required=True)
    ap.add_argument("--mode", choices=["single", "dual", "lq"], required=True)
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--params", default=None, help="nombre de estudio (p. ej. mix_single_effb0) o JSON")
    ap.add_argument("--override", default=None, help="JSON con valores que reemplazan a los de --params")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.cmd == "cv" and args.params is None:
        args.params = f"mix_{args.mode}_{args.arch}"
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.85)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tune(args) if args.cmd == "tune" else cv(args)


if __name__ == "__main__":
    main()
