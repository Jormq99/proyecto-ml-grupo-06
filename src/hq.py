"""Entrenamiento SOLO con EEG de alta calidad (>=10 votos), optimización bayesiana de
hiperparámetros (Optuna/TPE) y evaluación de los EEG de baja calidad (<10 votos) como test.

Protocolo (sin fuga por paciente):
  * Se reutilizan los 5 folds agrupados por paciente.
  * Modelo del fold f: entrena con alta calidad de los folds != f; predice
      - validación: alta calidad del fold f
      - test:       baja calidad del fold f  (nunca usada para entrenar ni para elegir hiperparámetros)
  * La optimización bayesiana usa solo la KL de validación de alta calidad del fold 0.
    Por eso la CV de alta calidad se reporta también excluyendo el fold 0.

Uso:
    python -m src.hq tune --arch effb0 --trials 30
    python -m src.hq cv   --arch effb0 --params best          # mejores hiperparámetros del estudio
    python -m src.hq cv   --arch effb0 --params default --tag hq_effb0_default
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

from .config import CACHE_DIR, MODELS_DIR, OOF_DIR, RESULTS_DIR, SEED, TARGETS
from .evaluate import all_metrics, balanced_metrics, kl_divergence, kl_per_row
from .train import assemble, kl_loss, predict, seed_all

DEVICE = "cuda"
OPT_DIR = RESULTS_DIR / "optuna"
OPT_DIR.mkdir(exist_ok=True)
HQ_MIN_VOTES = 10
OBJECTIVE = "macro_kl"   # objetivo de la optimización: "macro_kl" (balanceada) o "kl"

ARCHS = {
    "effb0": {"timm": "efficientnet_b0.ra_in1k", "lr": (2e-4, 3e-3), "micro_bs": 32},
    "convnext_t": {"timm": "convnext_tiny.fb_in22k_ft_in1k", "lr": (2e-5, 5e-4), "micro_bs": 16},
}

# Configuración usada en la fase anterior, como punto de partida y como control "sin optimizar"
DEFAULTS = {
    "effb0": dict(lr=1e-3, weight_decay=1e-2, epochs=12, batch_size=32, drop_rate=0.2,
                  drop_path_rate=0.0, warmup=0.1, swap_p=0.5, n_masks=2, mask_frac=0.125, mixup=0.0, class_weight=0.0),
    "convnext_t": dict(lr=2e-4, weight_decay=1e-2, epochs=12, batch_size=32, drop_rate=0.2,
                       drop_path_rate=0.0, warmup=0.1, swap_p=0.5, n_masks=2, mask_frac=0.125, mixup=0.0, class_weight=0.0),
}


def macro_kl(Y, P):
    """KL promediada por clase de consenso: cada clase pesa igual (robusta al desbalance)."""
    klr, yt = kl_per_row(Y, P), Y.argmax(1)
    return float(np.mean([klr[yt == c].mean() for c in np.unique(yt)]))


# ----------------------------------------------------------------------------
def augment(x, swap_p, n_masks, mask_frac):
    """Intercambio de hemisferios + enmascarado tiempo/frecuencia, parametrizados."""
    B, _, H, W = x.shape
    swap = torch.rand(B, device=x.device) < swap_p
    x = torch.where(swap[:, None, None, None], x[:, [3, 2, 1, 0, 7, 6, 5, 4]], x)
    if n_masks == 0 or mask_frac <= 0:
        return x
    t = torch.arange(W, device=x.device)[None, :]
    f = torch.arange(H, device=x.device)[None, :]
    mask = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
    wmax, hmax = max(1, int(W * mask_frac)), max(1, int(H * mask_frac))
    for _ in range(n_masks):
        w = torch.randint(0, wmax + 1, (B, 1), device=x.device)
        s = (torch.rand(B, 1, device=x.device) * (W - w)).long()
        mask = mask * (~((t >= s) & (t < s + w)))[:, None, None, :]
        h = torch.randint(0, hmax + 1, (B, 1), device=x.device)
        s = (torch.rand(B, 1, device=x.device) * (H - h)).long()
        mask = mask * (~((f >= s) & (f < s + h)))[:, None, :, None]
    return x * mask


def fit(arch, p, X, Y, tr_idx, va_idx, report=None, log=None):
    """Entrena un modelo con hiperparámetros p. report(epoch_frac_step, val_kl) para poda."""
    seed_all(SEED)
    a = ARCHS[arch]
    model = timm.create_model(a["timm"], pretrained=True, num_classes=len(TARGETS), in_chans=1,
                              drop_rate=p["drop_rate"], drop_path_rate=p["drop_path_rate"])
    model = model.to(DEVICE).to(memory_format=torch.channels_last)
    bs = p["batch_size"]
    accum = max(1, bs // a["micro_bs"])
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    steps_ep = len(tr_idx) // bs
    steps = p["epochs"] * steps_ep
    warm = max(1, int(p["warmup"] * steps))

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm)))) + 1e-3
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    Yt = torch.from_numpy(Y).float()
    # peso por muestra = (1 / frecuencia de su clase de consenso en train) ** alpha, media 1
    cons = Y[tr_idx].argmax(1)
    freq = np.bincount(cons, minlength=Y.shape[1]) / len(cons)
    w_all = np.ones(len(Y), np.float32)
    w_all[tr_idx] = (1.0 / np.clip(freq[cons], 1e-3, None)) ** p.get("class_weight", 0.0)
    w_all[tr_idx] /= w_all[tr_idx].mean()
    Wt = torch.from_numpy(w_all)
    beta = torch.distributions.Beta(p["mixup"], p["mixup"]) if p["mixup"] > 0 else None
    Yva = Y[va_idx]
    for ep in range(p["epochs"]):
        model.train()
        perm = np.random.permutation(tr_idx)
        tot, n, t0 = 0.0, 0, time.time()
        for i in range(steps_ep):
            b = np.sort(perm[i * bs:(i + 1) * bs])
            opt.zero_grad(set_to_none=True)
            for mb in np.array_split(b, accum):
                xb = torch.from_numpy(X[mb]).to(DEVICE, non_blocking=True).float()
                yb = Yt[mb].to(DEVICE)
                wb = Wt[mb].to(DEVICE)
                xb = augment(xb, p["swap_p"], p["n_masks"], p["mask_frac"])
                if beta is not None and np.random.rand() < 0.5:   # mixup sobre imagen y distribución de votos
                    lam = beta.sample().item()
                    j = torch.randperm(len(mb), device=DEVICE)
                    xb = lam * xb + (1 - lam) * xb[j]
                    yb = lam * yb + (1 - lam) * yb[j]
                    wb = lam * wb + (1 - lam) * wb[j]
                xb = assemble(xb, "both").contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(xb)
                logp = torch.log_softmax(logits.float(), 1)
                per = (yb * (torch.log(yb.clamp_min(1e-8)) - logp)).sum(1)
                loss = (per * wb).mean()
                (loss * len(mb) / len(b)).backward()
                tot += loss.item() * len(mb); n += len(mb)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        pv = predict(model, X, va_idx, "both")
        kl, mkl = kl_divergence(Yva, pv), macro_kl(Yva, pv)
        if log is not None:
            log.append({"epoch": ep + 1, "train_loss": tot / n, "valid_kl_hq": kl,
                        "valid_macro_kl_hq": mkl, "sec": time.time() - t0})
        if report is not None:
            report(int(round(10 * (ep + 1) / p["epochs"])), mkl if OBJECTIVE == "macro_kl" else kl)   # paso = décimo del calendario
    return model


def load_data(only_hq=False):
    meta = pd.read_csv(CACHE_DIR / "meta.csv")
    X = np.load(CACHE_DIR / "images.npy", mmap_mode="r")
    keep = meta.cache_ok.to_numpy()
    if only_hq:
        keep = keep & (meta.total_votes.to_numpy() >= HQ_MIN_VOTES)
    idx = np.where(keep)[0]
    print(f"Cargando {len(idx)} imágenes en RAM...", flush=True)
    X = np.ascontiguousarray(X[idx])
    meta = meta.iloc[idx].reset_index(drop=True)
    return meta, X, meta[TARGETS].to_numpy(np.float32)


# ----------------------------------------------------------------------------
def suggest(trial, arch):
    lo, hi = ARCHS[arch]["lr"]
    return dict(
        lr=trial.suggest_float("lr", lo, hi, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        epochs=trial.suggest_int("epochs", 6, 20),
        batch_size=trial.suggest_categorical("batch_size", [16, 32]),
        drop_rate=trial.suggest_float("drop_rate", 0.0, 0.5),
        drop_path_rate=trial.suggest_float("drop_path_rate", 0.0, 0.3),
        warmup=trial.suggest_float("warmup", 0.0, 0.2),
        swap_p=trial.suggest_float("swap_p", 0.0, 0.5),
        n_masks=trial.suggest_int("n_masks", 0, 3),
        mask_frac=trial.suggest_float("mask_frac", 0.0, 0.25),
        mixup=trial.suggest_float("mixup", 0.0, 0.4),
        class_weight=trial.suggest_float("class_weight", 0.0, 1.0),
    )


def tune(args):
    meta, X, Y = load_data(only_hq=True)
    tr = np.where(meta.fold != args.fold)[0]
    va = np.where(meta.fold == args.fold)[0]
    print(f"Optimización {args.arch}: train {len(tr)} / valid {len(va)} EEG de alta calidad (fold {args.fold})")
    storage = f"sqlite:///{(OPT_DIR / f'{args.arch}.db').as_posix()}"
    study = optuna.create_study(
        study_name=args.arch, storage=storage, direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=5))
    if len(study.trials) == 0:
        study.enqueue_trial(DEFAULTS[args.arch])   # el primer intento es la configuración anterior

    def objective(trial):
        p = suggest(trial, args.arch)
        t0 = time.time()
        last = {}

        def report(step, kl):
            last["kl"] = kl
            trial.report(kl, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        try:
            model = fit(args.arch, p, X, Y, tr, va, report=report)
            del model
        finally:
            torch.cuda.empty_cache()
            trial.set_user_attr("minutes", round((time.time() - t0) / 60, 2))
        return last["kl"]

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True,
                   callbacks=[lambda s, t: print(f"trial {t.number} {t.state.name} "
                                                 f"kl={t.value} best={s.best_value:.4f}", flush=True)])
    df = study.trials_dataframe()
    df.to_csv(OPT_DIR / f"{args.arch}_trials.csv", index=False)
    best = {**DEFAULTS[args.arch], **study.best_params}
    json.dump({"value": study.best_value, "trial": study.best_trial.number, "params": best},
              open(OPT_DIR / f"{args.arch}_best.json", "w"), indent=2)
    print("MEJOR", study.best_value, best)


def cv(args):
    if args.params == "best":
        p = json.load(open(OPT_DIR / f"{args.arch}_best.json"))["params"]
    elif args.params == "default":
        p = DEFAULTS[args.arch]
    else:
        p = json.loads(args.params)
    tag = args.tag or f"hq_{args.arch}"
    print(tag, p, flush=True)
    meta, X, Y = load_data(only_hq=False)
    votes = meta.total_votes.to_numpy()
    hq = votes >= HQ_MIN_VOTES
    P = np.zeros_like(Y)
    log, t0 = [], time.time()
    for f in sorted(meta.fold.unique()):
        tr = np.where((meta.fold != f) & hq)[0]                 # SOLO alta calidad
        va = np.where((meta.fold == f) & hq)[0]
        te = np.where((meta.fold == f) & ~hq)[0]                # test: baja calidad, pacientes no vistos
        flog = []
        model = fit(args.arch, p, X, Y, tr, va, log=flog)
        P[va] = predict(model, X, va, "both")
        P[te] = predict(model, X, te, "both")
        print(f"fold {f}: valid_hq KL {kl_divergence(Y[va], P[va]):.4f} | test_lq KL "
              f"{kl_divergence(Y[te], P[te]):.4f}", flush=True)
        for r in flog:
            r["fold"] = f
        log += flog
        torch.save(model.state_dict(), MODELS_DIR / f"{tag}_fold{f}.pt")
        del model; torch.cuda.empty_cache()
    pd.DataFrame(log).to_csv(RESULTS_DIR / "history" / f"{tag}.csv", index=False)
    df = meta[["eeg_id", "patient_id", "fold", "total_votes"] + TARGETS].copy()
    df["split"] = np.where(hq, "valid_hq", "test_lq")
    df[[f"pred_{t}" for t in TARGETS]] = P
    df.to_csv(OOF_DIR / f"{tag}.csv", index=False)
    res = {"experiment": tag, "arch": args.arch, "params": json.dumps(p),
           "minutes": round((time.time() - t0) / 60, 1)}
    for name, m in [("valid_hq", hq), ("valid_hq_f1-4", hq & (meta.fold.to_numpy() != 0)), ("test_lq", ~hq)]:
        for k, v in {**all_metrics(Y[m], P[m]), **balanced_metrics(Y[m], P[m])}.items():
            res[f"{name}_{k}"] = v
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
    path = RESULTS_DIR / "metrics_hq.csv"
    out = pd.DataFrame([res])
    if path.exists():
        old = pd.read_csv(path)
        out = pd.concat([old[old.experiment != tag], out])
    out.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "cv"])
    ap.add_argument("--arch", choices=list(ARCHS), required=True)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--params", default="best")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.85)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tune(args) if args.cmd == "tune" else cv(args)


if __name__ == "__main__":
    main()
