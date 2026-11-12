"""Entrenamiento de CNNs preentrenadas (timm) sobre imágenes de espectrogramas.

Ejemplos:
    python -m src.train --model efficientnet_b0.ra_in1k --folds 0 --tag effb0_f0
    python -m src.train --model efficientnet_b0.ra_in1k --no-pretrained --folds 0 --tag effb0_scratch_f0
    python -m src.train --model efficientnet_b0.ra_in1k --input kaggle --folds 0 --tag effb0_kaggle_f0
    python -m src.train --model efficientnet_b0.ra_in1k --folds all --tag effb0_cv

Genera:
    results/oof/<tag>.csv          predicciones out-of-fold (etapa final)
    results/oof/<tag>_s1.csv       predicciones out-of-fold tras la etapa 1
    results/history/<tag>.csv      curvas de entrenamiento
    results/metrics.csv            una fila por experimento (se agrega)
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F

from .config import CACHE_DIR, MODELS_DIR, OOF_DIR, RESULTS_DIR, SEED, TARGETS
from .evaluate import all_metrics

HIST_DIR = RESULTS_DIR / "history"
HIST_DIR.mkdir(exist_ok=True)
DEVICE = "cuda"
ACCUM = 1


def seed_all(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------
# Imagen y aumentos (en GPU)
# ----------------------------------------------------------------------------
def assemble(x: torch.Tensor, mode: str) -> torch.Tensor:
    """(B,8,H,W) -> (B,1,4H,W*k). Columna izquierda: Kaggle; derecha: EEG."""
    B, _, H, W = x.shape
    k = x[:, :4].reshape(B, 1, 4 * H, W)
    e = x[:, 4:].reshape(B, 1, 4 * H, W)
    if mode == "kaggle":
        return k
    if mode == "eeg":
        return e
    return torch.cat([k, e], dim=3)


def augment(x: torch.Tensor) -> torch.Tensor:
    """x: (B,8,H,W). Intercambio de hemisferios + enmascarado tiempo/frecuencia."""
    B, _, H, W = x.shape
    swap = torch.rand(B, device=x.device) < 0.5
    x = torch.where(swap[:, None, None, None], x[:, [3, 2, 1, 0, 7, 6, 5, 4]], x)
    # enmascarado de tiempo (2 franjas) y frecuencia (2 franjas) por muestra
    t = torch.arange(W, device=x.device)[None, :]
    f = torch.arange(H, device=x.device)[None, :]
    mask = torch.ones(B, 1, H, W, device=x.device, dtype=x.dtype)
    for _ in range(2):
        w = torch.randint(0, W // 8, (B, 1), device=x.device)
        s = (torch.rand(B, 1, device=x.device) * (W - w)).long()
        mask = mask * (~((t >= s) & (t < s + w)))[:, None, None, :]
        h = torch.randint(0, H // 8, (B, 1), device=x.device)
        s = (torch.rand(B, 1, device=x.device) * (H - h)).long()
        mask = mask * (~((f >= s) & (f < s + h)))[:, None, :, None]
    return x * mask


# ----------------------------------------------------------------------------
def kl_loss(logits, y):
    return F.kl_div(F.log_softmax(logits.float(), 1), y, reduction="batchmean")


@torch.no_grad()
def predict(model, X, idx, mode, bs=128):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        xb = torch.from_numpy(X[idx[i:i + bs]]).to(DEVICE).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.softmax(model(assemble(xb, mode).contiguous(memory_format=torch.channels_last)).float(), 1).cpu())
    return torch.cat(out).numpy()


def run_stage(model, X, Y, tr_idx, epochs, lr, bs, mode, log, va_idx, Yva, votes_va, stage):
    if epochs == 0 or len(tr_idx) == 0:
        return
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * math.ceil(len(tr_idx) / bs)
    warm = max(1, int(0.1 * steps)) if stage == 1 else 1

    def lr_lambda(s):  # calentamiento lineal + decaimiento coseno
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm)))) + 1e-3
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    Yt = torch.from_numpy(Y).float()
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr_idx)
        tot, n, t0 = 0.0, 0, time.time()
        for i in range(0, len(perm) - bs + 1, bs):
            b = np.sort(perm[i:i + bs])
            opt.zero_grad(set_to_none=True)
            # acumulación de gradiente: micro-lotes de bs/ACCUM, mismo lote efectivo
            for mb in np.array_split(b, ACCUM):
                xb = torch.from_numpy(X[mb]).to(DEVICE, non_blocking=True).float()
                yb = Yt[mb].to(DEVICE)
                xb = assemble(augment(xb), mode).contiguous(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(xb)
                loss = kl_loss(logits, yb)
                (loss * len(mb) / len(b)).backward()
                tot += loss.item() * len(mb); n += len(mb)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
        p = predict(model, X, va_idx, mode)
        m = all_metrics(Yva, p, votes_va)
        rec = {"stage": stage, "epoch": ep + 1, "train_kl": tot / n, "valid_kl": m["kl"],
               "valid_kl_hq": m["kl_hq"], "valid_acc": m["accuracy"], "sec": time.time() - t0}
        log.append(rec)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0.ra_in1k")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--input", default="both", choices=["both", "kaggle", "eeg"])
    ap.add_argument("--folds", default="0", help="'all' o lista: 0,1")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--stage2-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cache-suffix", default="")
    ap.add_argument("--save", action="store_true", help="guardar pesos por fold")
    ap.add_argument("--accum", type=int, default=1, help="micro-lotes por paso (ahorra VRAM)")
    args = ap.parse_args()
    global ACCUM
    ACCUM = args.accum
    seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    # En Windows, si se excede la VRAM el driver usa RAM compartida y el entrenamiento se vuelve
    # ~50x más lento sin error. Limitar la fracción hace que falle con OOM explícito.
    torch.cuda.set_per_process_memory_fraction(0.85)

    meta = pd.read_csv(CACHE_DIR / f"meta{args.cache_suffix}.csv")
    X = np.load(CACHE_DIR / f"images{args.cache_suffix}.npy", mmap_mode="r")
    print("Cargando imágenes en RAM...", flush=True)
    X = np.ascontiguousarray(X)
    Y = meta[TARGETS].to_numpy(np.float32)
    ok = meta["cache_ok"].to_numpy()
    votes = meta["total_votes"].to_numpy()
    folds = sorted(meta.fold.unique()) if args.folds == "all" else [int(f) for f in args.folds.split(",")]

    oof = {1: [], 2: []}
    log, t_start = [], time.time()
    for f in folds:
        print(f"===== fold {f} =====", flush=True)
        tr = np.where((meta.fold != f) & ok)[0]
        va = np.where((meta.fold == f) & ok)[0]
        model = timm.create_model(args.model, pretrained=not args.no_pretrained,
                                  num_classes=len(TARGETS), in_chans=1,
                                  drop_rate=0.2).to(DEVICE).to(memory_format=torch.channels_last)
        flog = []
        run_stage(model, X, Y, tr, args.epochs, args.lr, args.bs, args.input, flog,
                  va, Y[va], votes[va], stage=1)
        oof[1].append((va, predict(model, X, va, args.input)))
        tr_hq = tr[votes[tr] >= 10]
        run_stage(model, X, Y, tr_hq, args.stage2_epochs, args.lr * 0.1, args.bs, args.input,
                  flog, va, Y[va], votes[va], stage=2)
        oof[2].append((va, predict(model, X, va, args.input)))
        for r in flog:
            r["fold"] = f
        log += flog
        if args.save:
            torch.save(model.state_dict(), MODELS_DIR / f"{args.tag}_fold{f}.pt")
        del model; torch.cuda.empty_cache()

    pd.DataFrame(log).to_csv(HIST_DIR / f"{args.tag}.csv", index=False)
    rows = []
    for stage, name in ((1, f"{args.tag}_s1"), (2, args.tag)):
        if stage == 2 and args.stage2_epochs == 0:
            continue
        idx = np.concatenate([i for i, _ in oof[stage]])
        P = np.concatenate([p for _, p in oof[stage]])
        df = meta.iloc[idx][["eeg_id", "patient_id", "fold", "total_votes"] + TARGETS].copy()
        df[[f"pred_{t}" for t in TARGETS]] = P
        df.to_csv(OOF_DIR / f"{name}.csv", index=False)
        m = all_metrics(Y[idx], P, votes[idx])
        rows.append({"experiment": name, "model": args.model, "pretrained": not args.no_pretrained,
                     "input": args.input, "folds": args.folds, "stage": stage,
                     "epochs": args.epochs + (args.stage2_epochs if stage == 2 else 0),
                     "n_valid": len(idx), "train_min": round((time.time() - t_start) / 60, 1), **m})
        print(name, json.dumps({k: round(v, 4) for k, v in m.items()}), flush=True)
    mpath = RESULTS_DIR / "metrics.csv"
    res = pd.DataFrame(rows)
    if mpath.exists():
        old = pd.read_csv(mpath)
        res = pd.concat([old[~old.experiment.isin(res.experiment)], res])
    res.to_csv(mpath, index=False)


if __name__ == "__main__":
    main()
