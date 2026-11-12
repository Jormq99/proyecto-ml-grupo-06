"""Precalcula metadatos consolidados, folds, imágenes (8 tiles) y features clásicas.

Uso:
    python -m src.build_cache            # todo
    python -m src.build_cache --limit 200  # prueba rápida
"""
import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import CACHE_DIR, TILE_H, TILE_W
from .data_preparation import add_folds, consolidate_by_eeg, load_train
from .features import band_power_features, build_sample


def _work(row):
    try:
        return build_sample(row), band_power_features(row), None
    except Exception as e:  # registrar y seguir
        return None, None, f"{row.eeg_id}: {e!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    meta = add_folds(consolidate_by_eeg(load_train()))
    if args.limit:
        meta = meta.sample(args.limit, random_state=0).reset_index(drop=True)
    suffix = f"_{args.limit}" if args.limit else ""
    n = len(meta)
    print(f"{n} muestras (eeg_id únicos)")

    imgs = np.lib.format.open_memmap(CACHE_DIR / f"images{suffix}.npy", mode="w+",
                                     dtype=np.float16, shape=(n, 8, TILE_H, TILE_W))
    feats, errors = [None] * n, []
    t0 = time.time()
    from types import SimpleNamespace
    rows = [SimpleNamespace(**r) for r in meta.to_dict("records")]
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (img, ft, err) in enumerate(tqdm(ex.map(_work, rows, chunksize=16), total=n)):
            if err:
                errors.append(err)
                imgs[i] = 0
                feats[i] = {}
            else:
                imgs[i] = img
                feats[i] = ft
    imgs.flush()
    pd.DataFrame(feats).to_csv(CACHE_DIR / f"band_features{suffix}.csv", index=False)
    meta["cache_ok"] = [f != {} for f in feats]
    meta.to_csv(CACHE_DIR / f"meta{suffix}.csv", index=False)
    print(f"Listo en {time.time() - t0:.0f}s. Errores: {len(errors)}")
    for e in errors[:20]:
        print("  ", e)


if __name__ == "__main__":
    main()
