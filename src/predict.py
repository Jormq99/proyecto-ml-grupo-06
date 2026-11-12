"""Inferencia sobre test.csv con los modelos guardados por fold (promedio de folds y de modelos).

    python -m src.predict --models cv_effb0:efficientnet_b0.ra_in1k cv_effb2:efficientnet_b2.ra_in1k
Genera results/submission.csv con el formato de Kaggle.
"""
import argparse
from types import SimpleNamespace

import numpy as np
import pandas as pd
import timm
import torch

from .config import DATA_DIR, MODELS_DIR, RESULTS_DIR, TARGETS
from . import features as F
from .train import assemble


def build_test_images(test: pd.DataFrame) -> np.ndarray:
    imgs = []
    for r in test.to_dict("records"):
        row = SimpleNamespace(eeg_id=r["eeg_id"], spectrogram_id=r["spectrogram_id"], eeg_offset=0, spec_offset=0)
        spec = pd.read_parquet(DATA_DIR / "test_spectrograms" / f"{row.spectrogram_id}.parquet")
        eeg = pd.read_parquet(DATA_DIR / "test_eegs" / f"{row.eeg_id}.parquet")
        imgs.append(np.concatenate([F.kaggle_spec_tiles(F.kaggle_spec_window(spec, 0)),
                                    F.eeg_spec_tiles(F.eeg_window(eeg, 0))]))
    return np.stack(imgs).astype(np.float32)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="tag:arquitectura_timm")
    ap.add_argument("--input", default="both")
    args = ap.parse_args()
    test = pd.read_csv(DATA_DIR / "test.csv")
    x = torch.from_numpy(build_test_images(test)).cuda()
    preds = []
    for spec in args.models:
        tag, arch = spec.split(":")
        for ckpt in sorted(MODELS_DIR.glob(f"{tag}_fold*.pt")):
            m = timm.create_model(arch, pretrained=False, num_classes=6, in_chans=1).cuda().eval()
            m.load_state_dict(torch.load(ckpt, map_location="cuda"))
            preds.append(torch.softmax(m(assemble(x, args.input)).float(), 1).cpu().numpy())
    P = np.mean(preds, 0)
    sub = pd.DataFrame(P, columns=TARGETS)
    sub.insert(0, "eeg_id", test.eeg_id.values)
    sub.to_csv(RESULTS_DIR / "submission.csv", index=False)
    print(f"{len(preds)} modelos promediados\n", sub)


if __name__ == "__main__":
    main()
