"""Configuración central del proyecto HMS.

Los datos (~30 GB) y los checkpoints viven FUERA del repositorio (y fuera de
OneDrive) para no sincronizarlos. Se pueden sobreescribir con variables de
entorno:
    HMS_DATA_DIR   -> carpeta con train.csv, train_eegs/, train_spectrograms/
    HMS_WORK_DIR   -> carpeta para caché de imágenes y checkpoints
"""
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("HMS_DATA_DIR", r"C:\Users\Luis\hms_data\raw"))
WORK_DIR = Path(os.environ.get("HMS_WORK_DIR", r"C:\Users\Luis\hms_data\work"))
CACHE_DIR = WORK_DIR / "cache"
MODELS_DIR = WORK_DIR / "models"

RESULTS_DIR = REPO_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
OOF_DIR = RESULTS_DIR / "oof"

for _d in (CACHE_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR, OOF_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
CLASSES = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]

# Señal EEG
EEG_FS = 200                 # Hz
EEG_WINDOW_S = 50            # segundos etiquetados por cada fila de train.csv
SPEC_WINDOW_S = 600          # 10 minutos de espectrograma Kaggle

# Montaje bipolar "doble banana" (4 cadenas x 4 derivaciones)
CHAINS = {
    "LL": [("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1")],
    "LP": [("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1")],
    "RP": [("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2")],
    "RR": [("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2")],
}
# Prefijos de regiones en los espectrogramas de Kaggle (mismo orden que CHAINS)
KAGGLE_REGIONS = ["LL", "LP", "RP", "RL"]

# Tamaño de cada "tile" (frecuencia x tiempo) en la imagen final
TILE_H, TILE_W = 128, 256
N_FOLDS = 5
SEED = 42
