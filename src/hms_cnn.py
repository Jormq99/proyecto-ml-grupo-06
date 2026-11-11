"""Entrenamiento inicial de una CNN para los espectrogramas HMS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

VOTE_COLUMNS = [
    "target_seizure",
    "target_lpd",
    "target_gpd",
    "target_lrda",
    "target_grda",
    "target_other",
]
CLASS_NAMES = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]


class SpectrogramCNN(nn.Module):
    """CNN pequeña para establecer una primera línea base reproducible."""

    def __init__(self, number_of_classes: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, number_of_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs).flatten(start_dim=1)
        return self.classifier(features)


def load_development_targets(project_root: Path, fold_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga solo las filas del conjunto de desarrollo preparadas por el notebook 02."""
    targets_path = project_root / "data" / "processed" / "development_targets.csv"
    if not targets_path.is_file():
        raise FileNotFoundError(
            f"No existe {targets_path}. Ejecute primero los notebooks 01 y 02."
        )

    targets = pd.read_csv(targets_path)
    required = {
        "eeg_id",
        "row_id",
        "patient_id",
        "spectrogram_id",
        "spectrogram_label_offset_seconds",
        "fold_id",
        *VOTE_COLUMNS,
    }
    missing = required.difference(targets.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {targets_path.name}: {sorted(missing)}")

    required_values = [
        "eeg_id",
        "patient_id",
        "spectrogram_id",
        "spectrogram_label_offset_seconds",
        "fold_id",
        *VOTE_COLUMNS,
    ]
    missing_rows = targets[required_values].isna().any(axis=1)
    if missing_rows.any():
        raise ValueError(f"{int(missing_rows.sum())} filas tienen valores faltantes en campos requeridos.")
    targets["fold_id"] = targets["fold_id"].astype(int)
    probabilities = targets[VOTE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Los objetivos de votos deben ser números finitos no negativos.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Los seis votos de cada fila deben sumar 1.")

    validation = targets.loc[targets["fold_id"] == fold_id].copy()
    training = targets.loc[targets["fold_id"] != fold_id].copy()
    if training.empty or validation.empty:
        available = sorted(targets["fold_id"].unique().tolist())
        raise ValueError(f"El fold {fold_id} no existe o no tiene filas. Folds: {available}")
    for column, entity in (
        ("patient_id", "pacientes"),
        ("eeg_id", "registros EEG"),
        ("spectrogram_id", "espectrogramas"),
    ):
        overlap = set(training[column]).intersection(validation[column])
        if overlap:
            raise ValueError(
                f"Hay {len(overlap)} {entity} compartidos entre train y validación."
            )
    return training, validation


def sample_rows(frame: pd.DataFrame, limit: int | None, random_state: int) -> pd.DataFrame:
    """Reduce un experimento opcionalmente; None conserva todas las filas."""
    if limit is None:
        return frame.reset_index(drop=True)
    if limit <= 0:
        raise ValueError("El límite de muestras debe ser mayor que cero.")
    if len(frame) <= limit:
        return frame.reset_index(drop=True)
    return frame.sample(n=limit, random_state=random_state).reset_index(drop=True)


def _resize_and_normalize(matrix: np.ndarray, image_size: int) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.log1p(np.clip(values, a_min=0.0, a_max=None))
    lower, upper = np.percentile(values, [1, 99])
    if upper > lower:
        values = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    else:
        values = np.zeros_like(values)

    source_time = np.linspace(0.0, 1.0, values.shape[1])
    target_time = np.linspace(0.0, 1.0, image_size)
    resized_time = np.stack(
        [np.interp(target_time, source_time, row) for row in values],
        axis=0,
    )
    source_frequency = np.linspace(0.0, 1.0, values.shape[0])
    target_frequency = np.linspace(0.0, 1.0, image_size)
    resized = np.stack(
        [
            np.interp(target_frequency, source_frequency, resized_time[:, column])
            for column in range(image_size)
        ],
        axis=1,
    )
    return resized[np.newaxis, ...].astype(np.float32)


def load_spectrogram_images(
    rows: pd.DataFrame,
    parquet_directory: Path,
    image_size: int,
    window_seconds: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Lee cada Parquet una vez y extrae la ventana correspondiente a cada etiqueta."""
    if not parquet_directory.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de espectrogramas: {parquet_directory}")

    images: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for spectrogram_id, group in rows.groupby("spectrogram_id", sort=False):
        parquet_path = parquet_directory / f"{int(spectrogram_id)}.parquet"
        if not parquet_path.is_file():
            raise FileNotFoundError(f"No existe el espectrograma: {parquet_path}")
        spectrogram = pd.read_parquet(parquet_path)
        if "time" not in spectrogram.columns:
            raise ValueError(f"{parquet_path.name} no contiene la columna 'time'.")

        times = pd.to_numeric(spectrogram["time"], errors="coerce").to_numpy(dtype=np.float64)
        frequency_columns = [column for column in spectrogram.columns if column != "time"]
        if not frequency_columns:
            raise ValueError(f"{parquet_path.name} no contiene columnas de frecuencia.")
        values = spectrogram[frequency_columns].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float32
        )

        for row in group.itertuples(index=False):
            start = float(row.spectrogram_label_offset_seconds)
            left = int(np.searchsorted(times, start, side="left"))
            right = int(np.searchsorted(times, start + window_seconds, side="left"))
            if right - left < 2:
                raise ValueError(
                    f"Ventana vacía para spectrogram_id={int(spectrogram_id)}, offset={start}."
                )
            window = values[left:right].T
            images.append(_resize_and_normalize(window, image_size))
            targets.append(np.asarray([getattr(row, column) for column in VOTE_COLUMNS], dtype=np.float32))

    return np.stack(images), np.stack(targets)


def prepare_image_cache(
    rows: pd.DataFrame,
    parquet_directory: Path,
    cache_directory: Path,
    image_size: int,
    window_seconds: int = 50,
) -> np.memmap:
    """Prepara las imágenes una vez en disco para procesar el conjunto completo sin agotar RAM."""
    cache_directory.mkdir(parents=True, exist_ok=True)
    image_path = cache_directory / f"hms_spectrogram_images_{image_size}.npy"
    metadata_path = cache_directory / f"hms_spectrogram_images_{image_size}.json"
    source_columns = [
        "row_id",
        "spectrogram_id",
        "spectrogram_label_offset_seconds",
        *VOTE_COLUMNS,
    ]
    signature = hashlib.sha256(
        pd.util.hash_pandas_object(rows[source_columns], index=False).to_numpy().tobytes()
    ).hexdigest()
    expected_metadata = {
        "signature": signature,
        "rows": len(rows),
        "image_size": image_size,
        "window_seconds": window_seconds,
    }

    if image_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata == expected_metadata:
                cached = np.load(image_path, mmap_mode="c")
                if cached.shape == (len(rows), 1, image_size, image_size):
                    print(f"Usando caché local de imágenes: {image_path}")
                    return cached
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    if not parquet_directory.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de espectrogramas: {parquet_directory}")
    rows = rows.reset_index(drop=True)
    image_cache = np.lib.format.open_memmap(
        image_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), 1, image_size, image_size),
    )
    rows_per_spectrogram = rows.groupby("spectrogram_id", sort=False).indices
    print(
        f"Preparando caché de {len(rows):,} espectrogramas/ventanas "
        f"({image_size}x{image_size}) en disco..."
    )
    for group_number, (spectrogram_id, positions) in enumerate(rows_per_spectrogram.items(), start=1):
        parquet_path = parquet_directory / f"{int(spectrogram_id)}.parquet"
        if not parquet_path.is_file():
            raise FileNotFoundError(f"No existe el espectrograma: {parquet_path}")
        spectrogram = pd.read_parquet(parquet_path)
        if "time" not in spectrogram.columns:
            raise ValueError(f"{parquet_path.name} no contiene la columna 'time'.")

        times = pd.to_numeric(spectrogram["time"], errors="coerce").to_numpy(dtype=np.float64)
        frequency_columns = [column for column in spectrogram.columns if column != "time"]
        if not frequency_columns:
            raise ValueError(f"{parquet_path.name} no contiene columnas de frecuencia.")
        values = spectrogram[frequency_columns].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float32
        )
        starts = rows.iloc[positions]["spectrogram_label_offset_seconds"].to_numpy(dtype=np.float64)
        for position, start in zip(positions, starts):
            left = int(np.searchsorted(times, start, side="left"))
            right = int(np.searchsorted(times, start + window_seconds, side="left"))
            if right - left < 2:
                raise ValueError(
                    f"Ventana vacía para spectrogram_id={int(spectrogram_id)}, offset={start}."
                )
            image_cache[position] = _resize_and_normalize(values[left:right].T, image_size)
        if group_number % 1000 == 0:
            print(f"Parquets procesados: {group_number:,}/{len(rows_per_spectrogram):,}")

    image_cache.flush()
    del image_cache
    metadata_path.write_text(json.dumps(expected_metadata, indent=2), encoding="utf-8")
    print(f"Caché guardada: {image_path}")
    return np.load(image_path, mmap_mode="c")


class SpectrogramDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, images: np.ndarray, targets: np.ndarray) -> None:
        self.images = images
        self.targets = torch.from_numpy(np.asarray(targets, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.images[index]), self.targets[index]


def _make_loader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    random_state: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(random_state)
    if shuffle:
        sampler = SubsetRandomSampler(indices.tolist(), generator=generator)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, sampler=indices.tolist())


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    total_loss = 0.0
    number_of_rows = 0
    with torch.no_grad():
        for images, soft_targets in loader:
            logits = model(images.to(device))
            batch_probabilities = torch.softmax(logits, dim=1)
            log_probabilities = torch.log_softmax(logits, dim=1)
            loss = -(soft_targets.to(device) * log_probabilities).sum(dim=1).sum()
            total_loss += float(loss.item())
            number_of_rows += len(images)
            probabilities.append(batch_probabilities.cpu().numpy())
            targets.append(soft_targets.numpy())
    return np.concatenate(probabilities), np.concatenate(targets), total_loss / number_of_rows


def _plot_training_history(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [int(item["epoch"]) for item in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [item["train_loss"] for item in history], marker="o", label="Entrenamiento")
    axes[0].plot(epochs, [item["validation_loss"] for item in history], marker="o", label="Validación")
    axes[0].set(title="Pérdida por época", xlabel="Época", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot(epochs, [item["validation_macro_f1"] for item in history], marker="o", label="Macro F1")
    axes[1].plot(epochs, [item["validation_accuracy"] for item in history], marker="o", label="Accuracy")
    axes[1].set(title="Métricas de validación por época", xlabel="Época", ylabel="Puntuación")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(figure)


def _plot_class_distribution(
    train_counts: np.ndarray,
    validation_counts: np.ndarray,
    output_path: Path,
) -> None:
    distribution = pd.DataFrame(
        {
            "Clase": CLASS_NAMES * 2,
            "Ejemplos": np.concatenate([train_counts, validation_counts]),
            "Conjunto": ["Entrenamiento"] * len(CLASS_NAMES) + ["Validación"] * len(CLASS_NAMES),
        }
    )
    figure, axis = plt.subplots(figsize=(10, 4))
    sns.barplot(data=distribution, x="Clase", y="Ejemplos", hue="Conjunto", ax=axis)
    axis.set(title="Distribución natural de clases (etiqueta = voto mayoritario)", xlabel="", ylabel="Filas")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(figure)


def _plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axis,
    )
    axis.set(title="Matriz de confusión — validación", xlabel="Predicción", ylabel="Etiqueta real")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(figure)


def run_experiment(
    project_root: Path,
    fold_id: int = 1,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    epochs: int = 3,
    batch_size: int = 64,
    image_size: int = 64,
    random_state: int = 42,
    save_model: bool = True,
) -> dict[str, Any]:
    """Entrena y evalúa usando todas las filas del fold, salvo límites explícitos."""
    if epochs <= 0 or batch_size <= 0 or image_size < 16:
        raise ValueError("epochs/batch_size deben ser positivos e image_size debe ser al menos 16.")
    project_root = Path(project_root).resolve()
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    full_training, full_validation = load_development_targets(project_root, fold_id)
    full_training_labels = full_training[VOTE_COLUMNS].to_numpy().argmax(axis=1)
    majority_class = int(np.bincount(full_training_labels, minlength=len(CLASS_NAMES)).argmax())
    training = sample_rows(full_training, max_train_samples, random_state)
    validation = sample_rows(full_validation, max_validation_samples, random_state + 1)
    parquet_directory = project_root / "data" / "train_spectrograms"
    cache_rows = pd.concat([training, validation]).sort_values("row_id").reset_index(drop=True)
    training_indices = np.flatnonzero(cache_rows["fold_id"].to_numpy() != fold_id)
    validation_indices = np.flatnonzero(cache_rows["fold_id"].to_numpy() == fold_id)
    image_cache = prepare_image_cache(
        cache_rows,
        parquet_directory,
        project_root / "data" / "processed",
        image_size,
    )
    targets = cache_rows[VOTE_COLUMNS].to_numpy(dtype=np.float32)
    dataset = SpectrogramDataset(image_cache, targets)
    train_labels = targets[training_indices].argmax(axis=1)
    validation_labels = targets[validation_indices].argmax(axis=1)
    train_counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    validation_counts = np.bincount(validation_labels, minlength=len(CLASS_NAMES))
    print(
        f"Fold de validación: {fold_id}; {len(train_labels):,} filas para entrenar y "
        f"{len(validation_labels):,} para validar. Pacientes sin solapamiento."
    )
    print("Clases en entrenamiento:", dict(zip(CLASS_NAMES, train_counts.tolist())))
    print("Clases en validación:", dict(zip(CLASS_NAMES, validation_counts.tolist())))
    train_loader = _make_loader(dataset, training_indices, batch_size, True, random_state)
    validation_loader = _make_loader(dataset, validation_indices, batch_size, False, random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrogramCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, soft_targets in train_loader:
            images = images.to(device)
            soft_targets = soft_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            log_probabilities = torch.log_softmax(model(images), dim=1)
            loss = -(soft_targets * log_probabilities).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        mean_loss = total_loss / len(train_labels)
        probabilities, observed_targets, validation_loss = _evaluate(model, validation_loader, device)
        epoch_predictions = probabilities.argmax(axis=1)
        epoch_labels = observed_targets.argmax(axis=1)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": float(accuracy_score(epoch_labels, epoch_predictions)),
            "validation_macro_f1": float(
                f1_score(epoch_labels, epoch_predictions, labels=list(range(len(CLASS_NAMES))), average="macro", zero_division=0)
            ),
        }
        history.append(epoch_metrics)
        print(
            f"Época {epoch}/{epochs} - pérdida train: {mean_loss:.4f}; "
            f"pérdida validación: {validation_loss:.4f}; "
            f"Macro F1 validación: {epoch_metrics['validation_macro_f1']:.4f}"
        )

    predicted_probabilities, observed_targets, _ = _evaluate(model, validation_loader, device)
    predictions = predicted_probabilities.argmax(axis=1)
    observed_labels = observed_targets.argmax(axis=1)
    majority_predictions = np.full_like(observed_labels, majority_class)
    labels = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(observed_labels, predictions, labels=labels)

    metrics: dict[str, Any] = {
        "fold_id": fold_id,
        "architecture": "SpectrogramCNN",
        "sampling_strategy": "full_development" if max_train_samples is None and max_validation_samples is None else "random_subset",
        "random_state": random_state,
        "baseline_reference": "majority class from all training folds before sampling",
        "epochs": epochs,
        "image_size": image_size,
        "train_samples": len(train_labels),
        "validation_samples": len(observed_labels),
        "train_class_counts": dict(zip(CLASS_NAMES, train_counts.tolist())),
        "validation_class_counts": dict(zip(CLASS_NAMES, validation_counts.tolist())),
        "training_history": history,
        "device": str(device),
        "cnn_accuracy": float(accuracy_score(observed_labels, predictions)),
        "cnn_macro_f1": float(f1_score(observed_labels, predictions, labels=labels, average="macro", zero_division=0)),
        "majority_class": CLASS_NAMES[majority_class],
        "majority_accuracy": float(accuracy_score(observed_labels, majority_predictions)),
        "majority_macro_f1": float(
            f1_score(observed_labels, majority_predictions, labels=labels, average="macro", zero_division=0)
        ),
        "class_precision": {
            name: float(value)
            for name, value in zip(
                CLASS_NAMES,
                precision_score(observed_labels, predictions, labels=labels, average=None, zero_division=0),
            )
        },
        "class_f1": {
            name: float(value)
            for name, value in zip(
                CLASS_NAMES,
                f1_score(observed_labels, predictions, labels=labels, average=None, zero_division=0),
            )
        },
        "class_recall": {
            name: float(value)
            for name, value in zip(
                CLASS_NAMES,
                recall_score(observed_labels, predictions, labels=labels, average=None, zero_division=0),
            )
        },
        "confusion_matrix": matrix.tolist(),
    }
    print("Resultados de validación:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    results_path = project_root / "results" / "hms_cnn_experiments.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    nested_metrics = (
        "class_precision",
        "class_f1",
        "class_recall",
        "confusion_matrix",
        "train_class_counts",
        "validation_class_counts",
        "training_history",
    )
    flat_metrics = {key: value for key, value in metrics.items() if key not in nested_metrics}
    for key in nested_metrics:
        flat_metrics[key] = json.dumps(metrics[key], ensure_ascii=False)
    new_result = pd.DataFrame([flat_metrics])
    if results_path.exists():
        previous_results = pd.read_csv(results_path)
        new_result = pd.concat([previous_results, new_result], ignore_index=True, sort=False)
    new_result.to_csv(results_path, index=False)
    print(f"Métricas guardadas en {results_path.relative_to(project_root)}")
    figure_directory = project_root / "results" / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    _plot_training_history(history, figure_directory / f"hms_cnn_training_fold_{fold_id}.png")
    _plot_class_distribution(
        train_counts,
        validation_counts,
        figure_directory / f"hms_cnn_class_distribution_fold_{fold_id}.png",
    )
    confusion_path = project_root / "results" / "hms_cnn_confusion_matrix.csv"
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(confusion_path)
    _plot_confusion_matrix(matrix, figure_directory / f"hms_cnn_confusion_matrix_fold_{fold_id}.png")
    print(f"Matriz de confusión guardada en {confusion_path.relative_to(project_root)}")

    if save_model:
        model_path = project_root / "models" / "hms_cnn_baseline.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "image_size": image_size,
                "fold_id": fold_id,
                "metrics": metrics,
            },
            model_path,
        )
        print(f"Modelo guardado en {model_path.relative_to(project_root)}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena una CNN inicial con datos HMS.")
    parser.add_argument("--fold-id", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-save-model", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    run_experiment(
        project_root=project_root,
        fold_id=args.fold_id,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        random_state=args.random_state,
        save_model=not args.no_save_model,
    )


if __name__ == "__main__":
    main()
