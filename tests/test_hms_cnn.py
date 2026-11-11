import numpy as np
import pandas as pd
import torch

from src.hms_cnn import (
    CLASS_NAMES,
    VOTE_COLUMNS,
    SpectrogramCNN,
    _resize_and_normalize,
    load_development_targets,
    prepare_image_cache,
    sample_rows,
)


def test_spectrogram_resize_returns_normalized_single_channel_image():
    image = _resize_and_normalize(np.arange(20, dtype=np.float32).reshape(4, 5), 32)

    assert image.shape == (1, 32, 32)
    assert np.isfinite(image).all()
    assert image.min() >= 0
    assert image.max() <= 1


def test_cnn_returns_one_score_per_class():
    model = SpectrogramCNN()

    scores = model(torch.zeros(2, 1, 64, 64))

    assert scores.shape == (2, len(CLASS_NAMES))


def test_sample_rows_is_reproducible_without_rebalancing():
    rows = []
    for class_index in range(len(CLASS_NAMES)):
        count = 50 if class_index == 0 else 10
        for _ in range(count):
            row = {column: 0.0 for column in VOTE_COLUMNS}
            row[VOTE_COLUMNS[class_index]] = 1.0
            rows.append(row)
    frame = pd.DataFrame(rows)

    first = sample_rows(frame, limit=30, random_state=4)
    second = sample_rows(frame, limit=30, random_state=4)

    assert len(first) == 30
    assert first.equals(second)
    assert (first[VOTE_COLUMNS].to_numpy().argmax(axis=1) == 0).sum() > 15


def test_sample_rows_none_keeps_the_complete_dataset():
    frame = pd.DataFrame({"value": range(20)})

    sampled = sample_rows(frame, limit=None, random_state=4)

    assert len(sampled) == len(frame)
    assert sampled["value"].tolist() == frame["value"].tolist()


def test_image_cache_materializes_windows_and_reuses_cache(tmp_path):
    parquet_directory = tmp_path / "spectrograms"
    cache_directory = tmp_path / "cache"
    parquet_directory.mkdir()
    times = np.arange(0, 60, 2)
    frame = pd.DataFrame({"time": times, "frequency_a": times + 1, "frequency_b": times + 2})
    frame.to_parquet(parquet_directory / "123.parquet")
    rows = pd.DataFrame(
        {
            "row_id": [0, 1],
            "spectrogram_id": [123, 123],
            "spectrogram_label_offset_seconds": [0.0, 6.0],
            **{
                column: [1.0 if index == 0 else 0.0, 1.0 if index == 1 else 0.0]
                for index, column in enumerate(VOTE_COLUMNS)
            },
        }
    )

    first = prepare_image_cache(rows, parquet_directory, cache_directory, image_size=16)
    second = prepare_image_cache(rows, parquet_directory, cache_directory, image_size=16)

    assert first.shape == (2, 1, 16, 16)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_prepared_fold_keeps_patients_separate_and_within_development():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    training, validation = load_development_targets(root, fold_id=1)

    assert set(training["patient_id"]).isdisjoint(validation["patient_id"])
    assert set(training["eeg_id"]).isdisjoint(validation["eeg_id"])
    assert set(training["spectrogram_id"]).isdisjoint(validation["spectrogram_id"])
    assert validation["fold_id"].eq(1).all()
    assert training["fold_id"].ne(1).all()
