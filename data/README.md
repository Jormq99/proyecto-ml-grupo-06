# Datos del proyecto HMS

## Fuente y licencia

Dataset **HMS - Harmful Brain Activity Classification**, publicado para la competencia de Kaggle: <https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data>.
La licencia indicada por Kaggle es CC BY-NC 4.0. Se requiere iniciar sesión y aceptar las reglas de la competencia antes de descargar.

## Descarga y ubicacion

Descargue el archivo de la competencia HMS desde Kaggle y extraiga el contenido directamente dentro de `data/`. Para los notebooks 01 y 02 se requieren:

```text
data/train.csv
data/train_spectrograms/<spectrogram_id>.parquet
```

`train.csv` contiene identificadores de paciente, EEG, segmentos y seis conteos de votos (`seizure_vote`, `lpd_vote`, `gpd_vote`, `lrda_vote`, `grda_vote`, `other_vote`). Los espectrogramas están repartidos en archivos Parquet individuales; el nombre de cada archivo es su `spectrogram_id`. Los EEG crudos pueden conservarse para experimentos posteriores, pero no son necesarios para los notebooks 01 y 02.

El notebook 01 busca `data/train.csv` (y conserva rutas alternativas bajo `data/raw/`). El notebook 02 busca `data/train_spectrograms/`. Los notebooks no leen `test.csv`, `sample_submission.csv`, `test_eegs` ni `test_spectrograms`.

## Protocolo y reproducibilidad

1. Ejecute `notebooks/01_exploracion.ipynb` primero. Reserva pacientes completos antes del EDA y escribe un manifiesto local en `data/processed/`.
2. Ejecute `notebooks/02_preprocesamiento.ipynb`. Consume el manifiesto, conserva los seis votos como objetivos probabilisticos y crea folds agrupados por paciente.
3. Descargue las dependencias con `pip install -r requirements.txt`; `pyarrow` permite leer los archivos Parquet.
4. No suba el ZIP, los datos crudos ni los artefactos de `data/processed/` al repositorio. `.gitignore` excluye el ZIP y las carpetas de entrenamiento, test y procesamiento para evitar agregar archivos grandes o datos de evaluación.

La competencia contiene segmentos EEG y etiquetas con desacuerdo entre especialistas. Por ello se conserva la distribucion de votos, se agrupa la validacion por paciente y se reserva el holdout interno para una unica evaluacion final, una vez congelado el pipeline.
