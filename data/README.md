# Datos del proyecto HMS

## Fuente y licencia

Dataset **HMS - Harmful Brain Activity Classification**, publicado para la competencia de Kaggle: <https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data>.
La licencia indicada por Kaggle es CC BY-NC 4.0. Se requiere iniciar sesión y aceptar las reglas de la competencia antes de descargar.

## Descarga y ubicacion

Descargue el archivo de la competencia HMS desde Kaggle y extraiga el contenido directamente dentro de `data/`. Para los notebooks 01, 02, 03 y 04 se requieren:

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
5. Para entrenar la CNN use todo el conjunto de desarrollo sin tocar el holdout, ejecute `python -m src.hms_cnn` desde la raíz del proyecto. El valor predeterminado usa el fold 1 para validación, los otros cuatro folds para entrenamiento y tres épocas. Se crea una caché local de imágenes bajo `data/processed/`; debe haber espacio libre suficiente (aproximadamente 1.4 GB para imágenes de 64 × 64).
6. Abra `notebooks/03_modelado.ipynb` y ejecute `Run All` para ver las curvas de entrenamiento, distribución de clases y matriz de confusión. Luego ejecute `notebooks/04_evaluacion.ipynb` para comparar la CNN con la referencia mayoritaria y ver precisión, recall y F1 por clase. Las figuras se guardan en `results/figures/`; los CSV y modelos no se suben a Git.
7. El fold 0 fue utilizado en una prueba piloto y el fold 1 en la primera corrida completa. No los use para seguir eligiendo modelos; para comparaciones posteriores use folds 2–4 con el mismo protocolo, y mantenga el holdout de pacientes reservado para una evaluación final.
8. En VS Code, abra `repo_machine_learning` como carpeta del proyecto (no solo la carpeta padre `repos`). Para los notebooks seleccione `Python (repo_machine_learning .venv)`. La configuración de `.vscode/settings.json` apunta al intérprete `.venv\Scripts\python.exe`; si ese entorno todavía no existe, créelo e instale `requirements.txt` antes de ejecutar el notebook.

La competencia contiene segmentos EEG y etiquetas con desacuerdo entre especialistas. Por ello se conserva la distribucion de votos, se agrupa la validacion por paciente y se reserva el holdout interno para una unica evaluacion final, una vez congelado el pipeline.
