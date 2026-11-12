# Datos del proyecto HMS

El repositorio tiene dos acercamientos independientes (ver `docs/crispdm.md`). Las secciones siguientes describen los datos de la **Iteración 1** (`notebooks/01`–`04`, `src/hms_cnn.py`). La **Iteración 2** (`notebooks/iteracion_2/`, resto de `src/`) se describe al final.

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

---

## Iteración 2: datos fuera del repositorio

### Fuente

Competencia de Kaggle [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
(Harvard Medical School / Sunstella Foundation, 2024). Uso sujeto a las reglas de la competencia
(es necesario aceptarlas en Kaggle antes de descargar).

| Atributo | Valor |
|----------|-------|
| Filas etiquetadas (`train.csv`) | 106 800 ventanas de 50 s |
| EEG únicos (`eeg_id`) | 17 089 |
| Espectrogramas únicos | 11 138 |
| Pacientes | 1 950 |
| Señal | 19 electrodos 10-20 + ECG, 200 Hz |
| Espectrogramas | 4 regiones (LL, LP, RP, RL) × 100 frecuencias (0.59–19.92 Hz), resolución 2 s |
| Objetivo | Conteo de votos de expertos en 6 clases: Seizure, LPD, GPD, LRDA, GRDA, Other |
| Métrica oficial | Divergencia KL entre la distribución de votos y la predicha |
| Tamaño | ~20 GB comprimido, ~30 GB descomprimido |

### Descarga

Los datos **no** se guardan dentro del repositorio (ni en OneDrive). La ruta se configura en
`src/config.py` o con variables de entorno:

```bash
export HMS_DATA_DIR=/ruta/a/hms/raw      # train.csv, train_eegs/, train_spectrograms/
export HMS_WORK_DIR=/ruta/a/hms/work     # caché de imágenes (~9 GB) y checkpoints
```

> El cliente `kaggle competitions download` (v1.7) carga el zip completo en memoria (>30 GB de RAM).
> Se recomienda descargar con `curl`, que escribe en disco de forma incremental y permite reanudar:

```bash
cd $HMS_DATA_DIR
U=$(python -c "import json,os;d=json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')));print(d['username']+':'+d['key'])")
curl -L -C - -u "$U" -o hms.zip \
  https://www.kaggle.com/api/v1/competitions/data/download-all/hms-harmful-brain-activity-classification
unzip -q hms.zip && rm hms.zip
```

### Estructura descargada

```
raw/
├── train.csv                 # metadatos y votos por ventana
├── test.csv, sample_submission.csv
├── train_eegs/<eeg_id>.parquet               # EEG crudo (filas = muestras a 200 Hz)
├── train_spectrograms/<spectrogram_id>.parquet  # columna time + 400 columnas REGION_FREQ
├── test_eegs/, test_spectrograms/
└── example_figures/          # PDFs con ejemplos de cada patrón
```

### Columnas de `train.csv`

| Columna | Descripción |
|---------|-------------|
| `eeg_id`, `eeg_sub_id` | Registro EEG y número de sub-ventana etiquetada |
| `eeg_label_offset_seconds` | Inicio de la ventana de 50 s dentro del EEG |
| `spectrogram_id`, `spectrogram_sub_id`, `spectrogram_label_offset_seconds` | Igual para el espectrograma de 10 min |
| `label_id` | Identificador único de la fila |
| `patient_id` | Paciente (se usa para agrupar los folds) |
| `expert_consensus` | Clase con más votos |
| `seizure_vote` … `other_vote` | Conteo de votos por clase |

### Notas para el modelado

- **Sin nulos** en el CSV; sí hay NaN y artefactos dentro de los parquet de EEG/espectrogramas.
- **Desbalance**: por EEG único, *Seizure* y *Other* son las clases más frecuentes y *LRDA* la menos frecuente.
- **Votos**: 1–28 por ventana; las de <10 votos tienen otra distribución de clases (ver EDA).
- **Fuga**: sub-ventanas del mismo EEG son casi duplicadas y los pacientes son casi monoclase →
  una muestra por `eeg_id` y validación agrupada por `patient_id`.

### Preprocesamiento reproducible

```bash
python -m src.build_cache        # genera meta.csv, images.npy y band_features.csv en HMS_WORK_DIR/cache
```

Ver `notebooks/iteracion_2/01_exploracion.ipynb` y `notebooks/iteracion_2/02_preprocesamiento.ipynb`.
