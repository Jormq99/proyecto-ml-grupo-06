# HMS Harmful Brain Activity Classification

## Integrantes
* **Jordani Mejía Quirós**  
  *Maestría en Computación, Instituto Tecnológico de Costa Rica*  
  **Carné:** 2018105537

* **Luis Fernando Solano Coto**  
  *Maestría en Computación, Instituto Tecnológico de Costa Rica*  
  **Carné:** 2026800988

## Descripción del Proyecto

Proyecto aplicado de la asignatura de **Machine Learning** (Maestría en Computación, TEC) desarrollado bajo la metodología **CRISP-DM**.

### Problema Abordado
Desarrollo de un sistema automático de **clasificación multiclase de patologías cerebrales** en registros de electroencefalograma (EEG), identificando automáticamente:
- **Convulsiones (Seizure)**
- **Descargas Periódicas Lateralizadas (LPD)**
- **Descargas Periódicas Generalizadas (GPD)**
- **Actividad Rítmica Lenta Lateralizada (LRDA)**
- **Actividad Rítmica Lenta Generalizada (GRDA)**
- **Actividad Normal (Other)**

### Objetivo General
Entrenar un modelo predictivo robusto que clasifique automáticamente patologías cerebrales en registros EEG con sensibilidad y especificidad clínicamente aceptables, mejorando el diagnóstico médico y reduciendo la carga de trabajo en neurólogos y radiólogos.

## Estructura del Proyecto

El repositorio contiene **dos iteraciones de CRISP-DM**. Son dos acercamientos independientes al mismo problema, cada uno con su propio código, partición y protocolo (ver [docs/crispdm.md](docs/crispdm.md)).

```
proyecto-ml-grupo-06/
│
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias Python (ambas iteraciones)
├── .gitignore
├── run_experiments.sh                 # It. 2 · ciclo 2.1: comparación en el fold 0 y CV de las CNNs
├── run_hq.sh                          # It. 2 · ciclo 2.2: Optuna, CV en alta calidad, auditoría
├── run_mixed.sh                       # It. 2 · ciclo 2.3: modelo único, dos cabezas, especialistas
│
├── docs/
│   ├── propuesta.md                   # Propuesta inicial del proyecto
│   ├── crispdm.md                     # Progreso CRISP-DM (Iteraciones 1 y 2)
│   ├── bitacora.md                    # Síntesis semanal de avances
│   ├── Estado_del_Arte_HMS.md
│   ├── informe_final.md               # Informe consolidado (Etapa 7)
│   └── informe_resultados.md          # Informe de resultados de la Iteración 2
│
├── notebooks/
│   ├── 01_exploracion.ipynb           # It. 1 · EDA y reserva del holdout de pacientes
│   ├── 02_preprocesamiento.ipynb      # It. 1 · folds y objetivos probabilísticos
│   ├── 03_modelado.ipynb              # It. 1 · CNN pequeña desde cero
│   ├── 04_evaluacion.ipynb            # It. 1 · evaluación frente a la clase mayoritaria
│   └── iteracion_2/
│       ├── 01_exploracion.ipynb       # EDA de EEG crudo y espectrogramas
│       ├── 02_preprocesamiento.ipynb  # Filtrado, montaje bipolar, imágenes de 8 espectrogramas
│       ├── 03_modelado.ipynb          # Baselines y CNNs preentrenadas
│       ├── 04_evaluacion.ipynb        # Evaluación del ensamble
│       ├── 05_optimizacion_alta_calidad.ipynb  # Optuna, entrenamiento ≥10 votos, auditoría
│       ├── 06_alta_y_baja_calidad.ipynb        # ¿Un modelo o dos? Enrutamiento y sobreconfianza
│       └── _py/                       # Fuente de cada notebook (formato percent)
│
├── src/
│   ├── __init__.py
│   ├── hms_cnn.py                     # It. 1 · CNN de una banda (64 × 64)
│   ├── config.py                      # It. 2 · rutas, clases, montaje bipolar
│   ├── data_preparation.py            # It. 2 · consolidación por eeg_id y folds por paciente
│   ├── features.py                    # It. 2 · EEG → espectrogramas; features de bandas
│   ├── build_cache.py                 # It. 2 · precalcula imágenes y features
│   ├── baselines.py                   # It. 2 · prior, regresión logística, gradient boosting
│   ├── train.py                       # It. 2 · CNNs (timm), pérdida KL, 2 etapas
│   ├── evaluate.py                    # It. 2 · KL, accuracy, F1, AUC, sensibilidad/especificidad
│   ├── predict.py                     # It. 2 · inferencia sobre test.csv
│   ├── hq.py, ensemble_opt.py, imbalance_audit.py   # It. 2 · ciclo 2.2
│   ├── mixed.py, routing.py           # It. 2 · ciclo 2.3
│   └── viz.py                         # It. 2 · estilo de gráficos
│
├── tests/test_hms_cnn.py              # It. 1 · pruebas del módulo hms_cnn
├── tools/py2nb.py                     # It. 2 · convierte notebooks/iteracion_2/_py/*.py en .ipynb
│
├── data/README.md                     # Cómo descargar HMS (secciones por iteración)
└── results/
    ├── metrics*.csv                   # It. 2 · métricas por experimento
    ├── history/, optuna/, mixed/      # It. 2 · curvas, búsquedas de hiperparámetros, enrutamiento
    └── figures/                       # It. 1: *.png en la raíz · It. 2: subcarpetas por notebook
```

Los datos (~25 GB), la caché de imágenes de la Iteración 2 (~9 GB) y los pesos entrenados viven **fuera del repositorio**.

## Instalación y Ejecución

### Requisitos
- Python 3.8+ (Iteración 2: Python 3.11 y GPU NVIDIA con CUDA)
- pip

### Setup Inicial
```bash
# Clonar o descargar el repositorio
cd proyecto-ml-grupo-06

# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
python -c "import pandas; import sklearn; print('Setup correcto')"
```

### Descargar Datos
[Instrucciones específicas en `data/README.md`]

### Iteración 1
Ejecutar `notebooks/01`–`04` en orden y `python -m src.hms_cnn` (ver `data/README.md`).

### Iteración 2
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m src.build_cache                                   # ≈7 min: imágenes y features
bash run_experiments.sh screening                           # ciclo 2.1: fold 0
bash run_experiments.sh cv effb0 efficientnet_b0.ra_in1k 1e-3
bash run_experiments.sh cv convnext_t convnext_tiny.fb_in22k_ft_in1k 2e-4 2
bash run_hq.sh                                              # ciclo 2.2
bash run_mixed.sh                                           # ciclo 2.3
python tools/py2nb.py notebooks/iteracion_2/_py/01_exploracion.py notebooks/iteracion_2/01_exploracion.ipynb
```

## Resultados principales

| | Iteración 1 | Iteración 2 |
|---|---|---|
| Modelo | CNN pequeña desde cero, 64 × 64, una banda | Ensamble de CNNs preentrenadas (EfficientNet-B0 + ConvNeXt-T) sobre 8 espectrogramas |
| Validación | Fold 1 (16 826 filas), holdout de pacientes reservado | CV de 5 folds por paciente (17 089 EEG), predicciones fuera de fold |
| Macro F1 | 0.384 | 0.633 (ciclo 2.1) |
| Accuracy | 0.391 | 0.717 (ciclo 2.1) |
| KL (métrica oficial) | — | 0.576 (ciclo 2.1) · **0.567** (ciclo 2.3, dos cabezas) |

Las cifras de las dos iteraciones no son el mismo experimento: cambian la partición, la unidad de muestra (fila o EEG) y la métrica principal.

**Ciclos de la Iteración 2** (detalle en [docs/informe_resultados.md](docs/informe_resultados.md)):
- **2.1:** el preentrenamiento aporta ~0.095 de KL. El EEG crudo aporta más que los espectrogramas de Kaggle. El ensamble mejora un 35 % la KL del mejor baseline clásico (0.882 → 0.576).
- **2.2:** entrenar solo con EEG de ≥10 votos mejora un poco la alta calidad (KL 0.342), pero empeora *Seizure* y los EEG de <10 votos (KL 0.867 frente a 0.681).
- **2.3:** un solo modelo con dos cabezas (alta y baja calidad) y una compuerta supera a los especialistas separados y a la cascada por confianza: KL macro 0.399 en alta y 0.755 en baja calidad, KL 0.567, AUC 0.923.

## Metodología: CRISP-DM

El proyecto sigue las 6 fases de CRISP-DM:

1. **Comprensión del Negocio** → Definición del problema, justificación
2. **Comprensión de los Datos** → Análisis exploratorio, características
3. **Preparación de los Datos** → Limpieza, transformación, feature engineering
4. **Modelado** → Selección y entrenamiento de modelos
5. **Evaluación** → Validación, métricas, análisis de errores
6. **Despliegue** → Documentación, reproducibilidad, conclusiones

**Ver detalles en:** [docs/crispdm.md](docs/crispdm.md)

## Estado del Proyecto

| Etapa | Actividad | Iteración 1 | Iteración 2 |
|-------|-----------|-------------|-------------|
| 1 | Propuesta Inicial | Completado | Reformulada (métrica KL) |
| 2 | Configuración y Planificación | Completado | Completado |
| 3 | Comprensión de Datos (EDA) | Parcial | Completado |
| 4 | Preparación + Baseline | En Progreso | Completado |
| 5 | Modelado y Experimentación | En Progreso (primera CNN) | Completado (3 ciclos) |
| 6 | Evaluación e Interpretación | En Progreso (resultados preliminares) | Completado (validación cruzada) |
| 7 | Entrega Final | Pendiente | Pendiente |

**Última actualización:** 2026-11-11

## Documentación

- [Propuesta Inicial](docs/propuesta.md) - Definición del problema y justificación
- [CRISP-DM Progress](docs/crispdm.md) - Avance por fases (actualizado semanalmente)
- [Bitácora](docs/bitacora.md) - Síntesis semanal de avances
- [Informe Final](docs/informe_final.md) - Consolidación del proyecto (Etapa 7)
- [Informe de resultados de la Iteración 2](docs/informe_resultados.md) - EDA, CNNs preentrenadas, optimización y ciclos 2.1–2.3

## Gestión de Tareas

Las tareas del proyecto se gestionan mediante **GitHub Issues**. Cada issue incluye:
- Fase CRISP-DM asociada
- Responsable (mediante @menciones)
- Descripción y criterio de cierre
- Registro de tiempo (comentarios: "Time spent: Xh")

Ver: [GitHub Issues](../../issues)

## Tablero de Avance

Seguimiento visual en **GitHub Projects** (Kanban board):
- **Backlog**: Tareas sin iniciar
- **In Progress**: Tareas activas
- **In Review**: Esperando revisión
- **Done**: Completadas

Ver: [GitHub Projects](../../projects)

## Contacto y Dudas

- **Jordani Mejía**: [detalles de contacto]
- **Luis Fernando Solano**: [detalles de contacto]

---

**Curso:** Machine Learning - Maestría en Computación  
**Instituto:** Instituto Tecnológico de Costa Rica (TEC)  
**Año:** 2026
