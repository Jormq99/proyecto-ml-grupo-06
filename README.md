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

```
repo-machine-learning/
│
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias Python
├── .gitignore                         # Archivos a excluir de git
│
├── docs/
│   ├── propuesta.md                   # Propuesta inicial del proyecto
│   ├── crispdm.md                     # Progreso CRISP-DM (6 fases)
│   ├── bitacora.md                    # Síntesis semanal de avances
│   └── informe_final.md               # Informe consolidado (Etapa 7)
│
├── notebooks/
│   ├── 01_exploracion.ipynb           # Análisis exploratorio de EEG (EDA)
│   ├── 02_preprocesamiento.ipynb      # Limpieza y feature engineering de EEG
│   ├── 03_modelado.ipynb              # Entrenamiento de modelos
│   └── 04_evaluacion.ipynb            # Evaluación y análisis de resultados
│
├── src/
│   ├── __init__.py
│   ├── data_preparation.py            # Carga y limpieza de datos EEG
│   ├── features.py                    # Ingeniería de características (espectros, bandas)
│   ├── train.py                       # Entrenamiento de modelos
│   └── evaluate.py                    # Evaluación y métricas (sensibilidad, AUC-ROC)
│
├── data/
│   ├── README.md                      # Documentación sobre datos (como descargar HMS)
│   └── raw/                           # Datos sin procesar (descargados de Kaggle)
│
└── results/
    ├── figures/                       # Visualizaciones (confusion matrix, ROC curves)
    └── metrics.csv                    # Resultados de modelos
```

## Instalación y Ejecución

### Requisitos
- Python 3.8+
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

| Etapa | Actividad | Estado |
|-------|-----------|--------|
| 1 | Propuesta Inicial | Completado |
| 2 | Configuración y Planificación | En Progreso |
| 3 | Comprensión de Datos (EDA) | Pendiente |
| 4 | Preparación + Baseline | Pendiente |
| 5 | Modelado y Experimentación | Pendiente |
| 6 | Evaluación e Interpretación | Pendiente |
| 7 | Entrega Final | Pendiente |

**Última actualización:** 2026-09-01

## Documentación

- [Propuesta Inicial](docs/propuesta.md) - Definición del problema y justificación
- [CRISP-DM Progress](docs/crispdm.md) - Avance por fases (actualizado semanalmente)
- [Bitácora](docs/bitacora.md) - Síntesis semanal de avances
- [Informe Final](docs/informe_final.md) - Consolidación del proyecto (Etapa 7)

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
