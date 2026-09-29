# Datos del Proyecto: Next-Best-Product

## Descripción General

Esta carpeta contiene (o referencias a) los datos utilizados en el proyecto de predicción de Next-Best-Product.

---

## Estructura

```
data/
├── README.md              # Este archivo
└── raw/                   # Datos sin procesar
    ├── [dataset_file].csv
    └── README_RAW.md      # Documentación específica
```

---

## Cómo Descargar los Datos

### Opción 1: Dataset Público (Kaggle)

[A COMPLETAR CON INSTRUCCIONES ESPECÍFICAS]

```bash
# Requerimientos:
# 1. Tener kagglehub instalado: pip install kagglehub
# 2. Autenticación Kaggle configurada (~/.kaggle/kaggle.json)

# Comando:
python -c "
import kagglehub
path = kagglehub.dataset_download('[competition_name]')
print(f'Dataset descargado en: {path}')
"

# O manualmente:
# 1. Ir a [URL Kaggle]
# 2. Descargar archivo .csv
# 3. Copiar a data/raw/
```

### Opción 2: Dataset Privado (Si aplica)

[A COMPLETAR CON INSTRUCCIONES]

```bash
# Contactar a [responsable]
# Descargar archivo
# Colocar en data/raw/
```

### Opción 3: Dataset Sintético

```bash
# Generar datos sintéticos
python src/generate_synthetic_data.py --output data/raw/synthetic_data.csv
```

---

## Descripción del Dataset

[A COMPLETAR DURANTE EDA]

| Atributo | Valor |
|----------|-------|
| **Nombre** | [A COMPLETAR] |
| **Fuente** | [A COMPLETAR] |
| **Tamaño** | [A COMPLETAR] muestras |
| **Período temporal** | [A COMPLETAR] |
| **Variables** | [A COMPLETAR] características |
| **Licencia** | [A COMPLETAR] |

### Variables Principales

[Tabla a completar durante EDA]

---

## Uso en el Proyecto

### Carga de Datos

```python
# En src/data_preparation.py
import pandas as pd

def load_data(path='data/raw/[dataset].csv'):
    """Carga el dataset principal"""
    df = pd.read_csv(path)
    return df

# Uso:
from src.data_preparation import load_data
df = load_data()
```

### Preprocesamiento

Ver: `notebooks/02_preprocesamiento.ipynb`

```python
from src.data_preparation import prepare_data
X_train, X_test, y_train, y_test, scaler = prepare_data(df)
```

---

## Notas Importantes

### Privacidad
[Si aplica: datos contienen información sensible, manejar con cuidado]

### Desbalance
[A COMPLETAR: Distribución de clases, ratio]

### Valores Faltantes
[A COMPLETAR: Porcentaje y estrategia de manejo]

### Outliers
[A COMPLETAR: Detección y manejo]

---

## .gitignore

Los datos crudos (*.csv, *.parquet, *.pkl) NO se suben al repositorio (ver `.gitignore`).

**Razones:**
- Archivos muy pesados
- Privacidad (si datos sensibles)
- Facilita reproducibilidad (descargar datos es procedimiento estándar)

**Si necesitas compartir datos:**
- Usar GitHub LFS (Large File Storage)
- Compartir enlace de descarga en este archivo
- Usar cloud storage (Google Drive, Dropbox)

---

## Reproducibilidad

Para garantizar reproducibilidad:
1. ✅ Documentar fuente de datos
2. ✅ Documentar pasos de descarga
3. ✅ Fijar random seeds
4. ✅ Usar versiones específicas de librerías (`requirements.txt`)

**Ver:** `notebooks/01_exploracion.ipynb` para verificación de datos

---

## Contacto

Si tienes dudas sobre los datos:
- Jordani Mejía Quirós
- Luis Fernando Solano Coto

---

**Última actualización:** 26/09/2026 
**Estado:** A completar durante proyecto
