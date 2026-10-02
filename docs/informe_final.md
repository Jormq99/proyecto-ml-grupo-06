# Informe Final: Clasificación Multiclase de Patologías en Actividad Cerebral

**Curso:** Machine Learning - Maestría en Computación
**Institución:** Instituto Tecnológico de Costa Rica (TEC)
**Dataset:** HMS - Harmful Brain Activity Classification (Kaggle)
**Año:** 2026

**Autores:**
- Jordani Mejía Quirós (2018105537)
- Luis Fernando Solano Coto (2026800988)

**Fecha de entrega:** [A COMPLETAR]

---

## Resumen Ejecutivo

[A COMPLETAR AL FINAL DEL PROYECTO]

### Síntesis
[1 párrafo: Problema de clasificación de patologías cerebrales en EEG, metodología CRISP-DM aplicada, modelo seleccionado, resultado en Macro F1-Score/AUC-ROC, impacto potencial en diagnóstico médico]

### Contribuciones Principales
[Listado de hallazgos: qué bandas de frecuencia son más predictivas, qué patologías se confunden, interpretabilidad del modelo]

### Resultados Clave
- **Métrica principal (Macro F1-Score)**: [A COMPLETAR]
- **Métrica clínica (AUC-ROC)**: [A COMPLETAR]
- **Sensibilidad Seizure (crítica)**: [A COMPLETAR]%
- **Modelo seleccionado**: [A COMPLETAR]
- **Mejora vs baseline**: [A COMPLETAR]%

---

## 1. Introducción

### 1.1 Contexto del Problema

[A COMPLETAR]
- Importancia del diagnóstico automático de patologías cerebrales en EEG
- Desafíos de interpretación manual (tiempo, consistencia, escalabilidad)
- Estado del arte en clasificación de EEG con Machine Learning
- Importancia clínica de alta sensibilidad para convulsiones

### 1.2 Motivación

¿Por qué es importante este proyecto?

[A COMPLETAR]
- Impacto potencial en diagnóstico médico rápido y accesible
- Oportunidad académica de aplicar ML a datos médicos reales
- Exploración de feature engineering en señales EEG
- Manejo de desbalance severo de clases en diagnóstico

### 1.3 Objetivos
[Referencia a sección correspondiente en propuesta.md]

---

## 2. Revisión de Metodología CRISP-DM

### 2.1 Fase 1: Business Understanding

[Resumen de propuesta.md]

**Conclusión de fase:**
- ✅ Problema bien definido: Clasificación multiclase de 6 categorías de patologías
- ✅ Objetivos claros: Sensibilidad clínica, interpretabilidad
- ✅ Métricas de éxito identificadas: Macro F1, AUC-ROC, Sensibilidad/Especificidad
- ✅ Riesgos documentados: Desbalance, artefactos, generalización

### 2.2 Fase 2: Data Understanding

#### Descripción del Dataset

| Atributo | Detalle |
|----------|---------|
| **Fuente** | Kaggle HMS Competition |
| **Tamaño** | [A COMPLETAR] muestras |
| **Features** | [A COMPLETAR] características (bandas + estadísticas) |
| **Período temporal** | 10-60 segundos de actividad cerebral |
| **Granularidad** | Uno-a-uno: cada registro → una categoría de patología |
| **Clases objetivo** | 6 categorías: Seizure, LPD, GPD, LRDA, GRDA, Other |

#### Distribución de Clases

[Incluir gráfica de barplot de clases]

- **Clase más frecuente**: [A COMPLETAR] - [%]
- **Clase menos frecuente**: [A COMPLETAR] - [%]
- **Desbalance**: [A COMPLETAR] ratio
- **Problema**: Severamente desbalanceado

#### Variables Clave

[Tabla descriptiva de características espectrales principales]

| Variable | Tipo | Media | Desv. Est. | Notas |
|----------|------|-------|-----------|-------|
| Delta (0.5-4 Hz) | Float | [A COMPLETAR] | [A COMPLETAR] | Baja frecuencia, importante para convulsiones |
| Theta (4-8 Hz) | Float | [A COMPLETAR] | [A COMPLETAR] | Asociado con somnolencia |
| Alpha (8-12 Hz) | Float | [A COMPLETAR] | [A COMPLETAR] | Reposo, baseline normal |
| Beta (12-30 Hz) | Float | [A COMPLETAR] | [A COMPLETAR] | Actividad cognitiva |
| Gamma (30-100 Hz) | Float | [A COMPLETAR] | [A COMPLETAR] | Alta frecuencia |

#### Valores Faltantes y Artefactos

- **Missing values**: [A COMPLETAR]%
- **Artefactos detectados**: [A COMPLETAR]
- **Estrategia de tratamiento**: [A COMPLETAR]

#### Análisis Exploratorio de Tiempo

[Incluir gráficas]

- **Rango temporal**: [A COMPLETAR]
- **Densidad de transacciones**: [A COMPLETAR]
- **Patrones estacionales**: [A COMPLETAR]
- **Evidencia de degradación temporal**: [A COMPLETAR]

**Conclusión de fase:**
- Dataset adecuado para objetivos propuestos
- Características temporales significativas identificadas
- Preparado para preprocessing

### 2.3 Fase 3: Data Preparation

#### Limpieza de Datos

**Valores faltantes:**
- Métodos aplicados: [A COMPLETAR]
- Datos eliminados: [A COMPLETAR] filas
- Datos imputados: [A COMPLETAR] valores

**Outliers:**
- Método de detección: [IQR / Isolation Forest / DBSCAN]
- Outliers identificados: [A COMPLETAR]
- Tratamiento: [Eliminación / Transformación / Retención]

#### Transformaciones

**Variables numéricas:**
- Normalización: StandardScaler aplicado en [A COMPLETAR]
- Escalado: [A COMPLETAR]

**Variables categóricas:**
- Codificación: [OneHotEncoder / LabelEncoder]
- Categorías creadas: [A COMPLETAR]

#### Feature Engineering

**Features Estáticos (Baseline):**

| Feature | Descripción | Fórmula |
|---------|-------------|---------|
| Recency | Antigüedad última transacción | `días desde última transacción` |
| Frequency | Número de transacciones | `count(transacciones)` |
| Monetary | Monto total/promedio | `sum/mean(montos)` |
| [Feature X] | [Descripción] | [Fórmula] |

**Features Temporales (Time-Decay):**

| Feature | Descripción | Fórmula |
|---------|-------------|---------|
| Weighted Sum | Suma ponderada de montos | `Σ monto_i * exp(-t_i/τ)` |
| Weighted Freq | Frecuencia ponderada | `Σ exp(-t_i/τ)` |
| Estimated Halflife | Vida media estimada | `τ = [PENDIENTE]` |
| [Feature Y] | [Descripción] | [Fórmula] |

**Features de Grafos (Dinámicos):**

[A COMPLETAR si se implementaron]

#### Dataset Final

**Train set:**
- Muestras: [A COMPLETAR]
- Features: [A COMPLETAR]
- Distribución de clases: [A COMPLETAR]

**Test set:**
- Muestras: [A COMPLETAR]
- Features: [A COMPLETAR]
- Distribución de clases: [A COMPLETAR]

**Estrategia de split:**
- Método: Temporal split (train [t0-t1], test [t1-t2])
- Razón: Evitar data leakage y validar generalización temporal

**Conclusión de fase:**
- ✅ Dataset limpio y validado
- ✅ Features ingenierizadas para capturar degradación temporal
- ✅ Listo para modelado

### 2.4 Fase 4: Modeling

#### Enfoques Evaluados

**Enfoque 1: Estático (Baseline)**

Modelo: [A COMPLETAR]
```python
[Código del modelo]
```

**Parámetros:**
- [Param1]: [Valor]
- [Param2]: [Valor]

**Resultados en Train:**
- Accuracy: [A COMPLETAR]%
- Macro F1: [A COMPLETAR]%
- AUC-ROC: [A COMPLETAR]%

**Resultados en Test:**
- Accuracy: [A COMPLETAR]%
- Macro F1: [A COMPLETAR]%
- AUC-ROC: [A COMPLETAR]%
- NDCG@5: [A COMPLETAR]%
- NDCG@10: [A COMPLETAR]%

**Análisis:**
- [A COMPLETAR]

---

**Enfoque 2: Con Time-Decay**

Modelo: [A COMPLETAR]
```python
[Código del modelo]
```

**Parámetros:**
- [Param1]: [Valor]
- Vida media (τ) estimada: [A COMPLETAR] días

**Resultados en Test:**
- Accuracy: [A COMPLETAR]%
- Macro F1: [A COMPLETAR]%
- AUC-ROC: [A COMPLETAR]%
- NDCG@5: [A COMPLETAR]%
- NDCG@10: [A COMPLETAR]%

**Mejora vs Baseline:** [A COMPLETAR]%

**Análisis:**
- [A COMPLETAR]

---

**Enfoque 3: Dinámico (Grafos / LSTM)**

[A COMPLETAR SI SE IMPLEMENTÓ]

#### Hyperparameter Tuning

Modelo seleccionado: [A COMPLETAR]

**Estrategia:** Grid Search / Random Search

**Parámetros testeados:**
```
param_grid = {
    'max_depth': [A COMPLETAR],
    'learning_rate': [A COMPLETAR],
    'n_estimators': [A COMPLETAR]
}
```

**Mejores parámetros encontrados:**
- [Param1]: [Valor]
- [Param2]: [Valor]

**Mejora vs parámetros default:** [A COMPLETAR]%

#### Validación Cruzada

**Estrategia:** Stratified K-Fold (k=5) + Temporal Split

**Resultados:**
```
Fold 1: AUC-ROC = [A COMPLETAR]%
Fold 2: AUC-ROC = [A COMPLETAR]%
Fold 3: AUC-ROC = [A COMPLETAR]%
Fold 4: AUC-ROC = [A COMPLETAR]%
Fold 5: AUC-ROC = [A COMPLETAR]%

Promedio: [A COMPLETAR]% ± [A COMPLETAR]%
```

**Conclusión de fase:**
- ✅ Modelos entrenados y validados
- ✅ Hiperparámetros optimizados
- ✅ [Mejor modelo identificado]

### 2.5 Fase 5: Evaluation

#### Comparación de Modelos

| Modelo | Accuracy | Macro F1 | AUC-ROC | NDCG@5 | NDCG@10 | Tiempo (s) |
|--------|----------|----------|---------|--------|---------|-----------|
| Baseline (Estático) | [%] | [%] | [%] | [%] | [%] | [A COMPLETAR] |
| Con Time-Decay | [%] | [%] | [%] | [%] | [%] | [A COMPLETAR] |
| Dinámico (Grafos) | [%] | [%] | [%] | [%] | [%] | [A COMPLETAR] |

**Modelo seleccionado:** [A COMPLETAR]
**Justificación:** [A COMPLETAR]

#### Matriz de Confusión

[Incluir matriz visual]

**Interpretación:**
- Productos mejor predichos: [A COMPLETAR]
- Productos frecuentemente confundidos: [A COMPLETAR]
- Tasa de error para cada clase: [A COMPLETAR]

#### Feature Importance

[Incluir gráfica de top 10 features]

**Top 5 Features más importantes:**
1. [Feature A]: [A COMPLETAR]%
2. [Feature B]: [A COMPLETAR]%
3. [Feature C]: [A COMPLETAR]%
4. [Feature D]: [A COMPLETAR]%
5. [Feature E]: [A COMPLETAR]%

**Insight:** [A COMPLETAR]

#### SHAP Values (Explicabilidad)

[Incluir SHAP summary plot]

**Interpretación:**
- [A COMPLETAR]

#### Análisis de Errores

**Errores principales:**
1. Error: [Descripción]
   - Frecuencia: [A COMPLETAR]%
   - Causa probable: [A COMPLETAR]
   - Impacto: [A COMPLETAR]

2. Error: [Descripción]
   - [A COMPLETAR]

#### Validación de Hipótesis

**H1: Modelos con time-decay > Sin time-decay**
- Resultado: ✅ Confirmada / ❌ No confirmada / ⚠️ Parcialmente
- Diferencia: [A COMPLETAR]% en NDCG@5
- Significancia estadística: [t-test p-value = A COMPLETAR]
- Conclusión: [A COMPLETAR]

**H2: Grafos dinámicos > Features estáticas**
- Resultado: [A COMPLETAR]
- Conclusión: [A COMPLETAR]

**H3: Vida media varía por segmento de cliente**
- Resultado: [A COMPLETAR]
- Segmentos identificados: [A COMPLETAR]
- Valores de τ por segmento: [A COMPLETAR]

**Conclusión de fase:**
- ✅ Evaluación completa
- ✅ Hipótesis testeadas
- ✅ Modelo final seleccionado

### 2.6 Fase 6: Deployment

#### Modelo Final

**Nombre:** [A COMPLETAR]
**Versión:** 1.0
**Fecha:** [A COMPLETAR]

**Parámetros finales:**
```
[A COMPLETAR]
```

**Performance:**
- AUC-ROC: [A COMPLETAR]%
- NDCG@5: [A COMPLETAR]%
- NDCG@10: [A COMPLETAR]%

#### Reproducibilidad

✅ **Checklist de reproducibilidad:**
- [x] Código modular en `src/`
- [x] Notebooks ejecutables de principio a fin
- [x] Seeds fijos para reproducibilidad
- [x] `requirements.txt` con versiones exactas
- [x] Instrucciones de setup en `README.md`
- [x] Documentación CRISP-DM completa

**Cómo reproducir el proyecto:**
```bash
# 1. Clonar y setup
git clone [REPO]
pip install -r requirements.txt

# 2. Descargar datos
python src/data_preparation.py --download

# 3. Ejecutar pipeline completo
jupyter nbconvert --to notebook --execute notebooks/01_exploracion.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_preprocesamiento.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_modelado.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_evaluacion.ipynb

# 4. Cargar modelo entrenado
python src/evaluate.py --model_path results/model_final.pkl
```

---

## 3. Conclusiones

### 3.1 Hallazgos Principales

[A COMPLETAR]

1. **Hallazgo 1:** [Descripción y significancia]
2. **Hallazgo 2:** [Descripción y significancia]
3. **Hallazgo 3:** [Descripción y significancia]

### 3.2 Respuesta a Preguntas de Investigación

**Pregunta principal:** ¿Incorporar time-decay y grafos dinámicos mejora la predicción de NBP?

**Respuesta:** [A COMPLETAR]

**Evidencia:** [A COMPLETAR]

### 3.3 Impacto y Aplicabilidad

- Impacto académico: [A COMPLETAR]
- Impacto práctico para institución bancaria: [A COMPLETAR]
- Transferibilidad a otros dominios: [A COMPLETAR]

---

## 4. Limitaciones

1. **Limitación 1:** [Descripción]
   - Impacto: [A COMPLETAR]
   - Estrategia de mitigation: [A COMPLETAR]

2. **Limitación 2:** [Descripción]
   - Impacto: [A COMPLETAR]
   - Estrategia de mitigation: [A COMPLETAR]

3. **[A COMPLETAR]**

---

## 5. Trabajo Futuro

### 5.1 Mejoras Técnicas

1. **Implementación de GNNs completos**
   - Descripción: Extender a Graph Neural Networks más complejos
   - Beneficio esperado: Capturar patrones relacionales más sofisticados

2. **Temporal Point Processes**
   - Descripción: Modelar tiempos de eventos explícitamente
   - Beneficio esperado: Mejor captura de dinámicas temporales

3. **Ensemble Methods**
   - Descripción: Combinar múltiples enfoques (estático + dinámico)
   - Beneficio esperado: Mejorar robustez y generalización

### 5.2 Extensiones del Proyecto

1. **Segmentación de clientes**
   - Usar modelos diferentes por segmento
   - Vida media específica por segmento

2. **Incorporar contexto externo**
   - Variables macroeconómicas
   - Comportamiento del mercado

3. **Despliegue en producción**
   - API REST para predicciones en tiempo real
   - Monitoreo y reentrenamiento periódico

---

## 6. Apéndices

### A. Código Completo

[Referencia a notebooks y scripts]

Disponible en:
- `notebooks/01_exploracion.ipynb`
- `notebooks/02_preprocesamiento.ipynb`
- `notebooks/03_modelado.ipynb`
- `notebooks/04_evaluacion.ipynb`
- `src/` (scripts modulares)

### B. Figuras y Gráficas

Todas disponibles en `results/figures/`:
- EDA visualizations
- Confusion matrices
- Feature importance plots
- Learning curves
- SHAP plots

### C. Métricas Detalladas

Ver `results/metrics.csv`

### D. Declaración de Uso de IA Generativa

[Si se utilizó IA para documentación, debugging o código]

**Herramientas utilizadas:** [ej. Claude Haiku 4.5]
**Contexto de uso:** [ej. Generación de estructura CRISP-DM, debugging de XGBoost]
**Verificación:** [El equipo adaptó y validó manualmente todos los outputs]

---

## 7. Referencias

[A COMPLETAR]

### Papers y Libros
- [Referencia 1]
- [Referencia 2]

### Recursos Web
- [Referencia 1]
- [Referencia 2]

### Datasets
- [Fuente de datos utilizada]
- [Licencia]

---

## Firmas

| Rol | Nombre | Firma | Fecha |
|-----|--------|-------|-------|
| Autor 1 | Jordani Mejía Quirós | _____ | 2026-[MES]-[DÍA] |
| Autor 2 | Luis Fernando Solano Coto | _____ | 2026-[MES]-[DÍA] |
| Profesor | [Nombre Profesor] | _____ | 2026-[MES]-[DÍA] |

---

**Versión:** 1.0
**Última actualización:** [A COMPLETAR]
**Estado:** Plantilla para completar durante proyecto
