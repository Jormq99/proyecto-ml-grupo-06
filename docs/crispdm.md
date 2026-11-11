# Progreso CRISP-DM: Clasificación Multiclase de Patologías Cerebrales (HMS)

## Descripción General

Este documento registra el progreso del proyecto a través de las **6 fases de la metodología CRISP-DM**. Se actualiza continuamente durante el desarrollo, no solo al final.

**Última actualización:** 2026-11-10

---

## Fase 1: Business Understanding / Comprensión del Negocio

### Estado: ✅ Completado

#### 1.1 Definición del Problema
- ✅ **Problema identificado**: Clasificación multiclase de patologías cerebrales en registros EEG
- ✅ **Contexto**: Diagnóstico automatizado en neurología y neuroradiología
- ✅ **Desafío central**: Desbalance severo de clases, múltiples categorías de patologías
- ✅ **Novedad**: Aplicación real de ML supervisado en diagnóstico médico

#### 1.2 Objetivos y Preguntas de Investigación
- **Pregunta principal**: ¿Puede un modelo ML clasificar automáticamente patologías cerebrales con desempeño comparable al de neurólogos?
- **H1**: Modelo XGBoost supera Logistic Regression en AUC-ROC >= 5%
- **H2**: Con manejo de desbalance, Macro F1-Score >= 0.65
- **H3**: Características importantes concentradas en bandas delta-theta

#### 1.3 Métricas de Éxito
- **Métrica primaria**: Macro F1-Score (balance entre clases)
- **Métrica secundaria**: AUC-ROC por clase, Sensibilidad/Especificidad
- **Interpretabilidad**: Feature importance, SHAP values
- **Baseline esperado**: Logistic Regression ~0.55 F1, XGBoost ~0.70 F1

#### 1.4 Riesgos y Consideraciones
| Riesgo | Mitigation |
|--------|-----------|
| Desbalance severo de clases | Usar class weights, SMOTE oversample, stratified k-fold |
| Data leakage en normalización | Normalizar DESPUÉS de train-test split |
| Baja sensibilidad en clases raras | Aumentar weight de clases raras, resampling |
| Falta de interpretabilidad | SHAP values, feature importance, error analysis |

#### 1.5 Decisiones Metodológicas
- ✅ Usar CRISP-DM como framework
- ✅ Dataset: HMS Kaggle Competition (público, reproducible)
- ✅ Comparar baseline (Logistic Regression) vs avanzados (XGBoost, Neural Network)
- ✅ Métricas clínicas: AUC-ROC, sensibilidad, especificidad (no solo accuracy)
- ✅ Documentación completa en GitHub + Markdown
- ✅ Énfasis en interpretabilidad para confianza médica

**Próxima fase**: Data Understanding
**Responsable**: Jordani Mejía / Luis Fernando Solano

---

## Fase 2: Data Understanding / Comprensión de los Datos

### Estado: 🔄 Parcial

#### 2.1 Descripción del Dataset
- [x] Fuente de datos: HMS Kaggle Competition
- [x] 106,800 filas etiquetadas y 1,950 pacientes
- [x] Seis objetivos probabilísticos a partir de votos de especialistas
- [ ] Período temporal: 10-60 segundos de actividad cerebral por registro
- [ ] Granularidad: Uno-a-uno: cada registro → una categoría

#### 2.2 Variables Clave
| Variable | Tipo | Descripción | Valores/Rango |
|----------|------|-------------|------|
| Espectrograma | Array 2D | Tiempo × Frecuencia | Píxeles de imagen |
| Canales EEG | Time-series | 19 canales estándar | Amplitud en µV |
| Banda Delta | Float | 0.5-4 Hz | Features estadísticas |
| Banda Theta | Float | 4-8 Hz | Features estadísticas |
| Banda Alpha | Float | 8-12 Hz | Features estadísticas |
| Banda Beta | Float | 12-30 Hz | Features estadísticas |
| Banda Gamma | Float | 30-100 Hz | Features estadísticas |
| **Target: Patología** | Categorical | Categoría de diagnóstico | Seizure, LPD, GPD, LRDA, GRDA, Other |

#### 2.3 Análisis Exploratorio (EDA)

**Tareas principales:**
- [x] Distribución de votos por clase
- [x] Conteo de valores faltantes en metadatos de desarrollo
- [ ] Distribución por banda de frecuencia
- [ ] Análisis de outliers
- [ ] Correlaciones entre bandas
- [ ] Patrones visuales en espectrogramas por clase

**Visualizaciones esperadas:**
- [ ] Barplot de distribución de patologías (BALANCEADO vs DESBALANCEADO)
- [ ] Espectrogramas de ejemplo por cada clase
- [ ] Distribución de estadísticas por banda
- [ ] Matriz de correlaciones entre bandas

#### 2.4 Características Temporales de EEG
- [ ] Potencia espectral por banda (delta, theta, alpha, beta, gamma)
- [ ] Ratios entre bandas (ej: theta/alpha)
- [ ] Entropía espectral
- [ ] Simetría entre hemisferios cerebrales (canales izq vs derech)

#### 2.5 Decisiones Tomadas
- [x] Reservar pacientes completos antes del análisis y mantenerlos fuera de los experimentos iniciales.
- [x] Agrupar los folds por paciente y conservar los seis votos como objetivos probabilísticos.
- [ ] Completar el análisis de patrones espectrales y documentar hallazgos por clase.

**Próxima fase**: Data Preparation
**Responsable**: [A ASIGNAR]

---

## Fase 3: Data Preparation / Preparación de los Datos

### Estado: 🔄 Parcial

#### 3.1 Limpieza de Datos
- [ ] Tratamiento de valores faltantes: [Método]
- [ ] Detección y tratamiento de artefactos: [Método]
- [ ] Validación de integridad de espectrogramas
- [ ] Verificación de rangos válidos de amplitud EEG

#### 3.2 Transformaciones
- [ ] Normalización de potencia espectral: StandardScaler o MinMaxScaler
- [ ] Normalización de amplitud EEG: Z-score por canal
- [ ] Codificación de clases target: LabelEncoder para 6 categorías
- [ ] Conversión de espectrogramas a features (si necesario)

#### 3.3 Feature Engineering

**Features Estadísticos por Banda:**
- [ ] Media y std de potencia por banda (delta, theta, alpha, beta, gamma)
- [ ] Asimetría (skewness) de potencia por banda
- [ ] Curtosis (kurtosis) de potencia por banda
- [ ] Máximo y mínimo de potencia por banda

**Features de Complejidad:**
- [ ] Entropía espectral
- [ ] Entropía de Shanon
- [ ] Complejidad Lempel-Ziv

**Features Relativas:**
- [ ] Ratios de potencia: theta/alpha, delta/theta, etc.
- [ ] Total spectral power (suma todas las bandas)
- [ ] Ratios absolutas vs relativas

**Features Espaciales (Multi-canal):**
- [ ] Asimetría hemisférica: canal izquierdo vs derecho
- [ ] Conectividad entre canales (correlación)
- [ ] Índice de simetría cerebral

#### 3.4 Manejo de Desbalance de Clases
- **Estrategia**: Usar class weights en modelos + stratified k-fold
- **Alternativa 1**: SMOTE oversampling para clases raras
- **Alternativa 2**: Undersampling de clase mayoritaria (Other)
- **Métrica**: Usar Macro F1-Score en validación (no accuracy)

#### 3.5 Dataset Final
- **Preparación disponible**: folds agrupados por paciente, objetivos probabilísticos y extracción de ventanas de espectrograma.
- **Entrada inicial de la CNN**: ventana de 50 segundos, redimensionada a una matriz 64 × 64 de una sola banda.
- **Prueba inicial**: 480 filas de entrenamiento y 240 de validación; se conserva como piloto y no como resultado final.
- **Corrida actual**: fold 1 completo, con 67,305 filas de entrenamiento y 16,826 de validación, manteniendo la distribución natural. El holdout reservado no se usa.
- **Pendiente**: medir variación en folds no consultados (2–4) y justificar cualquier cambio de transformaciones o manejo del desbalance antes de ejecutarlo.

**Próxima fase**: Modeling
**Responsable**: [A ASIGNAR]

---

## Fase 4: Modeling / Modelado

### Estado: 🔄 Parcial — primera corrida completa de desarrollo ejecutada

#### 4.1 Modelo inicial
- **Arquitectura**: CNN pequeña entrenada desde cero con espectrogramas de una banda.
- **Entrenamiento**: 67,305 ejemplos, 3 épocas predefinidas, fold 1; CPU.
- **Validación**: 16,826 ejemplos del fold 1, separado por paciente, registro EEG y espectrograma.
- **Accuracy**: 0.3910; **Macro F1**: 0.3836.
- **Referencia de clase mayoritaria**: accuracy 0.1993; Macro F1 0.0554.
- **Recall por clase**: Seizure 0.4131, LPD 0.2901, GPD 0.1953, LRDA 0.5175, GRDA 0.5085, Other 0.3977.
- **Curva Macro F1 de validación por época**: 0.3162, 0.4212 y 0.3836. Se reporta la tercera época, fijada antes de la corrida; no se seleccionó la época con la validación.
- **Gráficas**: historial de entrenamiento, distribución de clases, matriz de confusión y métricas por clase, generadas en `results/figures/`.

Esta corrida usa todo el conjunto de desarrollo disponible para el fold, pero solo evalúa un fold ya consultado. Es una medición exploratoria; no permite conclusiones clínicas ni sustituye la evaluación final en el holdout.

#### 4.2 Pendiente
- [ ] Repetir el protocolo fijado en los folds de desarrollo no consultados 2–4 para medir variación.
- [ ] Probar una CNN preentrenada solo si puede hacerse una comparación controlada con la misma partición y datos.
- [ ] Investigar el sobreajuste observado en la pérdida de validación, sin reajustar retrospectivamente esta corrida.

**Próxima fase**: Evaluation
**Responsable**: [A ASIGNAR]

---

## Fase 5: Evaluation / Evaluación

### Estado: 🔄 Parcial — evaluación exploratoria en el fold 1

#### 5.1 Métricas de Rendimiento

| Métrica | Clase mayoritaria | CNN inicial |
|---------|-------------------|-------------|
| Accuracy | 0.1993 | 0.3910 |
| Macro F1 | 0.0554 | 0.3836 |

Estos valores corresponden a 16,826 ejemplos de validación en el fold 1; no son métricas del holdout. La corrida piloto de 240 ejemplos del fold 0, registrada arriba en esta bitácora, es distinta y no se mezcla con esta comparación.

#### 5.2 Análisis de Errores

La matriz de confusión del piloto aleatorio (240 ejemplos, filas = clase real, columnas = predicción) fue:

| Real \ Predicción | Seizure | LPD | GPD | LRDA | GRDA | Other |
|-------------------|--------:|----:|----:|-----:|-----:|------:|
| Seizure | 37 | 0 | 6 | 0 | 0 | 0 |
| LPD | 36 | 0 | 9 | 0 | 0 | 0 |
| GPD | 4 | 0 | 31 | 0 | 0 | 0 |
| LRDA | 32 | 0 | 1 | 0 | 0 | 0 |
| GRDA | 42 | 0 | 3 | 0 | 0 | 0 |
| Other | 35 | 0 | 4 | 0 | 0 | 0 |

El modelo solo predijo Seizure y GPD en esta corrida; cuatro clases quedaron sin aciertos. La matriz también se guarda en `results/hms_cnn_confusion_matrix.csv`.

Para la corrida completa del fold 1, el notebook 04 genera la matriz de confusión y gráficas de métricas por clase, distribución de clases y comparación con la referencia mayoritaria. La pérdida de validación disminuye hasta la época 2 y aumenta en la época 3; dado que las tres épocas se fijaron previamente, no se cambia retrospectivamente el resultado reportado.

#### 5.3 Interpretabilidad

**Feature Importance (Top 10):**
1. [Feature X]: [%]
2. [Feature Y]: [%]
...

**SHAP Analysis:**
- [ ] Dependence plots para features clave
- [ ] Force plots para predicciones individuales
- [ ] Insights de interpretabilidad

#### 5.4 Validación de Hipótesis

- **H1**: La CNN supera una referencia de clase mayoritaria.
  - [x] Resultado preliminar: sí en esta muestra pequeña; requiere más folds y datos.
- **H2**: Una CNN preentrenada supera la CNN entrenada desde cero.
  - [ ] Pendiente: `torchvision` no está instalado; no se ha probado transferencia de aprendizaje.

#### 5.5 Limitaciones
- [ ] [A documentar]

**Próxima fase**: Deployment
**Responsable**: [A ASIGNAR]

---

## Fase 6: Deployment / Despliegue y Conclusiones

### Estado: 📋 Pendiente

#### 6.1 Modelo Final Seleccionado
- **Nombre**: [PENDIENTE]
- **Versión**: [PENDIENTE]
- **Rendimiento**: Macro F1, sensibilidad por clase y métricas acordes con los objetivos probabilísticos.
- **Justificación**: [PENDIENTE]

#### 6.2 Reproducibilidad
- [ ] Código modular en src/
- [ ] Notebooks ejecutables de principio a fin
- [ ] Seeds fijos para reproducibilidad
- [ ] requirements.txt con versiones
- [ ] Instrucciones de setup en README.md

#### 6.3 Conclusiones Principales
- [ ] [A documentar]

#### 6.4 Recomendaciones
- [ ] [A documentar]

#### 6.5 Trabajo Futuro
- [ ] [A documentar]

---

## Declaración de Uso de IA Generativa

[Si se utilizó GPT/Claude para documentación o debugging, documentar aquí]

- **Herramienta usada**: [ej. Claude Haiku 4.5]
- **Contexto de uso**: [ej. Generación de estructura CRISP-DM, debugging de XGBoost]
- **Tipo de apoyo**: [ej. Template generado, revisión de código]
- **Verificación**: [ej. El equipo adaptó y validó manualmente todos los outputs]

## Protocolo para evitar leakage y snooping

Se adaptan buenas prácticas vistas en TP1, TP2 y TP3 al tipo de datos HMS:

1. **Separar por paciente, no por fila.** Una fila es un segmento, no una persona independiente. Si segmentos del mismo paciente aparecen en entrenamiento y validación, el modelo puede reconocer características personales y aparentar que generaliza mejor. Por eso se reserva un grupo de pacientes y se crean folds agrupados; además, el código comprueba que no se compartan registros EEG ni espectrogramas entre ambos lados.
2. **Mantener aparte el holdout final.** Se usa desarrollo para explorar y comparar; el holdout de pacientes reservado no participa en entrenamiento, selección, gráficos ni decisiones. Se consulta una sola vez cuando el pipeline quede congelado.
3. **Comparar candidatos con el mismo protocolo.** Igual fold, tamaño de muestra, semilla, métrica y presupuesto. El notebook 04 solo lee métricas guardadas; no vuelve a entrenar ni cambia el modelo.
4. **Registrar las validaciones ya vistas.** El fold 0 se usó para comprobar el flujo y el fold 1 para la primera corrida completa. Ambos se consideran diagnósticos, no se usarán para seguir escogiendo arquitecturas; las comparaciones posteriores se harán en los folds 2–4. El holdout permanece como evaluación final.
5. **Evitar transformaciones aprendidas con validación.** En esta CNN, el reescalado se calcula por imagen a partir de esa propia ventana; no se ajusta un escalador global con todo el conjunto. Si más adelante se agregan estadísticas globales, normalización por dataset o aumento aprendido, deben ajustarse solo dentro de cada entrenamiento.
6. **Respetar el objetivo temporal del problema.** Este proyecto clasifica una ventana EEG ya observada; no pronostica el siguiente instante, por lo que las ventanas se extraen del segmento etiquetado. Si se construye un sistema de pronóstico futuro, se deberán usar solo ventanas anteriores al instante de predicción, como se practica en TP2.

La referencia de división aleatoria estratificada de TP3 es útil cuando cada fila es independiente. En HMS se prioriza el paciente como grupo porque varios segmentos pueden proceder de la misma persona; conservar el grupo es más importante que repartir filas al azar.

---

## Resumen de Avance

| Fase | Estado | % Completado | Responsable |
|------|--------|-------------|-------------|
| 1. Business Understanding | ✅ Completado | 100% | Ambos |
| 2. Data Understanding | 🔄 Parcial | 60% | Ambos |
| 3. Data Preparation | 🔄 Parcial | 45% | Ambos |
| 4. Modeling | 🔄 Parcial | 20% | Ambos |
| 5. Evaluation | 🔄 Parcial | 15% | Ambos |
| 6. Deployment | 📋 Pendiente | 0% | [A ASIGNAR] |

**Próxima reunión de avance**: [FECHA]
**Responsable de seguimiento**: [A ASIGNAR]

---

**Versión**: 2.1
**Última actualización**: 2026-11-10
**Próxima revisión**: Próxima iteración del experimento
