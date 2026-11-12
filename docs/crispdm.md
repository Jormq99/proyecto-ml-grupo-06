# Progreso CRISP-DM: Clasificación Multiclase de Patologías Cerebrales (HMS)

## Descripción General

Este documento registra el progreso del proyecto a través de las **6 fases de la metodología CRISP-DM**. Se actualiza continuamente durante el desarrollo, no solo al final.

El proyecto avanza en **dos iteraciones de CRISP-DM que son dos acercamientos independientes** al mismo problema. Cada una recorre todas las fases con su propio código, partición y protocolo de validación, y conviven en el repositorio sin modificarse entre sí. La segunda toma como punto de partida las conclusiones de la primera, no su código.

| Iteración | Contenido | Dónde |
|---|---|---|
| **1** | CNN pequeña entrenada desde cero (64 × 64, una banda), holdout de pacientes, fold 1 | Fases 1–6 de este documento · `notebooks/01`–`04` · `src/hms_cnn.py` |
| **2** | CNNs preentrenadas en ImageNet, EEG crudo + espectrogramas, validación cruzada de 5 folds | Sección [Iteración 2](#iteración-2-cnns-preentrenadas-y-validación-cruzada-completa) · `notebooks/iteracion_2/` · `docs/informe_resultados.md` |

**Última actualización:** 2026-11-11

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

# Iteración 2: CNNs preentrenadas y validación cruzada completa

**Estado:** ✅ Cerrada (7 – 9 de noviembre de 2026) · **Responsable:** Luis Fernando Solano

## 2.0 Por qué una segunda iteración

La Iteración 1 dejó funcionando el flujo de datos y el protocolo contra la fuga de información. También dejó claros sus límites:

- Macro F1 de 0.384 en un solo fold, con una CNN pequeña entrenada desde cero sobre espectrogramas de 64 × 64 y una banda.
- Sobreajuste visible en la tercera época y entrenamiento en CPU.
- La hipótesis H2 ("una CNN preentrenada supera a la entrenada desde cero") quedó pendiente por falta de `torchvision` y de GPU.

La Iteración 2 es un segundo acercamiento: un pipeline nuevo, en `src/` (módulos distintos de `hms_cnn.py`) y `notebooks/iteracion_2/`, con GPU (RTX 4070 Ti SUPER, 16 GB). Recorre de nuevo las fases 1 a 5 para responder lo que quedó pendiente. No modifica los notebooks, el módulo ni los resultados de la Iteración 1.

## 2.1 Comprensión del negocio (reformulada)

- **Métrica principal: divergencia KL** entre la distribución de votos y la predicha. Es la métrica oficial de la competencia y respeta que el objetivo sea una distribución, no una etiqueta. Se siguen reportando Macro F1, accuracy, AUC y sensibilidad y especificidad por clase, para poder comparar con la Iteración 1.
- **Preguntas:**
  1. H2: ¿supera una CNN preentrenada a la misma arquitectura sin preentrenar?
  2. ¿Aporta el EEG crudo (convertido a espectrograma) información además de los espectrogramas de Kaggle?
  3. ¿Los EEG con pocos votos (< 10) ayudan o perjudican al entrenamiento?

## 2.2 Plan por ciclos

| Ciclo | Objetivo | Fases |
|---|---|---|
| 2.1 | EDA ampliado, preparación de imágenes de 512 × 512, baselines clásicos, comparación de 8 CNNs en el fold 0 y validación cruzada de 5 folds del mejor ensamble | 2, 3, 4, 5 |
| 2.2 | Auditoría de desbalance (KL macro), optimización bayesiana con Optuna y entrenamiento solo con EEG de alta calidad (≥ 10 votos) | 4, 5 |
| 2.3 | Un solo modelo frente a dos especialistas (alta y baja calidad) con reglas para elegir cuál usar | 4, 5 |

## 2.3 Cambios de protocolo respecto a la Iteración 1

- **Una muestra por `eeg_id`** (la sub-ventana central): el 89 % de los EEG repiten el mismo patrón de votos en sub-ventanas que se solapan.
- **Partición propia e independiente:** `StratifiedGroupKFold` de 5 folds agrupados por paciente (semilla 42) sobre los 1 950 pacientes de `train.csv`. La validación se hace con las predicciones fuera de fold de los 5 folds, sin holdout aparte. El holdout de pacientes pertenece al protocolo de la Iteración 1 y no se usa aquí. Por eso las cifras de las dos iteraciones se comparan como acercamientos distintos y no como el mismo experimento.

## 2.4 Resultados por ciclo

| Ciclo | Resultado principal | Decisión |
|---|---|---|
| 2.1 | Ensamble EfficientNet-B0 + ConvNeXt-T: KL 0.576 (0.347 en ≥10 votos), Macro F1 0.633, AUC 0.924. El mejor baseline clásico obtuvo KL 0.882. | Base para los ciclos siguientes |
| 2.2 | Solo ≥10 votos + Optuna: KL 0.342 en alta calidad, pero KL 0.867 en <10 votos y recall de *Seizure* de 0.62 → 0.45 | No descartar los EEG de pocos votos |
| 2.3 | Modelo de dos cabezas con compuerta: KL 0.567, KL macro 0.399 / 0.755, recall de *Seizure* 0.65 / 0.70 | **Modelo final de la Iteración 2** |

**Hipótesis:**
- **H2 confirmada:** con la misma arquitectura (EfficientNet-B0) y los mismos folds, el preentrenamiento en ImageNet mejora la KL en ~0.095 (0.743 → 0.648 en el fold 0).
- **El EEG crudo aporta:** solo EEG da KL 0.689 y solo espectrogramas de Kaggle 0.751; combinados, 0.648.
- **Los EEG de pocos votos sí aportan.** Representan las clases de forma distinta, pero un solo modelo que aprende de ambos grupos supera a los especialistas.

## 2.5 Las dos iteraciones lado a lado

| | Iteración 1 | Iteración 2 |
|---|---|---|
| Entrada | Espectrograma de Kaggle, 64 × 64, una banda | 4 espectrogramas de Kaggle + 4 del EEG crudo, 512 × 512 |
| Modelo | CNN pequeña desde cero, CPU | CNNs preentrenadas (timm), GPU, pérdida KL |
| Unidad de muestra | Fila de `train.csv` | Un EEG (sub-ventana central) |
| Validación | Fold 1; holdout de pacientes reservado | CV de 5 folds por paciente (predicciones fuera de fold) |
| Métrica principal | Macro F1 = 0.384 | KL = 0.567 (Macro F1 0.633 en el ciclo 2.1) |

Son acercamientos independientes, así que la tabla compara los enfoques, no el mismo experimento.

Como referencia descriptiva, sobre los 390 pacientes (3 946 EEG) que la Iteración 1 reserva como holdout, las predicciones fuera de fold del ensamble del ciclo 2.1 dan KL 0.514 y Macro F1 0.677. En el resto de pacientes dan KL 0.594 y Macro F1 0.617. Ninguna decisión de la Iteración 2 se tomó con esa división.

## 2.6 Limitaciones y próximos pasos

- Optimización de hiperparámetros solo en el fold 0 y con una semilla por experimento. Las diferencias menores a 0.01 de KL no son concluyentes.
- LRDA sigue siendo la clase más débil (sensibilidad 0.20 en el ciclo 2.1).
- Próximos pasos:
  - más contexto temporal del EEG;
  - modelos 1D sobre la señal cruda para diversificar el ensamble;
  - pseudo-etiquetado de los EEG de pocos votos;
  - varias semillas.

**Estado de la Iteración 2:** ✅ Cerrada (9 de noviembre de 2026). Integrada al repositorio el 11 de noviembre de 2026.

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

La tabla anterior corresponde a la Iteración 1. Iteración 2: fases 1–5 completadas en tres ciclos (ver [Iteración 2](#iteración-2-cnns-preentrenadas-y-validación-cruzada-completa)); la fase 6 queda pendiente para la entrega final.

**Próxima reunión de avance**: [FECHA]
**Responsable de seguimiento**: [A ASIGNAR]

---

**Versión**: 2.1
**Última actualización**: 2026-11-11
**Próxima revisión**: Próxima iteración del experimento
