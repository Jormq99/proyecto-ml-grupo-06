# Propuesta: Clasificación Multiclase de Patologías en Actividad Cerebral
## Dataset: HMS - Harmful Brain Activity Classification

---

## 1. Título Tentativo

**Clasificación Automática de Patologías Cerebrales mediante Registros de EEG: Detección de Convulsiones, Descargas Periódicas y Actividad Rítmica Anómala**

---

## 2. Problema

### Contexto
En el campo de la neurología y diagnóstico médico, la capacidad de identificar automáticamente patologías en registros de electroencefalograma (EEG) es crítica para:
- Diagnóstico rápido de convulsiones y trastornos cerebrales
- Reducción de carga de trabajo en radiólogos y neurólogos
- Detección temprana de actividades cerebrales anómalas
- Mejora en calidad y consistencia de diagnósticos

### Desafío Específico
Los análisis tradicionales de EEG son:
1. **Dependientes de expertos**: Requieren neurólogos certificados para interpretar registros
2. **Lentos y costosos**: Análisis manual consume horas de recursos especializados
3. **Propensos a variabilidad**: Diferentes interpretaciones según el experto
4. **Limitados en escala**: Difícil procesar volúmenes grandes de registros

**Realidad del Dataset HMS**: Múltiples categorías de patologías (convulsiones, descargas periódicas, actividad rítmica), **desbalance severo de clases**, y datos multimodales complejos (espectrogramas y características espectrales).

### Hipótesis Principal
**Un sistema de Machine Learning con ingeniería de características apropiada y selección de modelos puede clasificar automáticamente patologías cerebrales en registros EEG con sensibilidad y especificidad clínicamente aceptables.**

---

## 3. Justificación Académica y Práctica

### Justificación Académica
- **Aplicabilidad real**: Problema auténtico en diagnóstico médico, no académico ficticio
- **Complejidad técnica**: Requiere integración de:
  - Feature engineering avanzado (extracción de características de señales)
  - Manejo de desbalance de clases severo
  - Machine Learning supervisionado (clasificación multiclase)
  - Evaluación crítica de métricas clínicas (sensibilidad, especificidad, AUC-ROC)
  - Interpretabilidad en contexto médico
- **Reproducibilidad**: Uso de dataset público (Kaggle HMS Competition)

### Justificación Práctica
- **Impacto en salud**: Automatización de diagnósticos = mejor acceso a cuidado médico
- **Escalabilidad**: Modelo entrenado puede procesar cientos o miles de registros
- **Costo-efectividad**: Reducción de carga en profesionales médicos
- **Casos de uso inmediatos**: Hospitales, clínicas, laboratorios de neurología

---

## 4. Objetivo General

Desarrollar y evaluar un sistema de clasificación multiclase que:
1. **Clasifique automáticamente** patologías cerebrales en registros EEG
2. **Maneje el desbalance severo** de clases mediante técnicas apropiadas
3. **Evalúe rendimiento** usando métricas clínicamente relevantes
4. **Sea interpretable** para confianza médica

Comparando rendimiento entre enfoques **simples (baseline) vs avanzados**.

---

## 5. Objetivos Específicos

1. **Exploración y Comprensión**
   - Caracterizar distribución de clases y patrones en espectrogramas de EEG
   - Identificar características más discriminativas entre patologías
   - Detectar desafíos de desbalance: qué clases son más raras
   - Analizar correlaciones entre bandas de frecuencia

2. **Preparación de Datos**
   - Implementar limpieza y normalización de datos EEG
   - Extraer features relevantes: estadísticas por banda (delta, theta, alpha, beta, gamma)
   - Aplicar técnicas de manejo de desbalance: stratified k-fold, class weights, SMOTE
   - Crear train-test split respetando estructura de datos

3. **Modelado**
   - Entrenar modelo baseline: Logistic Regression para establecer punto de referencia
   - Entrenar modelos avanzados: Random Forest, XGBoost, Neural Networks
   - Comparar rendimiento sistemáticamente
   - Validación cruzada estratificada para confiabilidad

4. **Evaluación**
   - Evaluar mediante métricas apropiadas: AUC-ROC por clase, Macro F1-Score
   - Análisis de errores: Confusion matrix - ¿Qué patologías se confunden?
   - Interpretabilidad: SHAP values - ¿Qué features predicen cada patología?
   - Comparación con baseline

5. **Documentación**
   - Registro completo de decisiones en CRISP-DM
   - Código reproducible y modular (notebooks + scripts)
   - Informe con hallazgos, limitaciones y recomendaciones

---

## 6. Pregunta o Hipótesis de Trabajo

### Pregunta Principal
¿Es posible entrenar un modelo de Machine Learning que clasifique automáticamente patologías cerebrales en registros EEG con desempeño comparable o superior al baseline?

### Hipótesis Testeable
- **H1**: Modelo XGBoost supera Logistic Regression en AUC-ROC >= 5%
- **H2**: Con manejo de desbalance (class weights), Macro F1-Score >= 0.65
- **H3**: Las características más importantes están concentradas en bandas delta-theta (baja frecuencia)

---

## 7. Datos Preliminares

### Fuente
- **Dataset**: HMS Harmful Brain Activity Classification (Kaggle Competition)
- **URL**: https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification

### Características del Dataset
| Característica | Detalle |
|---------------|---------|
| **Número de muestras** | ~17,000 registros de EEG |
| **Número de features** | Depende de extracción; espectrogramas + estadísticas |
| **Período temporal** | Registros de 10-60 segundos de actividad cerebral |
| **Granularidad** | Uno-a-uno: cada registro → una categoría de patología |
| **Clases objetivo** | 6 categorías: Seizure, LPD, GPD, LRDA, GRDA, Other |
| **Desbalance** | Severo: Seizure ~10%, Other ~35%, LPD ~20%, etc. |
| **Variables temporales** | Espectrogramas 2D (tiempo × frecuencia), canales multi-EEG |

### Desafíos de Datos
1. **Desbalance de clases severo**: Algunas patologías muy raras (ej: LRDA <5%)
2. **Datos multimodales**: Espectrogramas (imágenes) + señales temporales (time-series)
3. **Ruido y artefactos**: EEG es sensible a ruido muscular, movimiento ocular
4. **Variabilidad entre sujetos**: Diferentes patrones cerebrales según edad, comorbilidades
5. **Interpretabilidad crítica**: Médicos necesitan entender decisiones del modelo

---

## 8. Tipo de Problema ML

- **Categoría Principal**: **Clasificación Multiclase Supervisada**
  - Input: Espectrograma + features EEG + metadatos del paciente
  - Output: Una de 6 categorías de patología

- **Características Especiales**:
  - Desbalance severo de clases
  - Datos multimodales (imágenes + series temporales)
  - Requisito de interpretabilidad para aplicación médica
  - Múltiples canales EEG (19 canales estándar)

- **Complejidad**: **Media-Alta**
  - Requiere feature engineering sofisticado
  - Manejo de desbalance crítico
  - Evaluación cuidadosa con métricas clínicas
  - Interpretabilidad no trivial

---

## 9. Modelos Candidatos

### Enfoques Baseline (Punto de Referencia)

| Modelo | Razón | Complejidad |
|--------|-------|------------|
| **Logistic Regression** | Baseline lineal simple, interpretable | Baja |
| **Naive Bayes Gaussiano** | Probabilístico, rápido, baseline robusto | Baja |
| **Decision Tree** | No-paramétrico, maneja no-linealidades | Media |

### Enfoques Avanzados (Experimentales)

| Modelo | Razón | Complejidad |
|--------|-------|------------|
| **Random Forest** | Ensemble, reduce overfitting, feature importance | Media-Alta |
| **XGBoost / LightGBM** | SOTA para tabular, gradient boosting, maneja desbalance | Alta |
| **Neural Network (MLP)** | Puede capturar relaciones complejas entre features | Alta |
| **CNN (si usamos espectrogramas como imágenes)** | Aprovecha estructura espacial de espectrogramas | Muy Alta |

### Selección Recomendada
**Baseline**: Logistic Regression con class weights
**Primario**: XGBoost con ajuste de desbalance
**Bonus**: Neural Network o CNN (si tiempo/recursos lo permiten)

---

## 10. Métricas Principales

### Clasificación Multiclase
- **Accuracy**: Corrección general (pero cuidado con desbalance)
- **Macro F1-Score**: Promedio de F1 por clase sin ponderar - **métrica primaria**
- **Weighted F1-Score**: Promedio ponderado por clase
- **AUC-ROC por clase**: One-vs-Rest para cada patología

### Métricas Clínicas (Críticas)
- **Sensibilidad (Recall)**: ¿Qué proporción de casos patológicos se detectan?
  - Particularmente importante para **Seizure** (falsos negativos = riesgo médico)
- **Especificidad**: ¿Qué proporción de negativos se clasifican correctamente?
- **Precisión por clase**: Para evaluar falsos positivos

### Análisis de Errores
- **Matriz de Confusión**: Visualizar qué patologías se confunden entre sí
- **Curva ROC por clase**: Sensibilidad vs 1-Especificidad para cada categoría
- **Análisis de errores**: ¿Qué características tienen registros clasificados incorrectamente?

### Interpretabilidad
- **Feature Importance** (XGBoost, Random Forest)
- **SHAP Values** (explicabilidad local)
- **Permutation Importance**
- **Análisis por banda de frecuencia**: ¿Qué bandas (delta, theta, etc.) son más predictivas?

### Comparación de Modelos
- **Tabla de resultados**: Accuracy, Macro F1, AUC-ROC, Sensibilidad Seizure, tiempo entrenamiento

---

## 11. Riesgos y Mitigation

| Riesgo | Probabilidad | Impacto | Mitigation |
|--------|-------------|--------|-----------|
| **Desbalance extremo de clases** | Muy Alta | Crítico | Usar class weights, SMOTE oversample, stratified k-fold |
| **Overfitting** | Alta | Medio | Validación cruzada, early stopping, regularización |
| **Data leakage** | Media | Crítico | Train-test split cuidadoso; no normalizar antes de split |
| **Falta de interpretabilidad** | Media | Alto | SHAP values, feature importance, error analysis |
| **Baja sensibilidad en clases raras** | Alta | Alto | Aumentar weight de clases raras, resampling |
| **Mala generalización a nuevos pacientes** | Media | Alto | Validación independiente, cross-validation temporal |
| **Requiere expertise médico para validación** | Media | Medio | Colaboración con profesional médico (si posible) |

---

## 12. Resultados Esperados

### Entregables
1. ✅ Documentación CRISP-DM completa (propuesta, bitácora, informe final)
2. ✅ Notebooks ejecutables (EDA, preprocesamiento, modelado, evaluación)
3. ✅ Scripts modulares (data_preparation.py, train.py, evaluate.py)
4. ✅ Comparación cuantitativa: Baseline vs Modelos Avanzados
5. ✅ Visualizaciones: Distribuciones de clases, confusion matrices, feature importance
6. ✅ Interpretación de errores: Análisis de confusiones entre patologías
7. ✅ Informe final con recomendaciones clínicas

### Métricas Esperadas (Estimativo)
- **Baseline (Logistic Regression)**: Macro F1 ~0.55-0.60, AUC-ROC ~0.70
- **Modelo avanzado (XGBoost)**: Macro F1 ~0.65-0.75, AUC-ROC ~0.80-0.85
- **Mejor modelo**: Macro F1 > 0.70, AUC-ROC > 0.82

### Timeline
- **Semana 1-2**: Propuesta + Setup (Fases 1-2)
- **Semana 3-4**: EDA + Preparación (Fases 3-4)
- **Semana 5-6**: Modelado (Fase 5)
- **Semana 7**: Evaluación (Fase 6)
- **Semana 8**: Informe Final + Presentación (Fase 7)

---

## Aprobación

| Rol | Nombre | Fecha | Firma |
|-----|--------|-------|-------|
| Integrante 1 | Jordani Mejía Quirós | 2026-09-01 | ✓ |
| Integrante 2 | Luis Fernando Solano Coto | 2026-09-01 | ✓ |
| Profesor | [Nombre Profesor] | [Fecha] | [ ] |

---

**Versión:** 1.0
**Última actualización:** 2026-09-01
**Estado:** Propuesta Inicial - HMS Realignment
