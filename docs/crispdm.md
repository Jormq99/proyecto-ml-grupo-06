# Progreso CRISP-DM: Clasificación Multiclase de Patologías Cerebrales (HMS)

## Descripción General

Este documento registra el progreso del proyecto a través de las **6 fases de la metodología CRISP-DM**. Se actualiza continuamente durante el desarrollo, no solo al final.

**Última actualización:** 2026-09-01

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

### Estado: 📋 Pendiente

#### 2.1 Descripción del Dataset
- [ ] Fuente de datos: HMS Kaggle Competition (Confirmado)
- [ ] Número de muestras: ~17,000 registros EEG
- [ ] Número de features: [A determinar en EDA]
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
- [ ] Distribución de clases (target): % de cada patología
- [ ] Distribución por banda de frecuencia
- [ ] Valores faltantes y outliers
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
- [ ] [Serán documentadas aquí]

**Próxima fase**: Data Preparation
**Responsable**: [A ASIGNAR]

---

## Fase 3: Data Preparation / Preparación de los Datos

### Estado: 📋 Pendiente

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
- **Train shape**: [X_train (Nsamples, features)] - Estratificado
- **Test shape**: [X_test (Nsamples, features)]
- **Features totales**: [Número]
- **Distribución train**: [% por clase]
- **Distribución test**: [% por clase]

**Próxima fase**: Modeling
**Responsable**: [A ASIGNAR]

---

## Fase 4: Modeling / Modelado

### Estado: 📋 Pendiente

#### 4.1 Enfoques Evaluados

##### Enfoque 1: Estático (Baseline)

**Modelo**: Logistic Regression (Multinomial)
```python
LogisticRegression(
    multi_class='multinomial',
    max_iter=1000,
    class_weight='balanced'
)
```
- **AUC-ROC**: [PENDIENTE]
- **NDCG@5**: [PENDIENTE]
- **Tiempo entrenamiento**: [PENDIENTE]s
- **Características**: Features estáticas (Recency, Frequency, Monetary)

**Modelo**: XGBoost con features estáticas
```python
XGBClassifier(
    objective='multi:softprob',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)
```
- **AUC-ROC**: [PENDIENTE]
- **NDCG@5**: [PENDIENTE]
- **Feature Importance**: [PENDIENTE]

##### Enfoque 2: Con Time-Decay

**Modelo**: XGBoost con time-decay features
```python
# Features incluyen:
# - exp(-t/τ) donde τ = vida media estimada
# - Suma ponderada de montos
# - Frecuencia ponderada
```
- **AUC-ROC**: [PENDIENTE]
- **NDCG@5**: [PENDIENTE]
- **Mejora vs baseline**: [PENDIENTE]%
- **Vida media (τ) estimada**: [PENDIENTE] días

##### Enfoque 3: Dinámico (Grafos)

**Modelo**: GNN o LSTM (si complejidad lo permite)
- **Arquitectura**: [PENDIENTE]
- **AUC-ROC**: [PENDIENTE]
- **NDCG@5**: [PENDIENTE]
- **Complejidad computacional**: [PENDIENTE]

#### 4.2 Hyperparameter Tuning

**Modelo seleccionado**: [A DECIDIR]

Grid Search / Random Search:
- [ ] Parámetros testeados: [PENDIENTE]
- [ ] Mejor combinación: [PENDIENTE]
- [ ] Mejora vs default: [PENDIENTE]%

#### 4.3 Validación
- [ ] Stratified K-Fold (k=5): [PENDIENTE]
- [ ] Temporal Split (train [t0-t1], test [t1-t2]): [PENDIENTE]
- [ ] Learning curves: [Sobreajuste detectado?]

**Próxima fase**: Evaluation
**Responsable**: [A ASIGNAR]

---

## Fase 5: Evaluation / Evaluación

### Estado: 📋 Pendiente

#### 5.1 Métricas de Rendimiento

| Métrica | Baseline | Mejor Modelo | Mejora |
|---------|----------|-------------|--------|
| Accuracy | [%] | [%] | [%] |
| Macro F1 | [%] | [%] | [%] |
| AUC-ROC | [%] | [%] | [%] |
| NDCG@5 | [%] | [%] | [%] |
| NDCG@10 | [%] | [%] | [%] |
| Precision@5 | [%] | [%] | [%] |
| Recall@5 | [%] | [%] | [%] |

#### 5.2 Análisis de Errores

**Matriz de Confusión:**
```
[Matriz será completada durante evaluación]
```

**Productos frecuentemente confundidos:**
- Producto A ↔ Producto B: [% de confusión]
- Razón: [Análisis]
- Impacto: [Crítico / Medio / Bajo]

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

- **H1**: Modelos con time-decay > Sin time-decay
  - [ ] **Resultado**: [PENDIENTE]
  - [ ] **Diferencia estadística**: [t-test / Mann-Whitney]

- **H2**: Grafos dinámicos > Features estáticas
  - [ ] **Resultado**: [PENDIENTE]
  - [ ] **Diferencia estadística**: [PENDIENTE]

- **H3**: Vida media varía por segmento
  - [ ] **Resultado**: [PENDIENTE]
  - [ ] **Segmentos identificados**: [PENDIENTE]

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
- **Rendimiento**: AUC-ROC [%], NDCG@5 [%]
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

---

## Resumen de Avance

| Fase | Estado | % Completado | Responsable |
|------|--------|-------------|-------------|
| 1. Business Understanding | 🔄 En Progreso | 80% | Ambos |
| 2. Data Understanding | 📋 Pendiente | 0% | [A ASIGNAR] |
| 3. Data Preparation | 📋 Pendiente | 0% | [A ASIGNAR] |
| 4. Modeling | 📋 Pendiente | 0% | [A ASIGNAR] |
| 5. Evaluation | 📋 Pendiente | 0% | [A ASIGNAR] |
| 6. Deployment | 📋 Pendiente | 0% | [A ASIGNAR] |

**Próxima reunión de avance**: [FECHA]
**Responsable de seguimiento**: [A ASIGNAR]

---

**Versión**: 2.0
**Última actualización**: 2026-08-29
**Próxima revisión**: Semana de 2026-09-05
