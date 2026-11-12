# Informe de resultados de la Iteración 2: CNNs preentrenadas para HMS

**Proyecto:** Clasificación multiclase de patrones de actividad cerebral dañina en EEG (Kaggle HMS)
**Fases CRISP-DM cubiertas:** 2 (comprensión de datos), 3 (preparación), 4 (modelado), 5 (evaluación)
**Fecha:** noviembre 2026

Este informe documenta la **Iteración 2** de CRISP-DM, un acercamiento independiente del de la Iteración 1
(CNN desde cero, `notebooks/01`–`04` y `src/hms_cnn.py`), con su propio código, partición y protocolo. Se organiza en
tres ciclos: **2.1** (secciones 1–8), **2.2** (sección 9) y **2.3** (sección 10). Seguimiento por fases en
[docs/crispdm.md](crispdm.md).
Las etiquetas "fase 1", "iteración 2" e "iteración 3" que aparecen en los CSV, figuras y notebooks 05–06
corresponden a los ciclos 2.1, 2.2 y 2.3.

Notebooks: [01_exploracion](../notebooks/iteracion_2/01_exploracion.ipynb) · [02_preprocesamiento](../notebooks/iteracion_2/02_preprocesamiento.ipynb) ·
[03_modelado](../notebooks/iteracion_2/03_modelado.ipynb) · [04_evaluacion](../notebooks/iteracion_2/04_evaluacion.ipynb)
Métricas crudas: [results/metrics.csv](../results/metrics.csv) · Figuras: [results/figures/](../results/figures/)

---

## 1. Resumen

Se completó el análisis exploratorio del dataset HMS y se evaluaron CNNs preentrenadas en ImageNet sobre
imágenes de espectrogramas construidas a partir del EEG. El modelo final —un ensamble 50/50 de
**EfficientNet-B0 y ConvNeXt-Tiny**, validado con 5 folds agrupados por paciente— obtiene:

| Métrica | Todas las muestras | Muestras de alto acuerdo (≥10 votos) |
|---|---|---|
| **KL (métrica oficial, menor es mejor)** | **0.576** | **0.347** |
| Accuracy (clase mayoritaria) | 71.7 % | 73.8 % |
| F1 macro | 0.633 | 0.586 |
| AUC macro uno-contra-resto | 0.924 | 0.928 |

Frente al mejor baseline clásico (gradient boosting sobre potencias por banda: KL 0.882 / 0.656) la mejora es
de ~35 % / ~47 %. El preentrenamiento, el uso del EEG crudo además de los espectrogramas de Kaggle y el
entrenamiento en dos etapas contribuyen cada uno de forma medible.

## 2. Datos y hallazgos del EDA

- `train.csv`: 106 800 ventanas de 50 s de **17 089 EEG** de **1 950 pacientes**, sin nulos ni duplicados.
  EEG a 200 Hz con 19 electrodos + ECG; espectrogramas de 10 min con 4 regiones × 100 frecuencias (0.59–19.92 Hz).
- **El objetivo es una distribución de votos**, no una etiqueta. Los expertos discrepan sobre todo entre
  clases vecinas (LPD↔GPD, LRDA↔GRDA, Seizure↔LPD) y entre cualquier patrón y *Other*.
- **Calidad de etiquetas:** ~63 % de las filas tienen <10 votos. Esas filas son ~30 % *Seizure*,
  mientras que entre las de ≥10 votos *Seizure* es solo ~1.5 % y dominan *Other* y *GPD*.
  Las dos poblaciones tienen distribuciones distintas.
- **Duplicación y fuga:** el 89 % de los EEG tiene exactamente el mismo patrón de votos en todas sus
  sub-ventanas (que se solapan). El 44 % de los pacientes es monoclase y el 1 % de los pacientes
  concentra el 19 % de las filas.
- **Señal:** offset DC de ~−100 a −150 µV, ~2.5 % de ventanas con NaN, ~30 % con amplitudes p99 > 500 µV
  (artefactos); ~6 % de los espectrogramas de 10 min tienen tramos sin datos.
- **Firmas por clase (medianas, ≥10 votos):** asimetría izquierda-derecha ~4× mayor en LPD/LRDA
  que en GPD/GRDA; GRDA/LRDA con 87–90 % de la potencia en delta; *Seizure* con la mayor actividad
  rápida y *line-length* (3.1 vs. 1.5–2.0).

## 3. Preparación

| Decisión | Motivo (EDA) |
|---|---|
| Una muestra por `eeg_id` (sub-ventana central; objetivo = media de las distribuciones de votos) | Sub-ventanas casi duplicadas |
| `StratifiedGroupKFold` por paciente, 5 folds (0 pacientes compartidos, clases balanceadas) | Pacientes casi monoclase → fuga |
| Imputación de NaN, Butterworth 0.5–40 Hz, recorte ±1024 µV, montaje bipolar doble banana | Offset DC, artefactos, faltantes |
| Imagen 512×512 = 4 espectrogramas Kaggle (10 min) + 4 espectrogramas STFT del EEG (50 s), log y estandarizados | La información está en frecuencia, periodicidad y lateralidad |
| Aumentos: intercambio de hemisferios + enmascarado tiempo/frecuencia | Ninguna clase depende del lado; robustez a tramos faltantes |
| Entrenamiento en 2 etapas: todo → afinado en ≥10 votos | Distribuciones distintas según calidad |

## 4. Modelado

Pérdida KL sobre la distribución de votos; AdamW, calentamiento + coseno, bf16, lote 32,
12 épocas (etapa 1) + 4 épocas con lr/10 (etapa 2). Modelos de `timm` con pesos de ImageNet.

### 4.1 Screening en el fold 0 (≈3 400 EEG de validación)

| Modelo | KL | KL ≥10 votos | Accuracy | AUC macro | min |
|---|---|---|---|---|---|
| prior de clases | 1.261 | 0.926 | — | 0.50 | — |
| HistGradientBoosting (65 features de bandas) | 0.962 | 0.666 | — | — | — |
| **EfficientNet-B2** | **0.646** | **0.349** | 0.709 | 0.907 | 19 |
| **EfficientNet-B0** | 0.648 | 0.351 | 0.707 | 0.909 | 13 |
| RegNetY-016 | 0.655 | 0.351 | 0.699 | 0.907 | 20 |
| ConvNeXt-Tiny | 0.655 | 0.368 | 0.715 | 0.908 | 22 |
| ResNet34 | 0.717 | 0.438 | 0.676 | 0.888 | 10 |
| EfficientNet-B0 **sin preentrenar** | 0.743 | 0.394 | 0.668 | 0.886 | 14 |
| EfficientNet-B0 solo espectrogramas Kaggle | 0.751 | 0.430 | 0.671 | 0.882 | 8 |
| EfficientNet-B0 solo espectrogramas EEG | 0.689 | 0.365 | 0.696 | 0.895 | 8 |

Conclusiones del screening:
1. Todas las CNNs preentrenadas superan con amplitud al mejor baseline clásico.
2. **El preentrenamiento en ImageNet aporta ~0.095 de KL** con la misma arquitectura.
3. **El EEG crudo aporta más que los espectrogramas de Kaggle**, y combinarlos es lo mejor.
4. La etapa 2 mejora siempre, sobre todo en alta calidad (p. ej. B0: 0.459 → 0.351).
5. Ensamble de pares en fold 0: **B0 + ConvNeXt-T = 0.602 / 0.321** (mejor par; familias distintas,
   errores menos correlacionados) → elegidos para validación cruzada completa.

### 4.2 Validación cruzada completa (5 folds, 17 089 EEG)

| Modelo | KL | KL ≥10 votos | Accuracy | F1 macro | AUC macro | Tiempo |
|---|---|---|---|---|---|---|
| prior uniforme | 1.462 | 1.169 | 16.4 % | 0.047 | 0.500 | — |
| prior de clases | 1.263 | 0.909 | 41.9 % | 0.098 | 0.500 | — |
| regresión logística (bandas) | 1.061 | 0.814 | 49.9 % | 0.354 | 0.756 | — |
| HistGradientBoosting (bandas) | 0.882 | 0.656 | 56.9 % | 0.467 | 0.835 | — |
| EfficientNet-B0 — solo etapa 1 | 0.683 | 0.511 | 70.9 % | 0.634 | 0.912 | |
| **EfficientNet-B0** | 0.623 | 0.385 | 70.6 % | 0.624 | 0.915 | 88 min |
| ConvNeXt-Tiny — solo etapa 1 | 0.694 | 0.508 | 70.9 % | 0.634 | 0.911 | |
| **ConvNeXt-Tiny** | 0.629 | 0.399 | 70.9 % | 0.624 | 0.914 | 223 min* |
| **Ensamble 50/50** | **0.576** | **0.347** | **71.7 %** | **0.633** | **0.924** | |

\* Incluye ~1 h 50 min de una época ralentizada porque la VRAM se compartió con otras aplicaciones;
el tiempo normal es ~2 h.

- La etapa 2 (afinado en ≥10 votos) mejora la KL en ~0.06 en todas las muestras y en ~0.11–0.13 en alta calidad.
- El peso óptimo del ensamble buscado en los OOF fue exactamente 0.5, por lo que la cifra del ensamble no
  está inflada por el ajuste de pesos.
- Variación entre folds del ensamble: KL 0.576 ± 0.027, KL ≥10 votos 0.347 ± 0.030, AUC 0.923 ± 0.007.

## 5. Evaluación del modelo final

**Métricas por clase (todas las muestras):**

| Clase | Soporte | Sensibilidad | Especificidad | Precisión | F1 | AUC |
|---|---|---|---|---|---|---|
| Seizure | 2 800 | 0.631 | 0.969 | 0.798 | 0.705 | 0.936 |
| LPD | 2 573 | 0.644 | 0.971 | 0.800 | 0.714 | 0.947 |
| GPD | 1 799 | 0.719 | 0.975 | 0.774 | 0.745 | 0.959 |
| LRDA | 934 | 0.201 | 0.988 | 0.485 | 0.284 | 0.879 |
| GRDA | 1 828 | 0.501 | 0.969 | 0.663 | 0.571 | 0.917 |
| Other | 7 155 | 0.900 | 0.706 | 0.688 | 0.779 | 0.902 |

Hallazgos:
1. **Las descargas periódicas (GPD, LPD) son las mejor reconocidas**; *Seizure* tiene AUC alta, pero en alta
   calidad su precisión es solo 0.40 (hay apenas 182 crisis con ≥10 votos).
2. **LRDA es el punto débil** (sensibilidad 0.20): la clase más rara y la más confundida también por los expertos.
3. **Sesgo hacia *Other*.** La mayoría de los errores son "patrón → Other". En casos LRDA los expertos dan en
   promedio 0.26 de sus votos a *Other* y el modelo 0.41; en GRDA 0.28 vs. 0.36. El diagrama de confiabilidad
   muestra *Other* sobre-predicho.
4. **El error sigue al desacuerdo humano:** KL 0.16 en casos unánimes vs. 0.48 cuando la clase mayoritaria tiene
   <50 % de los votos. Las muestras de 3–4 votos (etiquetas casi *one-hot*) son las de mayor KL (0.89 con B0).
5. **Peores casos:** GPD/LPD de alto acuerdo con descargas periódicas visibles, clasificados como *Other* con
   >80 % de confianza — fallos reales del modelo.
6. **Uso clínico:** con especificidades de 0.97–0.99 para las clases patológicas, sirve como herramienta de
   priorización de registros, no como diagnóstico autónomo.

Figuras: `results/figures/evaluacion/` (matriz de confusión, ROC, calibración, KL por votos, peores casos,
confusión humana vs. modelo).

## 6. Limitaciones

- Una sola semilla por experimento; las diferencias <0.01 de KL entre arquitecturas en el screening no son
  concluyentes.
- No se ajustó el número de épocas: la etapa 1 sobreajusta después de ~8 épocas.
- El peso del ensamble se eligió sobre los mismos OOF (optimista); el ensamble 50/50 es la cifra honesta.
- La validación usa todas las muestras; el test oculto de Kaggle era solo de alto acuerdo, por lo que
  `kl_hq` es la referencia más cercana, aunque no es directamente comparable con el leaderboard.
- Las clases poco frecuentes (LRDA, *Seizure* de alta calidad con solo 182 EEG) siguen siendo las más débiles.

## 7. Próximos pasos sugeridos

1. Más contexto temporal del EEG (ventana central de 10–20 s a mayor resolución, o varias escalas).
2. Modelos 1D sobre la señal cruda o modelos 2D sobre el EEG "apilado" como imagen, para diversificar el ensamble.
3. Pseudo-etiquetado / auto-destilación de las muestras de <10 votos, que fue clave en las mejores soluciones.
4. Ajuste de épocas y *learning rate* por arquitectura; varias semillas.
5. Ponderar las clases poco frecuentes o sobremuestrear LRDA.

## 8. Reproducción

```bash
python -m src.build_cache                 # ~7 min con 16 procesos: imágenes + features
bash run_experiments.sh screening         # baselines + 8 CNNs en el fold 0 (~2.5 h, RTX 4070 Ti SUPER)
bash run_experiments.sh cv effb0 efficientnet_b0.ra_in1k 1e-3
bash run_experiments.sh cv convnext_t convnext_tiny.fb_in22k_ft_in1k 2e-4 2
python tools/py2nb.py notebooks/iteracion_2/_py/03_modelado.py notebooks/iteracion_2/03_modelado.ipynb
python tools/py2nb.py notebooks/iteracion_2/_py/04_evaluacion.py notebooks/iteracion_2/04_evaluacion.ipynb
python -m src.predict --models cv_effb0:efficientnet_b0.ra_in1k cv_convnext_t:convnext_tiny.fb_in22k_ft_in1k
```
