# Bitácora del Proyecto: Clasificación Multiclase de Patologías Cerebrales (HMS)

Registro de avances contrastados con el historial de Git y con reportes de trabajo de los integrantes, distinguiendo lo realizado de lo planificado.

## Semana 1 (30 de agosto - 5 de septiembre de 2026)

### Avances registrados
- El 3 de septiembre se amplió el README con la descripción del problema HMS, el objetivo del proyecto, la metodología CRISP-DM y la estructura prevista del repositorio (`f0e25e2`).

### Horas registradas
- 3 h

## Semana 2 (6 - 12 de septiembre de 2026)

No se realizaron avances significativos en esta semana

## Semana 3-4 (13 - 26 de septiembre de 2026)

No se realizaron avances significativos en esta semana

## Semana 5-6 (27 de septiembre - 10 de octubre de 2026)

### Avances registrados al 28 de septiembre
- El commit `144ad46` agregó `data/README.md`, `docs/Estado_del_Arte_HMS.md` y `notebooks/01_exploracion.ipynb`.
- Se incorporó un documento de estado del arte sobre clasificación de actividad cerebral en señales EEG.
- Se creó el notebook de exploración, contiene una lista de tareas pendientes.

### Pendientes derivados de lo observado
- Alinear las plantillas de datos y exploración con el dataset HMS antes de usarlas como evidencia de análisis.

### Horas registradas
- 4 h

### Avances reportados al 1 de octubre
- 3 h de trabajo entre el 30 de septiembre y el 1 de octubre para alinear los notebooks de exploración y preprocesamiento con el dataset HMS, documentar la reserva del holdout por paciente y preparar las tareas de seguimiento.
- Se planifican 4 h adicionales para continuar el proyecto, visualizavión y análisis de información, planeamiento de los siguientes pasos.

### Evidencias y pendientes
- Rama de trabajo: `Jormq99/initial_exploration_and_preprocessing`.
- Actualización de `notebooks/01_exploracion.ipynb` y `notebooks/02_preprocesamiento.ipynb`.
- Pendiente: ejecutar con los archivos HMS descargados, revisar los hallazgos reales y continuar con modelado y evaluación.

### Horas registradas
- 3 h 30 de sept
- 2 h 01 de oct

---

**Última actualización:** 2026-08-01


## 07 de noviembre de 2026: Iteración 2, preparación de datos (Luis Fernando Solano)

Inicio de un segundo acercamiento, independiente de la Iteración 1: CNNs preentrenadas sobre EEG crudo y espectrogramas, con validación cruzada de 5 folds por paciente.

### Trabajo realizado
- Descarga del dataset fuera del repositorio y documentación en `data/README.md`.
- EDA del EEG crudo y de los espectrogramas (`notebooks/iteracion_2/01_exploracion`):
  - dos poblaciones según el número de votos (<10 y ≥10);
  - sub-ventanas casi duplicadas (89 % de los EEG repiten el patrón de votos);
  - pacientes casi monoclase y artefactos de señal.
- Preparación (`src/`):
  - una muestra por `eeg_id` y 5 folds por paciente;
  - filtrado y montaje bipolar;
  - imágenes de 8 espectrogramas por muestra y caché de imágenes y features.
- Baselines clásicos y comparación de 8 CNNs en el fold 0. El mejor baseline (gradient boosting) obtuvo KL 0.882.

### Horas registradas
- 4 h 07 de nov

## 08 de noviembre de 2026 — Primera prueba de una CNN

### Trabajo realizado
- Se confirmó que `train.csv` contiene 106,800 filas etiquetadas y 1,950 pacientes; el preprocesamiento ya había reservado el holdout interno por paciente y creado cinco folds agrupados.
- Se reemplazaron las plantillas de modelado y evaluación por notebooks orientados a HMS. Se agregó `src/hms_cnn.py`, una CNN pequeña de una sola banda, que usa ventanas de 50 segundos del espectrograma, las redimensiona a 64 × 64 y conserva los seis votos de especialistas como etiquetas probabilísticas.
- Se entrenó durante una época con 480 ejemplos y se evaluó con 240 ejemplos aleatorios del fold 0. No se usaron el holdout reservado ni los datos de test.
- Resultados de la prueba con muestreo aleatorio: accuracy **0.2833**, Macro F1 **0.1700**. La referencia de predecir siempre la clase más frecuente obtuvo accuracy **0.1792** y Macro F1 **0.0506**. Recall por clase: Seizure 0.8605, LPD 0, GPD 0.8857, LRDA 0, GRDA 0 y Other 0.
- Se guardaron las métricas en `results/hms_cnn_experiments.csv`, la matriz de confusión en `results/hms_cnn_confusion_matrix.csv` y el modelo local en `models/hms_cnn_baseline.pt`; estos artefactos están excluidos de Git.
- En esa etapa, las cuatro pruebas automatizadas del módulo pasaban. La ejecución fue en CPU; el entorno no tiene GPU ni `torchvision`.

### Interpretación y límites
- Es una prueba de flujo, no un modelo útil todavía: con 240 ejemplos y una época, no detectó cuatro de las seis clases. El Macro F1 bajo deja claro que hay que experimentar y mejorar.
- Hubo una corrida piloto con muestreo balanceado que no representa la distribución natural y una comparación de mayoría no válida para ese muestreo. Se conserva como prueba técnica, marcada como `class_stratified_pilot`; no se usa para afirmar mejora. La corrida aleatoria es la referencia preliminar comparable.
- La matriz de confusión muestra que la corrida aleatoria solo predijo Seizure y GPD; el recall de las otras cuatro clases fue cero. Por eso el Macro F1 bajo es más informativo que el accuracy en esta etapa.
- Se revisaron como referencia los protocolos de TP1, TP2 y TP3: separar test antes de elegir el modelo, ajustar transformaciones solo en entrenamiento, usar los mismos folds para comparar y no modificar el procedimiento después de ver test; en problemas temporales, construir características solo con información disponible hasta el origen de pronóstico. En HMS, se aplican estas ideas reservando pacientes completos y verificando también que no se compartan registros EEG ni espectrogramas entre train y validación.
- El fold 0 ya fue consultado para la prueba piloto y el fold 1 para la primera corrida completa. Para reducir el snooping, no se usarán sus puntuaciones para seguir seleccionando ni ajustando arquitecturas; las comparaciones posteriores se harán en los folds de desarrollo 2–4 con el mismo protocolo para cada modelo. El holdout de pacientes permanece intacto y se evaluará una sola vez cuando se congele el modelo y el análisis.
- No se ejecutó una red preentrenada: falta `torchvision` y no se debe asumir que pesos de imágenes naturales mejorarán espectrogramas EEG. Será una comparación posterior controlada, si se valida una dependencia compatible, usando los mismos folds y límites de datos.
- Esta conclusión corresponde solo a la prueba pequeña descrita arriba. No es un resultado clínico ni final.

### Horas registradas
- 5 h 08 de nov

## 08 de noviembre de 2026: Iteración 2, ciclos 2.1 y 2.2

### Trabajo realizado
- **Ciclo 2.1:** validación cruzada de 5 folds de EfficientNet-B0 y ConvNeXt-T, ensamble y evaluación por clase (`notebooks/iteracion_2/03`–`04`). El ensamble obtuvo KL **0.576**, Macro F1 **0.633** y AUC **0.924**.
- **Ciclo 2.2:**
  - Auditoría de desbalance: la accuracy y la KL simple están infladas por *Other*, así que desde aquí se optimiza la KL macro.
  - Optimización con Optuna y entrenamiento solo con EEG de ≥10 votos (`notebooks/iteracion_2/05`). Mejora la alta calidad (KL 0.342), pero empeoran *Seizure* y los EEG de <10 votos.

### Horas registradas
- 4 h 08 de nov (madrugada: CV y evaluación)
- 5 h 08 de nov (tarde: Optuna y alta calidad)

## 09 de noviembre de 2026: Iteración 2, ciclo 2.3

### Trabajo realizado
- Comparación de un modelo único con `lq_weight` optimizado, un modelo de dos cabezas con compuerta y especialistas de alta y baja calidad con cascada, compuerta y mezcla (`src/mixed.py`, `src/routing.py`, `notebooks/iteracion_2/06`).
- Mejor modelo: ensamble de dos cabezas, con KL **0.567** y KL macro 0.399 en alta y 0.755 en baja calidad. La cascada por confianza falla: envía ~95 % de los casos a un solo especialista.

### Interpretación y límites
- La Iteración 2 usa su propia partición de 5 folds sobre todos los pacientes, sin holdout aparte. Sus cifras no son directamente comparables con las de la Iteración 1.
- La optimización de hiperparámetros se hizo solo en el fold 0 y con una semilla por experimento.

### Horas registradas
- 5 h 09 de nov (madrugada: modelos de EfficientNet-B0)
- 5 h 09 de nov (mañana: modelos de ConvNeXt-T y enrutamiento)
- 1 h 09 de nov (tarde: notebook 06)

## 10 de noviembre de 2026 — Corrida completa inicial en desarrollo

### Trabajo realizado y resultados
- Se ejecutó el experimento completo predefinido en el fold 1: 67,305 filas para entrenamiento, 16,826 para validación, tres épocas y CPU. La división mantiene pacientes, registros EEG y espectrogramas separados. Se preservó la distribución natural de las clases y no se usó el holdout reservado.
- La CNN obtuvo accuracy **0.3910** y Macro F1 **0.3836** en validación. La referencia que predice siempre la clase más frecuente obtuvo accuracy **0.1993** y Macro F1 **0.0554**. Esto supera esa referencia simple, pero aún deja muchos errores y no demuestra generalización a pacientes no vistos fuera del esquema de folds.
- El recall por clase fue: Seizure **0.4131**, LPD **0.2901**, GPD **0.1953**, LRDA **0.5175**, GRDA **0.5085** y Other **0.3977**. GPD y LPD siguen siendo difíciles de detectar.
- Se generaron gráficas de historial de entrenamiento, distribución de clases, matriz de confusión y resumen de métricas por clase. El notebook 04 se ejecutó correctamente y produjo las gráficas de evaluación. La matriz de confusión completa está en `results/hms_cnn_confusion_matrix.csv`; los resultados y el modelo permanecen como artefactos locales ignorados por Git.
- Las seis pruebas automatizadas pasaron. No se agregó ningún archivo al staging.

### Interpretación y límites
- El Macro F1 de validación fue 0.3162, 0.4212 y 0.3836 en las épocas 1, 2 y 3. La tercera época fue la fijada de antemano y es la reportada; no se eligió la mejor época mirando la validación. El aumento de la pérdida de validación en la tercera época es una señal para investigar, no una razón para reajustar este experimento después de ver el fold.
- Este es un primer resultado con todos los ejemplos de desarrollo disponibles para el fold 1, no una prueba final. Los folds 0 y 1 ya fueron consultados; las comparaciones futuras deben usar los folds 2–4 con protocolo fijado de antemano, y el holdout reservado debe seguir intacto hasta congelar el análisis.
- El rendimiento no es clínico ni suficiente para uso real. Hace falta comparar de forma controlada y evaluar una sola vez en el holdout cuando el procedimiento esté congelado.

### Horas registradas
- 3 h 10 de nov

## 11 de noviembre de 2026: Integración de la Iteración 2 al repositorio

### Trabajo realizado
- Integración del trabajo local sobre los cambios de la Iteración 1 sin modificar sus archivos. Los notebooks de la Iteración 2 quedan en `notebooks/iteracion_2/`.
- Documentación en `docs/crispdm.md`, `docs/informe_resultados.md` y `README.md`. La subida se hace en tres PR, uno por bloque.

### Horas registradas
- 2 h 11 de nov

Las horas de la Iteración 2 se estimaron a partir de las marcas de tiempo de los logs y resultados de ejecución. Incluyen corridas de GPU sin supervisión continua.
