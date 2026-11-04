# Partición Oficial del Dataset IDRiD Sin Aumento de Datos

> **Principio de Evaluación Clínica Rigurosa:**
> - **Cero Aumento de Datos:** No se aplican rotaciones, giros horizontales (*flips*), traslaciones ni copias sintéticas en ningún subconjunto.
> - **Preservación Anatómica:** Se mantiene la orientación espacial fisiológica real de la retina (papila óptica nasal vs. fóvea temporal).
> - **Censo Completo del Dataset IDRiD:** El almacenamiento contiene **4,128 archivos** ($\times 8$), que corresponden exactamente a **516 imágenes originales de pacientes únicos** ($413$ en `train/` y $103$ en `test/`).

## 1. Censo Global del Dataset IDRiD (516 Originales / 4,128 en Disco) y Distribución del Split 80/20

| Partición / Carpeta | Archivos en Disco ($\times 8$) | **Total Imágenes Originales** | No Referible (0) | Referible (1) | Grado 0 (Normal) | Grado 1 (Leve) | Grado 2 (Mod) | Grado 3 (Sev) | Grado 4 (Prolif) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train (80% de 330 Original)** | 2,112 | **264** | 98 (37.1%) | 166 (62.9%) | 89 | 9 | 90 | 46 | 30 |
| **Val (20% de 330 Original)** | 528 | **66** | 25 (37.9%) | 41 (62.1%) | 22 | 3 | 22 | 12 | 7 |
| **Test (Parte A Original)** | 664 | **83** | 31 (37.3%) | 52 (62.7%) | 23 | 8 | 24 | 16 | 12 |
| *Subtotal Carpeta `train/`* | *3,304* | ***413*** | *154 (37.3%)* | *259 (62.7%)* | *134* | *20* | *136* | *74* | *49* |
| **Test Oficial Externo (`test/`)** | 824 | **103** | 39 (37.9%) | 64 (62.1%) | 34 | 5 | 32 | 19 | 13 |
| **TOTAL GLOBAL DATASET IDRiD** | **4,128** | **516** | **193 (37.4%)** | **323 (62.6%)** | **168** | **25** | **168** | **93** | **62** |

---

## 2. Archivos Generados en el Repositorio

| Nombre del Archivo | Descripción | N° Imágenes Originales |
|---|---|:---:|
| [`enfoque_sin_aumento_oficial/data_splits/split_sin_aumento_80_20.csv`](../enfoque_sin_aumento_oficial/data_splits/split_sin_aumento_80_20.csv) | Conjunto de desarrollo (Train + Val) con asignación de split 80/20 | **330** |
| [`enfoque_sin_aumento_oficial/data_splits/train_sin_aumento_80_20.csv`](../enfoque_sin_aumento_oficial/data_splits/train_sin_aumento_80_20.csv) | Subconjunto exclusivo de entrenamiento | **264** |
| [`enfoque_sin_aumento_oficial/data_splits/val_sin_aumento_80_20.csv`](../enfoque_sin_aumento_oficial/data_splits/val_sin_aumento_80_20.csv) | Subconjunto exclusivo de validación interna | **66** |
| [`enfoque_sin_aumento_oficial/data_splits/test_sin_aumento_83.csv`](../enfoque_sin_aumento_oficial/data_splits/test_sin_aumento_83.csv) | Subconjunto de prueba independiente ciego (Parte A de los 7 pasos) | **83** |
| [`enfoque_sin_aumento_oficial/data_splits/dataset_completo_original_413.csv`](../enfoque_sin_aumento_oficial/data_splits/dataset_completo_original_413.csv) | Consolidado de las 413 originales de la carpeta `train/` | **413** |
| [`enfoque_sin_aumento_oficial/data_splits/test_oficial_103_sin_aumento.csv`](../enfoque_sin_aumento_oficial/data_splits/test_oficial_103_sin_aumento.csv) | Las 103 imágenes originales de la carpeta `test/` oficial IDRiD | **103** |
| [`enfoque_sin_aumento_oficial/data_splits/dataset_completo_original_516.csv`](../enfoque_sin_aumento_oficial/data_splits/dataset_completo_original_516.csv) | **Censo Global Completo de las 516 imágenes originales del dataset IDRiD** | **516** |
| [`enfoque_sin_aumento_oficial/data_splits/split_sin_aumento_80_20.joblib`](../enfoque_sin_aumento_oficial/data_splits/split_sin_aumento_80_20.joblib) | Diccionario serializado con DataFrames y metadatos estadísticos | **80/20** |

---

## 3. Desempeño de EfficientNet-B0 en Ambos Conjuntos de Prueba (Sin Aumento)

- **Sobre el Test de 83 imágenes (Parte A - Guía de 7 pasos):**
  - ROC-AUC = **0.9950** [IC 95%: 0.9840 - 1.0000], Accuracy = **92.77%**, Sensibilidad = **92.31%**, Especificidad = **93.55%**, F1 = **0.9412**.
- **Sobre el Test Oficial Externo de 103 imágenes (`test/` carpeta física):**
  - ROC-AUC = **0.8906**, Accuracy = **83.50%**, Sensibilidad = **89.06%**, Especificidad = **74.36%**, F1 = **0.8702** (TN=29, FP=10, FN=7, TP=57).
