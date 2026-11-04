# Partición Oficial del Dataset IDRiD Sin Aumento de Datos

> **Principio de Evaluación Clínica Rigurosa:**
> - **Cero Aumento de Datos:** No se aplican rotaciones, giros horizontales (*flips*), traslaciones ni copias sintéticas en ningún subconjunto.
> - **Preservación Anatómica:** Se mantiene la orientación espacial fisiológica real de la retina (papila óptica nasal vs. fóvea temporal).
> - **Partición a Nivel de Paciente (Patient-Level Split):** Cada uno de los 413 pacientes del estudio aparece exactamente **una sola vez**.

## 1. Distribución Numérica y Clínica de Pacientes (Split 80/20)

| Partición | Total Imágenes | No Referible (0) | Referible (1) | Grado 0 (Normal) | Grado 1 (Leve) | Grado 2 (Mod) | Grado 3 (Sev) | Grado 4 (Prolif) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train (80% Original)** | 264 | 98 (37.1%) | 166 (62.9%) | 89 | 9 | 90 | 46 | 30 |
| **Val (20% Original)** | 66 | 25 (37.9%) | 41 (62.1%) | 22 | 3 | 22 | 12 | 7 |
| **Test (Parte A Original)** | 83 | 31 (37.3%) | 52 (62.7%) | 23 | 8 | 24 | 16 | 12 |
| **Total Estudio Clínico** | 413 | 154 (37.3%) | 259 (62.7%) | 134 | 20 | 136 | 74 | 49 |

---

## 2. Archivos Generados en el Repositorio

| Nombre del Archivo | Descripción | N° Imágenes / Filas |
|---|---|:---:|
| [`data/splits/split_sin_aumento_80_20.csv`](file:///c:/Users/lauri/Documents/PROYECTO_FINAL/data/splits/split_sin_aumento_80_20.csv) | Conjunto de desarrollo (Train + Val) con asignación de split | 330 |
| [`data/splits/train_sin_aumento_80_20.csv`](file:///c:/Users/lauri/Documents/PROYECTO_FINAL/data/splits/train_sin_aumento_80_20.csv) | Subconjunto exclusivo de entrenamiento | 264 |
| [`data/splits/val_sin_aumento_80_20.csv`](file:///c:/Users/lauri/Documents/PROYECTO_FINAL/data/splits/val_sin_aumento_80_20.csv) | Subconjunto exclusivo de validación interna | 66 |
| [`data/splits/test_sin_aumento_83.csv`](file:///c:/Users/lauri/Documents/PROYECTO_FINAL/data/splits/test_sin_aumento_83.csv) | Subconjunto de prueba independiente ciego (Parte A) | 83 |
| [`data/splits/dataset_completo_original_413.csv`](file:///c:/Users/lauri/Documents/PROYECTO_FINAL/data/splits/dataset_completo_original_413.csv) | Dataset consolidado de los 413 pacientes con rutas absolutas | 413 |
| [`data/splits/split_sin_aumento_80_20.joblib`](file:///c:/Users/lauri/Documents/PROYECTO_FINAL/data/splits/split_sin_aumento_80_20.joblib) | Diccionario serializado con DataFrames y metadatos estadísticos | 80/20 |
| [`data/splits/split_sin_aumento_90_10.joblib`](file:///c:/Users/lauri/Documents/PROYECTO_FINAL/data/splits/split_sin_aumento_90_10.joblib) | Partición alternativa 90/10 para comparativas | 90/10 |

---

## 3. Garantías Metodológicas

1. **Cero Data Leakage:** Ningún paciente de entrenamiento o validación se encuentra en el conjunto de prueba independiente.
2. **Estratificación Multiclase Exacta:** La estratificación se ejecutó sobre la variable `Retinopathy grade` (escala 0 a 4 del ICDR), lo que garantiza automáticamente una proporción idéntica de la variable binaria `binary_label` (No Referible vs Referible).
3. **Rutas Físicas Verificadas:** El 100% de las 413 imágenes han sido verificadas en el almacenamiento local y enlazadas en la columna `file_path`.

