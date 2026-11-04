# 🔬 Enfoque Oficial Sin Aumento de Datos (Raw Fundus 80/20)

> **Estado:** Rama Principal de Producción y Entrega Oficial.  
> **Filosofía:** Cero manipulación sintética de la anatomía ocular. Pacientes únicos, evaluación reproducible y explicabilidad clínica.

---

## 1. Justificación Médica y Metodológica

En este enfoque, **no se aplicó ninguna técnica de data augmentation** (ni rotaciones, ni giros horizontales *flips*, ni variaciones de color o distorsiones geométricas).

1. **Preservación de la Anatomía Retiniana Fisiológica:**  
   En la exploración oftalmológica de fondo de ojo, la relación espacial entre la **papila óptica** (lado nasal) y la **mácula/fóvea** (lado temporal) determina la lateralidad y los cuadrantes clínicos de las lesiones. La inversión de ejes mediante *flips* desnaturaliza esta orientación.
2. **Cero Artefactos de Interpolación:**  
   Al no realizar rotaciones ni traslaciones con remuestreo bilineal, las lesiones microvasculares diminutas (microaneurismas y microhemorragias de 2–5 píxeles) no sufren degradación de bordes.
3. **Cero Data Leakage:**  
   Partición estricta a nivel de paciente (*Patient-Level Split*). Cada uno de los 413 pacientes del estudio IDRiD aparece exactamente **una sola vez**.

---

## 2. Partición Oficial del Dataset (Split 80/20 Puro)

| Subconjunto | N° Pacientes Únicos | No Referible (0) | Referible (1) | Grado 0 | Grado 1 | Grado 2 | Grado 3 | Grado 4 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Entrenamiento (Train)** | **264** | 98 (37.1%) | 166 (62.9%) | 89 | 9 | 90 | 46 | 30 |
| **Validación Interna (Val)** | **66** | 25 (37.9%) | 41 (62.1%) | 22 | 3 | 22 | 12 | 7 |
| **Prueba Independiente (Test Parte A)** | **83** | 31 (37.3%) | 52 (62.7%) | 23 | 8 | 24 | 16 | 12 |
| **Total Estudio Clínico** | **413** | **154 (37.3%)** | **259 (62.7%)** | **134** | **20** | **136** | **74** | **49** |

---

## 3. Estructura de este Módulo

- **📁 `data_splits/`:**
  - `split_sin_aumento_80_20.csv`: Censo de las 330 imágenes de desarrollo con columna de split.
  - `train_sin_aumento_80_20.csv`: 264 registros para entrenamiento.
  - `val_sin_aumento_80_20.csv`: 66 registros para validación.
  - `test_sin_aumento_83.csv`: 83 registros del conjunto ciego de prueba (Parte A).
  - `dataset_completo_original_413.csv`: Censo unificado con rutas absolutas locales (`file_path`).
  - `split_sin_aumento_80_20.joblib` / `90_10.joblib`: Serializaciones en Python.
- **📁 `scripts/`:**
  - `create_split_sin_aumento.py`: Generador y verificador estratificado del split puro.
  - `generate_gradcam_efficientnet.py`: Generador de mapas de interpretabilidad Grad-CAM y Smooth Grad-CAM++.
- **📁 `explicabilidad_xai/`:**
  - `gradcam_efficientnet_cualitativo.png`: Panel clínico de 4 filas $\times$ 5 columnas (TP, TN, FP, FN).
  - `gradcam_cuantitativo_83.csv`: Auditoría cuantitativa individual (Retina Energy Ratio, Focal Sparsity, Drop-in-Confidence).
  - `gradcam_analisis_cualitativo_cuantitativo.md`: Informe médico detallado de interpretabilidad.
