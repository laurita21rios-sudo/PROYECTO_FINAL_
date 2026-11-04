# 👁️ Sistema de Detección de Retinopatía Diabética Referible (RDR) y Explicabilidad Clínica (XAI)
> **Dataset:** Indian Diabetic Retinopathy Image Dataset (IDRiD) — 413 Pacientes Únicos  
> **Enfoque Metodológico:** Estrictamente **Sin Aumento de Datos** (*Zero Data Augmentation*) — Imágenes originales puras con Split Agrupado y Estratificado 80/20.  
> **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM, CUDA 12.4)  
> **Framework:** TensorFlow / Keras & PyTorch

---

## 🗂️ Organización del Repositorio

El repositorio contiene **exclusivamente el enfoque oficial sin aumento de datos** y la documentación de sustentación académica:

```text
PROYECTO_FINAL/
├── 📁 enfoque_sin_aumento_oficial/     # 🌟 PIPELINE OFICIAL SIN AUMENTO DE DATOS (80/20)
│   ├── 📁 notebooks/                   # 6 Cuadernos Jupyter limpios (01 a 06) sin data augmentation
│   ├── 📁 data_splits/                 # Splits oficiales (264 train / 66 val / 83 test independientes)
│   ├── 📁 explicabilidad_xai/          # Grad-CAM, Smooth Grad-CAM++, panel cualitativo y métricas
│   ├── 📁 scripts/                     # Scripts de partición limpia y generación de mapas de calor
│   └── 📄 README.md                    # Descripción metodológica y clínica
│
├── 📁 documentacion_y_sustentacion/    # 📚 DEFENSA Y SUSTENTACIÓN ACADÉMICA (7 PASOS)
│   ├── 📄 informe_completo_7_pasos.md  # Informe integral con los 7 puntos de evaluación
│   ├── 📄 material_apoyo_sustentacion.md # Guía para el jurado con preguntas y respuestas clínicas
│   ├── 📄 split_sin_aumento_resumen.md # Censo y validación estadística del split 80/20
│   ├── 📄 discussion.md                # Discusión técnica de trade-offs, DeLong y McNemar
│   ├── 🖼️ comparativa_modelos_benchmark.png # Figura comparativa para diapositivas
│   └── 📄 README.md                    # Índice documental
│
└── 📁 data/                            # Directorio local de imágenes IDRiD (excluido de git por tamaño)
```

---

## 🌟 1. Enfoque Oficial Sin Aumento de Datos (`enfoque_sin_aumento_oficial/`)

### A. Partición 80/20 Pura (Cero Data Leakage y Cero Aumento)
Cada paciente aparece exactamente **una sola vez** en el estudio (cero réplicas sintéticas ni distorsiones de la anatomía ocular):
- **Entrenamiento (Train):** 264 imágenes originales (166 Referibles / 98 No Referibles).
- **Validación Interna (Val):** 66 imágenes originales (41 Referibles / 25 No Referibles).
- **Prueba Independiente (Test Parte A):** 83 imágenes originales (52 Referibles / 31 No Referibles).
- **Total:** 413 pacientes únicos.

### B. Cuadernos Jupyter Reproducibles (`enfoque_sin_aumento_oficial/notebooks/`)
1. [`01_EfficientNetB0_Sin_Aumento.ipynb`](enfoque_sin_aumento_oficial/notebooks/01_EfficientNetB0_Sin_Aumento.ipynb) — **Modelo Ganador Base**
2. [`02_ResNet50_Sin_Aumento.ipynb`](enfoque_sin_aumento_oficial/notebooks/02_ResNet50_Sin_Aumento.ipynb) — Comparativa Base
3. [`03_VGG16_Sin_Aumento.ipynb`](enfoque_sin_aumento_oficial/notebooks/03_VGG16_Sin_Aumento.ipynb) — Comparativa Base
4. [`04_ConvNeXt_Tiny_Sin_Aumento.ipynb`](enfoque_sin_aumento_oficial/notebooks/04_ConvNeXt_Tiny_Sin_Aumento.ipynb) — Comparativa Base
5. [`05_CNN_Desde_Cero_Sin_Aumento.ipynb`](enfoque_sin_aumento_oficial/notebooks/05_CNN_Desde_Cero_Sin_Aumento.ipynb) — Línea Base sin pre-entrenamiento
6. [`06_EfficientNetB0_CBAM_Residual_Sin_Aumento.ipynb`](enfoque_sin_aumento_oficial/notebooks/06_EfficientNetB0_CBAM_Residual_Sin_Aumento.ipynb) — Atención CBAM Residual con escalares aprendibles $\gamma_c$ y $\gamma_s$
7. [`07_CapsuleNet_Sin_Aumento.ipynb`](enfoque_sin_aumento_oficial/notebooks/07_CapsuleNet_Sin_Aumento.ipynb) — Red de Cápsulas con *Dynamic Routing* (3 iteraciones)

### C. Mejor Modelo Seleccionado por Validación: **EfficientNetB0 (Baseline)**
- **Pérdida de Validación Mínima:** `val_loss = 0.1873` (menor entre los 5 modelos).
- **Capacidad Discriminativa:** AUC-ROC = **0.9950** [IC 95%: 0.984 - 1.000], PR-AUC = **0.9971**.
- **Desempeño en Test Ciego (N = 83):** Accuracy = **92.77%**, Sensibilidad = **92.31%**, Especificidad = **93.55%**, F1-Score = **0.9412**.
- **Pruebas Estadísticas Pareadas (Test N = 83):**
  - **Prueba de DeLong:** $Z = 8.9860, p < 0.0001$.
  - **Prueba de McNemar:** $\chi^2 = 35.5577, p = 2.48 \times 10^{-9}$.

### D. Explicabilidad Clínica (XAI): Grad-CAM vs. Smooth Grad-CAM++
- **Retina Energy Ratio:** En Smooth Grad-CAM++, el **90.20%** de la atención recae estrictamente dentro de la cavidad retiniana biológica, eliminando sesgos en los bordes negros del retinógrafo.
- **Fidelidad Causal (Drop-in-Confidence):** La confianza diagnóstica colapsa en promedio un **28.06%** (y hasta un **99.41%**) al ocluir las áreas focales patológicas identificadas.
- **Panel Cualitativo:** 4 filas (TP, TN, FP, FN) $\times$ 5 columnas en alta resolución ubicado en [`enfoque_sin_aumento_oficial/explicabilidad_xai/gradcam_efficientnet_cualitativo.png`](enfoque_sin_aumento_oficial/explicabilidad_xai/gradcam_efficientnet_cualitativo.png).

---

## 📚 2. Material de Sustentación (`documentacion_y_sustentacion/`)
Diseñado para la preparación de la defensa ante el jurado calificador:
- [`informe_completo_7_pasos.md`](documentacion_y_sustentacion/informe_completo_7_pasos.md): Desarrollo punto por punto de los 7 requisitos metodológicos.
- [`material_apoyo_sustentacion.md`](documentacion_y_sustentacion/material_apoyo_sustentacion.md): Preguntas y respuestas clínicas para la defensa.
