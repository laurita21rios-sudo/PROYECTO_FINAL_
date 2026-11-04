# 👁️ Sistema de Detección de Retinopatía Diabética Referible (RDR) y Explicabilidad Clínica (XAI)
> **Dataset:** Indian Diabetic Retinopathy Image Dataset (IDRiD) — 413 Pacientes Únicos  
> **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM, CUDA 12.4)  
> **Framework:** PyTorch & TensorFlow / Keras

---

## 🗂️ Organización del Repositorio

El proyecto se encuentra organizado de forma modular en tres grandes áreas de trabajo:

```text
PROYECTO_FINAL/
├── 📁 enfoque_sin_aumento_oficial/     # 🌟 ENFOQUE DEFINITIVO: Sin Data Augmentation (Imágenes puras 80/20)
│   ├── 📁 data_splits/                 # Splits oficiales (264 train / 66 val / 83 test independientes)
│   ├── 📁 explicabilidad_xai/          # Grad-CAM, Smooth Grad-CAM++, panel cualitativo y métricas
│   ├── 📁 scripts/                     # Scripts de partición limpia y generación de mapas de calor
│   └── 📄 README.md                    # Descripción metodológica y clínica
│
├── 📁 historial_con_aumento_previo/    # 📦 HISTORIAL EXPERIMENTAL: Modelos con aumento previo (x8)
│   ├── 📁 modelos_experimentales/      # EfficientNetB0, ResNet50, ConvNeXt, VGG16, CNN, CBAM
│   ├── 📁 data_splits_con_aumento/     # Splits previos con réplicas aumentadas (2,112 imágenes)
│   ├── 📁 benchmarks_y_predicciones/   # Tablas comparativas con IC 95% (Bootstrap), Brier y curvas
│   ├── 📁 scripts_anteriores/          # Pipeline de entrenamiento unificado y GPU
│   ├── 📁 notebooks/                   # Cuadernos Jupyter reproducibles
│   └── 📄 README.md                    # Guía de experimentos históricos
│
├── 📁 documentacion_y_sustentacion/    # 📚 DEFENSAS Y SUSTENTACIÓN ACADÉMICA
│   ├── 📄 material_apoyo_sustentacion.md # Guía para el jurado con preguntas y respuestas clínicas
│   ├── 📄 split_sin_aumento_resumen.md # Censo y validación estadística del split 80/20
│   ├── 📄 discussion.md                # Discusión técnica de trade-offs
│   ├── 🖼️ comparativa_modelos_benchmark.png # Figura para diapositivas
│   └── 📄 README.md                    # Índice documental
│
└── 📁 data/                            # Directorio local de imágenes IDRiD (excluido de git por tamaño)
```

---

## 🌟 1. Enfoque Oficial Sin Aumento de Datos (`enfoque_sin_aumento_oficial/`)

### A. Partición 80/20 Pura (Cero Data Leakage)
Cada paciente aparece exactamente **una sola vez** en el estudio (cero réplicas sintéticas ni distorsiones de la anatomía ocular):
- **Entrenamiento (Train):** 264 imágenes originales (166 Referibles / 98 No Referibles).
- **Validación Interna (Val):** 66 imágenes originales (41 Referibles / 25 No Referibles).
- **Prueba Independiente (Test Parte A):** 83 imágenes originales (52 Referibles / 31 No Referibles).
- **Total:** 413 pacientes únicos.

### B. Mejor Modelo Seleccionado por Validación: **EfficientNetB0 (Baseline)**
- **Pérdida de Validación Mínima:** `val_loss = 0.1873` (menor entre los 5 modelos).
- **Capacidad Discriminativa:** AUC-ROC = **0.9950** [IC 95%: 0.984 - 1.000], PR-AUC = **0.9971**.
- **Desempeño en Test Ciego (N = 83):** Accuracy = **92.77%**, Sensibilidad = **92.31%**, Especificidad = **93.55%**, F1-Score = **0.9412**.

### C. Explicabilidad Clínica (XAI): Grad-CAM vs. Smooth Grad-CAM++
- **Retina Energy Ratio:** En Smooth Grad-CAM++, el **90.20%** de la atención recae estrictamente dentro de la cavidad retiniana biológica, eliminando sesgos en los bordes negros del retinógrafo.
- **Fidelidad Causal (Drop-in-Confidence):** La confianza diagnóstica colapsa en promedio un **28.06%** (y hasta un **99.41%**) al ocluir las áreas focales patológicas identificadas.
- **Panel Cualitativo:** 4 filas (TP, TN, FP, FN) $\times$ 5 columnas en alta resolución ubicado en [`enfoque_sin_aumento_oficial/explicabilidad_xai/gradcam_efficientnet_cualitativo.png`](enfoque_sin_aumento_oficial/explicabilidad_xai/gradcam_efficientnet_cualitativo.png).

---

## 📦 2. Historial de Experimentos Previos (`historial_con_aumento_previo/`)
Conserva todos los cuadernos, modelos experimentales y métricas generadas con técnicas previas de aumentación de datos ($\times 8$), permitiendo comparar el impacto de la aumentación frente a las imágenes originales puras.

---

## 📚 3. Material de Sustentación (`documentacion_y_sustentacion/`)
Diseñado para la preparación de la defensa ante el jurado calificador:
- Respuestas sustentadas a preguntas difíciles sobre *data leakage*, calibración con Brier Score, selección de umbral clínico ($\tau = 0.50$), y justificación del split 80/20 vs 90/10.

---

## 🚀 Requisitos y Reproducibilidad

```bash
# Clonar el repositorio
git clone https://github.com/laurita21rios-sudo/PROYECTO_FINAL_.git
cd PROYECTO_FINAL_

# Sincronizar entorno virtual con uv
uv sync

# Ejecutar auditoría de interpretabilidad del enfoque oficial
python enfoque_sin_aumento_oficial/scripts/generate_gradcam_efficientnet.py
```
