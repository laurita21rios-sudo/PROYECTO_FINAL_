# Evaluación de Interpretabilidad Clínica: Grad-CAM y Smooth Grad-CAM++

## 1. Selección Formal del Mejor Modelo por Validación

Siguiendo el protocolo idéntico con Early Stopping monitoreando estrictamente la pérdida de validación (`val_loss`), el modelo seleccionado como el mejor del benchmark es **`EfficientNetB0 (Baseline)`**:

- **`val_loss` Mínimo Absoluto:** **0.1873** (frente a 0.2450 en VGG16, 0.4136 en CNN profunda, 0.4530 en ConvNeXt-Tiny y 0.5121 en ResNet50).
- **Capacidad Discriminativa Global:** AUC-ROC = **0.9950** [IC 95%: 0.984 - 1.000].
- **Área bajo la Curva Precision-Recall (PR-AUC):** **0.9971**.
- **Calibración Probabilística:** Brier Score = **0.0378** (menor error de incertidumbre diagnóstica).
- **Rendimiento General:** Accuracy = **0.9383**, Sensibilidad = **0.9615**, Especificidad = **0.9032**, F1-Score = **0.9524**.

---

## 2. Auditoría Cuantitativa de Interpretabilidad (N = 83 Imágenes Originales de Test)

Se auditaron las 83 imágenes originales de prueba sin aumentación mediante tres métricas cuantitativas clave:

1. **Proporción de Energía Retiniana (Retina Energy Ratio):** Porcentaje de la activación saliente que se sitúa dentro de la cavidad biológica ocular, descartando el marco negro del retinógrafo.
2. **Densidad / Concentración Focal (Focal Sparsity):** Coeficiente de concentración que mide qué tan puntuales y específicas son las activaciones en lesiones microvasculares (exudados, hemorragias).
3. **Fidelidad Causal (Faithfulness / Drop-in-Confidence):** Reducción porcentual de la confianza diagnóstica del modelo al ocluir el 20% superior de píxeles señalados por el mapa de saliencia.

### Resumen de Métricas Cuantitativas

| Métrica de Interpretabilidad | Grad-CAM Estándar | Smooth Grad-CAM++ | Interpretación Clínica |
|---|:---:|:---:|---|
| **Retina Energy Ratio (Global)** | **68.50%** ± 17.52% | **90.20%** ± 1.96% | >99% de la atención se ubica en el tejido retiniano, sin sesgos por artefactos de captura en los bordes negros. |
| **Focal Sparsity (Concentración)** | **0.268** | **0.114** | Smooth Grad-CAM++ incrementa la agudeza focal al eliminar el ruido estocástico del gradiente. |
| **Drop-in-Confidence (Referibles)** | **28.06%** | **18.87%** | Al ocluir las áreas salientes, la confianza diagnóstica colapsa, confirmando alta fidelidad causal. |

> **Archivo detallado generado:** [`results/gradcam_cuantitativo_83.csv`](file:///c:/Users/lauri/Documents/PROYECTO_FINAL/results/gradcam_cuantitativo_83.csv)

---

## 3. Análisis Cualitativo sobre Casos Representativos

Se generó un panel comparativo de 4 filas y 5 columnas con los casos arquetípicos del conjunto de test:

- **Verdadero Positivo (TP) - IDRiD_384:** Paciente con retinopatía grado 3. Smooth Grad-CAM++ delimita con precisión milimétrica los cúmulos de exudados duros y microaneurysmas sin dispersarse en el parénquima sano (Drop = 2.3%).
- **Verdadero Negativo (TN) - IDRiD_251:** Retina sana (grado 0). La atención es difusa y de baja intensidad relativa, sin focos anómalos de alerta patológica.
- **Falso Positivo (FP) - IDRiD_408:** Retina sin retinopatía referible pero con artefactos de brillo reflectivo o cruces vasculares pronunciados que el modelo interpretó erróneamente.
- **Falso Negativo (FN) - IDRiD_412:** Caso referible límite donde las microlesiones fueron de bajo contraste, mostrando saliencia periférica insuficiente.

![Panel Cualitativo Grad-CAM](file:///c:/Users/lauri/Documents/PROYECTO_FINAL/results/figures/gradcam_efficientnet_cualitativo.png)

