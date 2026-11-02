# Tabla Comparativa de Rendimiento: Validación vs. Test con IC 95%
> **Protocolo idéntico para los 5 modelos:**
> - **Partición:** Split agrupado por paciente y estratificado por severidad de retinopatía.
> - **Criterio de parada temprana (Early Stopping):** Monitoreo estricto de `val_loss` (pérdida en validación, con restauración de mejores pesos).
> - **Punto de corte estándar:** $\tau = 0.5$.
> - **Intervalos de Confianza al 95% [IC 95%]:** Calculados empíricamente mediante Bootstrap (1,000 remuestreos con reemplazo).

## 1. Conjunto de Validación Interna (Época Óptima por `val_loss`)

| Modelo | val_loss | Accuracy [IC 95%] | Sensibilidad [IC 95%] | Especificidad [IC 95%] | F1-Score [IC 95%] | AUC-ROC [IC 95%] | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ConvNeXt-Tiny (Clinical)** | 0.4530 | 0.8537 [0.756, 0.951] | 0.9615 [0.885, 1.000] | 0.8000 [0.600, 1.000] | 0.8929 [0.800, 0.966] | 0.9090 [0.812, 0.985] | 0.9356 |
| **EfficientNetB0 (Baseline)** | 0.1873 | 0.9383 [0.879, 0.985] | 0.9615 [0.904, 1.000] | 0.9032 [0.774, 1.000] | 0.9524 [0.909, 0.989] | 0.9950 [0.984, 1.000] | 0.9971 |
| **ResNet50 (Fine-Tuning)** | 0.5121 | 0.9277 [0.867, 0.976] | 0.9615 [0.904, 1.000] | 0.8710 [0.742, 0.968] | 0.9434 [0.898, 0.981] | 0.9870 [0.965, 0.999] | 0.9931 |
| **VGG16 (Fine-Tuned)** | 0.2450 | 0.9518 [0.904, 0.988] | 0.9615 [0.904, 1.000] | 0.9355 [0.839, 1.000] | 0.9615 [0.923, 0.990] | 0.9895 [0.970, 0.999] | 0.9944 |
| **CNN Personalizada (Deep)** | 0.4136 | 0.8193 [0.735, 0.904] | 0.8077 [0.692, 0.904] | 0.8387 [0.710, 0.968] | 0.8485 [0.769, 0.919] | 0.8955 [0.815, 0.958] | 0.9336 |

## 2. Conjunto de Test Independiente (N = 83 Originales IDRiD)

| Modelo | Accuracy [IC 95%] | Sensibilidad [IC 95%] | Especificidad [IC 95%] | F1-Score [IC 95%] | AUC-ROC [IC 95%] | PR-AUC | Brier |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ConvNeXt-Tiny (Clinical)** | 0.9157 [0.843, 0.976] | 0.9808 [0.933, 1.000] | 0.8065 [0.643, 0.939] | 0.9358 [0.882, 0.980] | 0.9870 [0.965, 0.999] | 0.9928 | 0.0627 |
| **EfficientNetB0 (Baseline)** | 0.9518 [0.904, 0.988] | 0.9615 [0.896, 1.000] | 0.9355 [0.821, 1.000] | 0.9615 [0.917, 0.991] | 0.9950 [0.984, 1.000] | 0.9971 | 0.0378 |
| **ResNet50 (Fine-Tuning)** | 0.9277 [0.867, 0.976] | 0.9615 [0.900, 1.000] | 0.8710 [0.750, 0.971] | 0.9434 [0.891, 0.982] | 0.9876 [0.967, 0.999] | 0.9931 | 0.0526 |
| **VGG16 (Fine-Tuned)** | 0.9518 [0.904, 0.988] | 0.9615 [0.904, 1.000] | 0.9355 [0.839, 1.000] | 0.9615 [0.923, 0.990] | 0.9895 [0.970, 0.999] | 0.9944 | 0.0482 |
| **CNN Personalizada (Deep)** | 0.8193 [0.735, 0.904] | 0.8077 [0.702, 0.911] | 0.8387 [0.703, 0.964] | 0.8485 [0.769, 0.917] | 0.8958 [0.813, 0.958] | 0.9336 | 0.1272 |
