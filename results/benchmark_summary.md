# Benchmark Consolidado — Retinopatía Diabética (IDRiD)
> Todos los modelos fueron evaluados rigurosamente sobre las **83 imágenes originales** de pacientes únicos (sin data augmentation en validación/test).

| Modelo | N | Umbral (τ) | Sensibilidad | Especificidad | Accuracy | F1-Score | AUC-ROC | PR-AUC | Brier Score | CM (TN/FP/FN/TP) |
|---|---|---|---|---|---|---|---|---|---|---|
| **ConvNeXt-Tiny (Clinical)** | 83 | 0.379 | 1.0000 | 0.6774 | 0.8795 | 0.9123 | 0.9870 | 0.9929 | 0.0627 | 21/10/0/52 |
| **ResNet50 (Fine-Tuning Opt)** | 83 | 0.632 | 0.9423 | 0.9355 | 0.9398 | 0.9515 | 0.9876 | 0.9931 | 0.0526 | 29/2/3/49 |
| **ResNet50 (Baseline 0.5)** | 83 | 0.500 | 0.9615 | 0.8710 | 0.9277 | 0.9434 | 0.9876 | 0.9931 | 0.0526 | 27/4/2/50 |
| **EfficientNetB0 (Baseline, N=83)** | 83 | 0.500 | 0.9615 | 0.9355 | 0.9518 | 0.9615 | 0.9950 | 0.9971 | 0.0378 | 29/2/2/50 |
| **CNN Personalizada (Custom Deep)** | 83 | 0.500 | 0.8077 | 0.8387 | 0.8193 | 0.8485 | 0.8958 | 0.9336 | 0.1272 | 26/5/10/42 |
| **CapsuleNet** | 83 | 0.500 | 0.8846 | 0.7742 | 0.8434 | 0.8762 | 0.9107 | 0.9475 | 0.1229 | 24/7/6/46 |
| **EfficientNetB0 + CBAM** | 83 | 0.500 | 0.4038 | 0.3871 | 0.3976 | 0.4565 | 0.3679 | 0.5416 | 0.2522 | 12/19/31/21 |
| **VGG16 (Fine-Tuned)** | 83 | 0.500 | 0.9615 | 0.9355 | 0.9518 | 0.9615 | 0.9895 | 0.9944 | - | 29/2/2/50 |
