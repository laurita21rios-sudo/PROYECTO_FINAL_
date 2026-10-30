# Benchmark Consolidado de Modelos - Detección de Retinopatía Diabética (IDRiD)

| Modelo | Umbral | Sensibilidad | Especificidad | Accuracy | F1-Score | AUC-ROC | PR-AUC | Propósito / Diagnóstico Clínico |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **ConvNeXt-Tiny (Clinical)** | 0.379 | 1.0000 | 0.6774 | 0.8795 | 0.9123 | 0.9870 | 0.9929 | Screening clínico: Sensibilidad 100% (FN=0) |
| **ResNet50 (Fine-Tuning Opt)** | 0.632 | 0.9423 | 0.9355 | 0.9398 | 0.9515 | 0.9876 | 0.9931 | Mayor F1-Score (0.9515) y balance general |
| **ResNet50 (Baseline 0.5)** | 0.500 | 0.9615 | 0.8710 | 0.9277 | 0.9434 | 0.9876 | 0.9931 | Umbral estándar 0.5 |
| **CNN Personalizada (Custom Deep)** | 0.500 | 0.8077 | 0.8387 | 0.8193 | 0.8485 | 0.8958 | 0.9336 | Baseline con explicabilidad Grad-CAM auditada |
| **CapsuleNet** | 0.500 | 0.8846 | 0.7742 | 0.8434 | 0.8762 | - | - | Red de cápsulas para relaciones espaciales |
| **EfficientNetB0 + CBAM** | 0.500 | 0.4231 | 0.1613 | 0.3253 | 0.4400 | 0.1948 | - | Colapso/bajo desempeño de atención CBAM |
