# 📦 Historial de Experimentos con Aumento de Datos (Data Augmentation)

> **Estado:** Archivo Histórico y Comparativo.  
> **Contenido:** Todos los modelos, benchmarks, notebooks y datos aumentados generados durante las fases exploratorias del proyecto.

---

## 1. Descripción de la Fase Experimental

En esta etapa inicial del proyecto, se exploró el uso de técnicas de aumento de datos (*Data Augmentation*) con réplicas geométricas y fotométricas ($\times 8$), generando conjuntos aumentados para contrarrestar el tamaño muestral:
- **Entrenamiento:** 264 pacientes originales $\times 8$ réplicas = **2,112 imágenes aumentadas** (o 2,376 imágenes en 90/10).
- **Validación y Test:** Se mantuvieron siempre en formato original para evitar inflación de métricas.

---

## 2. Contenido de este Módulo

- **📁 `modelos_experimentales/`:**
  - `EfficientNetB0_Baseline/`: Checkpoints, métricas y dashboards del modelo ganador.
  - `ResNet50/`: Fine-tuning de ResNet50 con transfer learning.
  - `ConvNeXt_Tiny_Clinical/`: Arquitectura moderna ConvNeXt adaptada a oftalmología.
  - `CNN_Personalizada/`: Arquitectura convolucional propia de 4 bloques.
  - `CNN_Sin_Aumento_RDR/`: Experimento preliminar de CNN sin aumentación.
  - `EfficientNetB0_CBAM/`: EfficientNet con mecanismo de atención por canales y espacial (CBAM).
  - `modelos_guardados/`: Checkpoints serializados y pesos (.pth, .keras).
- **📁 `data_splits_con_aumento/`:**
  - Archivos CSV y Joblib con las 2,112 y 2,376 imágenes aumentadas por paciente.
- **📁 `benchmarks_y_predicciones/`:**
  - `benchmark_val_test_ic95.md` y `.json`: Comparativa de los 5 modelos con Intervalos de Confianza al 95% (Bootstrap 1,000 iteraciones).
  - `benchmark_summary.md` y `.json`: Resúmenes consolidados previos.
  - `gpu_predictions/`: Predicciones directas generadas con la GPU NVIDIA RTX 4050.
  - `figuras_anteriores/`: Gráficos de entrenamiento, matrices de confusión y curvas ROC históricas.
- **📁 `scripts_anteriores/`:**
  - Scripts utilizados para el entrenamiento unificado, bootstrap e inferencia por GPU.
- **📁 `notebooks/`:**
  - Cuadernos Jupyter reproducibles de cada arquitectura individual (ResNet50, EfficientNetB0, VGG16, ConvNeXt, CNN).
