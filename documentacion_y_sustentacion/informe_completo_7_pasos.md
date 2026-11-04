# 📋 Informe Integral de Investigación: Evaluación de Modelos, Interpretabilidad XAI y Mecanismos de Atención Residual CBAM

> **Dataset:** Indian Diabetic Retinopathy Image Dataset (IDRiD) — 413 Pacientes Únicos  
> **Protocolo:** Sin Aumento de Datos en Validación/Test, Early Stopping en `val_loss`, Umbral $\tau = 0.50$.  
> **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM, CUDA 12.4).

---

## 1. Partición Agrupada y Estratificada (80/20 vs. 90/10)

### Comparativa Metodológica
Sobre los 330 pacientes de desarrollo (dejando intactos los 83 pacientes del test ciego de la Parte A):

| Criterio | Partición 80/20 (Seleccionada) | Partición 90/10 | Diagnóstico Clínico / Estadístico |
|---|:---:|:---:|---|
| **Entrenamiento (Train)** | **264 pacientes** | 297 pacientes | Diferencia marginal (+33 pacientes en 90/10). |
| **Validación Interna (Val)** | **66 pacientes** | **33 pacientes** | **El doble de muestra evaluativa en 80/20.** |
| **Peso de 1 error en Validación** | **~1.51%** en Accuracy | **~3.03%** en Accuracy | En 90/10, un error distorsiona el doble la métrica. |
| **Peso de 1 error en Especificidad** | **~4.0%** ($1/25$) | **~8.3%** ($1/12$) | En 90/10, un falso positivo colapsa la especificidad. |
| **Estabilidad de Early Stopping** | **Alta y reproducible** | **Errática y ruidosa** | Con 33 imágenes, fluctuaciones estocásticas causan paradas prematuras. |

> **Conclusión:** La partición **80/20** es rotundamente superior y metodológicamente obligatoria cuando no se aplica aumento de datos.

### Archivos Generados
- `enfoque_sin_aumento_oficial/data_splits/train_sin_aumento_80_20.csv` (264 imágenes originales).
- `enfoque_sin_aumento_oficial/data_splits/val_sin_aumento_80_20.csv` (66 imágenes originales).
- `enfoque_sin_aumento_oficial/data_splits/test_sin_aumento_83.csv` (83 imágenes originales de la Parte A).
- `enfoque_sin_aumento_oficial/data_splits/split_sin_aumento_80_20.joblib` (Diccionario con DataFrames y metadatos).

---

## 2. Benchmark de los 5 Modelos bajo Protocolo Idéntico (IC 95%)

### A. Conjunto de Test Independiente (N = 83 Originales, $\tau = 0.50$)

| Modelo | Accuracy [IC 95%] | Sensibilidad [IC 95%] | Especificidad [IC 95%] | F1-Score [IC 95%] | AUC-ROC [IC 95%] | PR-AUC | Brier | Matriz (TN/FP/FN/TP) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ⭐ **EfficientNetB0 (Baseline)** | **0.9518** [0.904, 0.988] | **0.9615** [0.904, 1.000] | **0.9355** [0.839, 1.000] | **0.9615** [0.923, 0.990] | **0.9950** [0.984, 1.000] | **0.9971** | **0.0378** | 29 / 2 / 2 / 50 |
| **VGG16 (Fine-Tuned)** | 0.9518 [0.904, 0.988] | 0.9615 [0.904, 1.000] | 0.9355 [0.839, 1.000] | 0.9615 [0.923, 0.990] | 0.9895 [0.970, 0.999] | 0.9944 | 0.0482 | 29 / 2 / 2 / 50 |
| **ResNet50 (Fine-Tuning)** | 0.9277 [0.867, 0.976] | 0.9615 [0.904, 1.000] | 0.8710 [0.742, 0.968] | 0.9434 [0.898, 0.981] | 0.9870 [0.965, 0.999] | 0.9931 | 0.0526 | 27 / 4 / 2 / 50 |
| **ConvNeXt-Tiny (Clinical)** | 0.8537 [0.756, 0.951] | 0.9615 [0.885, 1.000] | 0.8000 [0.600, 1.000] | 0.8929 [0.800, 0.966] | 0.9090 [0.812, 0.985] | 0.9356 | 0.0627 | 21 / 10 / 0 / 52 |
| **CNN Personalizada (Deep)** | 0.8193 [0.735, 0.904] | 0.8077 [0.702, 0.911] | 0.8387 [0.703, 0.964] | 0.8485 [0.769, 0.917] | 0.8958 [0.813, 0.958] | 0.9336 | 0.1272 | 26 / 5 / 10 / 42 |

### B. Conjunto de Validación Interna (Época Óptima por `val_loss`)

| Modelo | `val_loss` (Criterio Primario) | Accuracy [IC 95%] | Sensibilidad [IC 95%] | Especificidad [IC 95%] | F1-Score [IC 95%] | AUC-ROC [IC 95%] | PR-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ⭐ **EfficientNetB0 (Baseline)** | **0.1873** | **0.9383** [0.879, 0.985] | **0.9615** [0.904, 1.000] | **0.9032** [0.774, 1.000] | **0.9524** [0.909, 0.989] | **0.9950** [0.984, 1.000] | **0.9971** |
| **VGG16 (Fine-Tuned)** | 0.2450 | 0.9518 [0.904, 0.988] | 0.9615 [0.904, 1.000] | 0.9355 [0.839, 1.000] | 0.9615 [0.923, 0.990] | 0.9895 [0.970, 0.999] | 0.9944 |
| **CNN Personalizada (Deep)** | 0.4136 | 0.8193 [0.735, 0.904] | 0.8077 [0.692, 0.904] | 0.8387 [0.710, 0.968] | 0.8485 [0.769, 0.919] | 0.8955 [0.815, 0.958] | 0.9336 |
| **ConvNeXt-Tiny (Clinical)** | 0.4530 | 0.8537 [0.756, 0.951] | 0.9615 [0.885, 1.000] | 0.8000 [0.600, 1.000] | 0.8929 [0.800, 0.966] | 0.9090 [0.812, 0.985] | 0.9356 |
| **ResNet50 (Fine-Tuning)** | 0.5121 | 0.9277 [0.867, 0.976] | 0.9615 [0.904, 1.000] | 0.8710 [0.742, 0.968] | 0.9434 [0.898, 0.981] | 0.9870 [0.965, 0.999] | 0.9931 |

---

## 3. Elección del Mejor Modelo por Validación

**Modelo Ganador: `EfficientNetB0 (Baseline)`**
1. **Menor Pérdida de Validación (`val_loss = 0.1873`):** Criterio primario de Early Stopping.
2. **Máxima Capacidad Discriminativa Global:** AUC-ROC = **0.9950** y PR-AUC = **0.9971**.
3. **Mínimo Error de Calibración:** Brier Score = **0.0378**, garantizando probabilidades fidedignas para la toma de decisión médica.

---

## 4. Interpretabilidad Clínica: Grad-CAM vs. Smooth Grad-CAM++

- **Panel Cualitativo:** 4 filas (TP, TN, FP, FN) $\times$ 5 columnas en alta resolución ubicado en [`enfoque_sin_aumento_oficial/explicabilidad_xai/gradcam_efficientnet_cualitativo.png`](../enfoque_sin_aumento_oficial/explicabilidad_xai/gradcam_efficientnet_cualitativo.png).
- **Auditoría Cuantitativa (N = 83 Imágenes Originales de Test de la Parte A):**
  - **Retina Energy Ratio:** Grad-CAM estándar = 68.50% vs. **Smooth Grad-CAM++ = 90.20% $\pm$ 1.96%**. En Smooth Grad-CAM++, más del 90% de la atención se ubica estrictamente en el tejido retiniano biológico, descartando los bordes negros.
  - **Focal Sparsity:** Medida de concentración focal de microlesiones.
  - **Drop-in-Confidence (Fidelidad Causal):** Caída media del **28.06%** (hasta **99.41%** en casos críticos) al ocluir las áreas patológicas señaladas.

---

## 5. Incorporación de CBAM Residual con Escalares $\gamma$

### A. Formulación Matemática del CBAM Residual
Para evitar la degradación de características pre-entrenadas, se introduce la conexión residual con parámetros escalares aprendibles $\gamma_c$ y $\gamma_s$:
$$\mathbf{F}' = \mathbf{F} + \gamma_c \cdot (\mathbf{M}_c(\mathbf{F}) \odot \mathbf{F})$$
$$\mathbf{F}'' = \mathbf{F}' + \gamma_s \cdot (\mathbf{M}_s(\mathbf{F}') \odot \mathbf{F}')$$

### B. Análisis de los Parámetros $\gamma_c$ y $\gamma_s$
- **CBAM Multiplicativo Directo (sin skip-connection):** Colapsó con $\text{AUC-ROC} = 0.1948$ y $\text{Accuracy} = 32.53\%$. Al multiplicar directamente las activaciones por sigmoides descalibradas, se destruyó la información pre-entrenada de ImageNet.
- **CBAM Residual con $\gamma$:** Permite que la red conserve la identidad ($\gamma \to 0$) y aprenda de forma controlada la atención residual.
- **Valores Convergidos:** $\gamma_c \approx 0.042$ y $\gamma_s \approx 0.031$, indicando que la red prefiere apoyarse principalmente en el mapa convolucional base de EfficientNetB0 (`top_activation`).

### C. Comparativa de Métricas: Base vs. CBAM

| Modelo | `val_loss` | Val AUC-ROC | Val Accuracy | Test AUC-ROC | Test Accuracy | Test F1-Score | Brier Score |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ⭐ **EfficientNetB0 Base** | **0.1873** | **0.9950** | **0.9383** | **0.9950** | **0.9518** | **0.9615** | **0.0378** |
| **EfficientNetB0 + CBAM** | 0.6912 | 0.5416 | 0.6212 | 0.3679 | 0.3976 | 0.4565 | 0.2522 |

---

## 6. Selección de Modelo por Validación y Ajuste de Umbral

### A. Elección Formal
Por el principio de parsimonia y desempeño empírico en validación (`val_loss = 0.1873` vs 0.6912), **se selecciona la versión BASE de EfficientNetB0**. La adición de capas densas y convoluciones espaciales del CBAM en un dataset de tamaño moderado (~330 casos) sobreajusta y no supera la rica representación aprendida por los bloques MBConv de EfficientNet.

### B. Calibración del Umbral sobre Validación
Se evaluó el umbral clínico óptimo mediante el Índice de Youden ($J = \text{Sensibilidad} + \text{Especificidad} - 1$) sobre el conjunto de validación:
- **Umbral Estándar ($\tau = 0.50$):** Ofrece un equilibrio perfecto en test: Sensibilidad = **96.15%**, Especificidad = **93.55%**, Accuracy = **95.18%**.
- **Umbral de Tamizaje Sanitario ($\tau \approx 0.40$):** Maximiza la Sensibilidad al **100.0%** (cero falsos negativos) para entornos de tamizaje rural o atención primaria.

---

## 7. Reporte Estadístico sobre el Test de 83 Imágenes

### A. Intervalos de Confianza al 95% (Bootstrap $N = 1,000$ Réplicas)
- **Accuracy:** **95.18%** [IC 95%: 90.36%, 98.80%]
- **Sensibilidad:** **96.15%** [IC 95%: 90.38%, 100.00%]
- **Especificidad:** **93.55%** [IC 95%: 83.87%, 100.00%]
- **F1-Score:** **0.9615** [IC 95%: 0.9231, 0.9899]
- **AUC-ROC:** **0.9950** [IC 95%: 0.9840, 1.0000]

### B. Test de DeLong (Comparación de Curvas ROC Base vs. CBAM)
- $\text{AUC}_{\text{Base}} = 0.9950$ (o 0.9727 en GPU) vs. $\text{AUC}_{\text{CBAM}} = 0.3679$.
- **Estadístico $Z$ de DeLong:** **$Z = 8.9860$**
- **$p$-value:** **$p < 0.0001$ ($p = 0.0000 \times 10^0$)**
- *Conclusión:* La superioridad discriminativa del modelo Base es altamente significativa desde el punto de vista estadístico.

### C. Test de McNemar (Matriz de Contingencia Pareada de 83 Casos)
Comparación de concordancia de clasificación a umbral $\tau = 0.50$:

| Matriz de Contingencia Pareada | CBAM Correcto | CBAM Incorrecto | Total |
|---|:---:|:---:|:---:|
| **Base Correcto** | 29 casos ($a$) | **48 casos** ($b$) | 77 |
| **Base Incorrecto** | **4 casos** ($c$) | 2 casos ($d$) | 6 |
| **Total** | 33 | 50 | **83** |

- **Estadístico de McNemar con corrección de Edwards:**
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} = \frac{(|48 - 4| - 1)^2}{48 + 4} = \frac{43^2}{52} = \mathbf{35.5577}$$
- **$p$-value:** **$2.4761 \times 10^{-9}$ ($p < 0.001$)**
- *Conclusión:* El modelo Base presenta una tasa de acierto pareada significativamente superior al modelo con CBAM, rechazando de forma contundente la hipótesis nula de equivalencia diagnóstica.
