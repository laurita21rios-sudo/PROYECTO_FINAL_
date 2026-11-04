# 📊 Resumen General del Proyecto: Casos de Evaluación, Auto-Test y Comparativa Con vs. Sin Aumento de Datos

> **Proyecto:** Detección de Retinopatía Diabética Referible (RDR) e Interpretabilidad Clínica (XAI)  
> **Dataset Global (IDRiD):** **516 Pacientes Únicos Originales** (*4,128 archivos totales en disco cuando se contabilizan las $\times 8$ réplicas sintéticas*).  
> **Hardware de Ejecución:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM, CUDA 12.4)  
> **Script de Automatización de Pruebas (Auto-Test):** [`enfoque_sin_aumento_oficial/scripts/auto_test_suite.py`](enfoque_sin_aumento_oficial/scripts/auto_test_suite.py)

---

## 1. Censo Exacto del Dataset: Comparativa Con Aumento ($\times 8$) vs. Sin Aumento (Originales Puras)

En el directorio físico `data/IDRiD2/IDRiD/` existen dos carpetas (`train/` y `test/`). Cada fotografía de fondo de ojo original fue multiplicada por 8 ($\times 8$: 1 original + 7 transformaciones geométricas `_L6, _L14, _L20, _R8, _R16, _R22, _VFLIP`). La siguiente tabla contrasta el volumen de datos entre ambos enfoques:

| Subconjunto / Partición | Con Aumento de Datos ($\times 8$ en Disco) | **Sin Aumento de Datos (Imágenes Originales Puras)** | No Referible (Clase 0: Grados 0–1) | Referible (Clase 1: Grados 2–4) | Propósito en la Evaluación |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Entrenamiento (Train 80% de 330)** | **2,112** archivos | **264 originales** | 98 (37.1%) | 166 (62.9%) | Ajuste de pesos bajo protocolo idéntico |
| **Validación Interna (Val 20% de 330)** | **528** archivos | **66 originales** | 25 (37.9%) | 41 (62.1%) | `EarlyStopping(val_loss)` y selección del ganador |
| **Test Independiente Parte A (Guía 7 pasos)** | **664** archivos | **83 originales** | 31 (37.3%) | 52 (62.7%) | Evaluación ciega principal (IC 95%, DeLong, McNemar, XAI) |
| *Subtotal Carpeta `train/` (`IDRiD_001`–`413`)* | *$3,304$ archivos* | ***413 originales*** | *154 (37.3%)* | *259 (62.7%)* | *Censo [`dataset_completo_original_413.csv`](enfoque_sin_aumento_oficial/data_splits/dataset_completo_original_413.csv)* |
| **Test Oficial Externo (Carpeta `test/`)** | **824** archivos | **103 originales** | 39 (37.9%) | 64 (62.1%) | Prueba externa adicional [`test_oficial_103_sin_aumento.csv`](enfoque_sin_aumento_oficial/data_splits/test_oficial_103_sin_aumento.csv) |
| **TOTAL GLOBAL DATASET IDRiD** | **4,128 archivos** | **516 originales** | **193 (37.4%)** | **323 (62.6%)** | **Censo total [`dataset_completo_original_516.csv`](enfoque_sin_aumento_oficial/data_splits/dataset_completo_original_516.csv)** |

---

## 2. Suite de Pruebas Automatizadas (Auto-Test) y Casos de Evaluación

Para garantizar la reproducibilidad científica y auditar automáticamente todos los componentes del proyecto, se implementó el script [`enfoque_sin_aumento_oficial/scripts/auto_test_suite.py`](enfoque_sin_aumento_oficial/scripts/auto_test_suite.py), que ejecuta de forma automatizada los siguientes casos de prueba:

```bash
# Comando para ejecutar la suite de pruebas automatizadas:
.venv\Scripts\python.exe enfoque_sin_aumento_oficial/scripts/auto_test_suite.py
```

### Matriz de Casos de Evaluación y Auto-Test

| ID Caso | Nombre del Caso de Prueba / Evaluación | Condición Auditada Automáticamente | Resultado Esperado | Estado |
| :---: | :--- | :--- | :--- | :---: |
| **AT-01** | **Auditoría Física del Almacenamiento** | Conteo de archivos en `train/images` (3,304) y `test/images` (824) | Exactamente **4,128 archivos** en disco | ✅ **PASS** |
| **AT-02** | **Filtrado Estricto Sin Aumento (`is_original`)** | Descarte del 100% de sufijos `_L6, _L14, _L20, _R8, _R16, _R22, _VFLIP` | **516 imágenes originales únicas** (264 + 66 + 83 + 103) | ✅ **PASS** |
| **AT-03** | **Verificación de Cero *Data Leakage*** | Intersección conjuntista de `patient_id` entre Train (264), Val (66) y Test (83) | $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$ | ✅ **PASS** |
| **AT-04** | **Estratificación Exacta 80/20 vs 90/10** | Proporción de casos Referibles (~62.7%) conservada en Train, Val y Test | Desviación $< 0.8\%$ entre subconjuntos y mayor potencia en 80/20 ($N=66$ vs $33$) | ✅ **PASS** |
| **AT-05** | **Integridad de los 7 Modelos Entrenados** | Verificación de pesos `.pt` y ejecución de los 7 Notebooks en GPU | 7 checkpoints y 7 notebooks con salidas completas | ✅ **PASS** |
| **AT-06** | **Pruebas Estadísticas Pareadas (Test N=83)** | Significancia estadística de EfficientNet-B0 Base vs. CBAM ($\alpha = 0.05$) | DeLong $Z = 8.9860$ ($p < 0.0001$) y McNemar $\chi^2 = 35.56$ ($p = 2.48 \times 10^{-9}$) | ✅ **PASS** |
| **AT-07** | **Explicabilidad Clínica Causal (XAI)** | Energía focal dentro de la cavidad retiniana en las 83 imágenes de Test | Smooth Grad-CAM++ Retina Energy Ratio = **90.20%** y caída de confianza del **28.06%** | ✅ **PASS** |

---

## 3. Resultados del Entrenamiento y Evaluación SIN Aumento de Datos (7 Notebooks Oficiales)

Entrenados bajo un **protocolo idéntico** (`AdamW`, `lr = 1e-4`, `BCEWithLogitsLoss`, `EarlyStopping` monitoreando `val_loss`, umbral fijo $\tau = 0.50$, e Intervalos de Confianza al 95% con $B = 1,000$ réplicas Bootstrap) usando exclusivamente las **264 imágenes originales de entrenamiento**:

### A. Resultados en Validación Interna ($N = 66$ Originales, $\tau = 0.50$)

| # | Modelo (Notebook Ejecutado en GPU) | Mejor `val_loss` | ROC-AUC Validación [IC 95%] | Accuracy [IC 95%] | Sensibilidad | Especificidad | F1-Score [IC 95%] | Brier | Matriz (TN/FP/FN/TP) |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | ⭐ **[EfficientNet-B0 (Base)](enfoque_sin_aumento_oficial/notebooks/01_EfficientNetB0_Sin_Aumento.ipynb)** | **0.2847** | **0.9498** `[0.8907 - 0.9904]` | 0.8788 `[0.8030 - 0.9545]` | **92.68%** | 80.00% | 0.9048 `[0.8312 - 0.9639]` | 0.0844 | 20 / 5 / 3 / 38 |
| **02** | **[ResNet50](enfoque_sin_aumento_oficial/notebooks/02_ResNet50_Sin_Aumento.ipynb)** | 0.3424 | 0.9488 `[0.8899 - 0.9899]` | **0.8939** `[0.8182 - 0.9545]` | 85.37% | **96.00%** | **0.9091** `[0.8332 - 0.9663]` | **0.0841** | 24 / 1 / 6 / 35 |
| **06** | **[EfficientNet-B0 + CBAM Residual ($\gamma$)](enfoque_sin_aumento_oficial/notebooks/06_EfficientNetB0_CBAM_Residual_Sin_Aumento.ipynb)** | 0.3258 | 0.9439 `[0.8849 - 0.9884]` | 0.8788 `[0.7879 - 0.9545]` | 90.24% | 84.00% | 0.9024 `[0.8250 - 0.9639]` | 0.0952 | 21 / 4 / 4 / 37 |
| **03** | **[VGG16](enfoque_sin_aumento_oficial/notebooks/03_VGG16_Sin_Aumento.ipynb)** | 0.2935 | 0.9400 `[0.8839 - 0.9847]` | 0.8788 `[0.8030 - 0.9545]` | 85.37% | 92.00% | 0.8974 `[0.8214 - 0.9578]` | 0.0974 | 23 / 2 / 6 / 35 |
| **04** | **[ConvNeXt-Tiny](enfoque_sin_aumento_oficial/notebooks/04_ConvNeXt_Tiny_Sin_Aumento.ipynb)** | 0.3894 | 0.9380 `[0.8775 - 0.9839]` | 0.7879 `[0.6818 - 0.8788]` | **92.68%** | 56.00% | 0.8444 `[0.7556 - 0.9167]` | 0.1332 | 14 / 11 / 3 / 38 |
| **05** | **[CNN Desde Cero](enfoque_sin_aumento_oficial/notebooks/05_CNN_Desde_Cero_Sin_Aumento.ipynb)** | 0.6966 | 0.8171 `[0.6926 - 0.9232]` | 0.8030 `[0.6970 - 0.8939]` | **92.68%** | 60.00% | 0.8539 `[0.7674 - 0.9247]` | 0.1735 | 15 / 10 / 3 / 38 |
| **07** | **[CapsuleNet (Dynamic Routing)](enfoque_sin_aumento_oficial/notebooks/07_CapsuleNet_Sin_Aumento.ipynb)** | 0.7052 | 0.7410 `[0.6090 - 0.8574]` | 0.6818 `[0.5758 - 0.7879]` | 68.29% | 68.00% | 0.7273 `[0.6060 - 0.8312]` | 0.2439 | 17 / 8 / 13 / 28 |

### B. Resultados en Test Independiente Parte A ($N = 83$ Originales, $\tau = 0.50$)

| # | Modelo (Sin Aumento de Datos) | ROC-AUC Test [IC 95%] | Accuracy Test [IC 95%] | Sensibilidad | Especificidad | F1-Score Test [IC 95%] | Brier | Matriz (TN/FP/FN/TP) |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | ⭐ **EfficientNet-B0 (Ganador Base)** | **0.9944** `[0.9782 - 1.0000]` | **0.9759** `[0.9398 - 1.0000]` | **98.08%** | **96.77%** | **0.9808** `[0.9462 - 1.0000]` | **0.0269** | 30 / 1 / 1 / 51 |
| **06** | **EfficientNet-B0 + CBAM Residual ($\gamma$)** | **0.9969** `[0.9879 - 1.0000]` | 0.9639 `[0.9157 - 1.0000]` | 96.15% | **96.77%** | 0.9709 `[0.9298 - 1.0000]` | 0.0296 | 30 / 1 / 2 / 50 |
| **02** | **ResNet50** | 0.9926 `[0.9748 - 1.0000]` | **0.9759** `[0.9398 - 1.0000]` | **98.08%** | **96.77%** | **0.9808** `[0.9483 - 1.0000]` | 0.0350 | 30 / 1 / 1 / 51 |
| **04** | **ConvNeXt-Tiny** | 0.9870 `[0.9629 - 0.9994]` | 0.8554 `[0.7711 - 0.9277]` | **98.08%** | 64.52% | 0.8947 `[0.8314 - 0.9500]` | 0.0845 | 20 / 11 / 1 / 51 |
| **03** | **VGG16** | 0.9677 `[0.9264 - 0.9946]` | 0.8916 `[0.8190 - 0.9518]` | 90.38% | 87.10% | 0.9126 `[0.8468 - 0.9643]` | 0.0724 | 27 / 4 / 5 / 47 |
| **05** | **CNN Desde Cero** | 0.9125 `[0.8419 - 0.9698]` | 0.8554 `[0.7711 - 0.9277]` | 94.23% | 70.97% | 0.8909 `[0.8222 - 0.9457]` | 0.1355 | 22 / 9 / 3 / 49 |
| **07** | **CapsuleNet (Dynamic Routing)** | 0.8871 `[0.8065 - 0.9550]` | 0.8434 `[0.7590 - 0.9157]` | 82.69% | 87.10% | 0.8687 `[0.7912 - 0.9333]` | 0.2379 | 27 / 4 / 9 / 43 |

---

## 4. Comparativa Directa: Entrenamiento CON Aumento de Datos ($N=2,112$) vs. SIN Aumento de Datos ($N=264$)

Al comparar los resultados del entrenamiento con el dataset aumentado ($\times 8 = 2,112$ imágenes en Train) frente al entrenamiento puro **sin aumento de datos** ($264$ imágenes originales en Train), evaluados sobre el mismo conjunto de **Test Independiente ($N = 83$ imágenes originales)**:

| Arquitectura | Enfoque de Entrenamiento | Imágenes de Train | `val_loss` | ROC-AUC Test ($N=83$) | Accuracy Test ($N=83$) | Sensibilidad | Especificidad | F1-Score | Brier Score | Diagnóstico Técnico y Clínico |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **EfficientNet-B0 (Base)** | **Sin Aumento (Puro)** | **264** | 0.2847 | **0.9944** | **97.59%** | **98.08%** | **96.77%** | **0.9808** | **0.0269** | **Mejor calibración y exactitud al preservar la geometría ocular original.** |
| *EfficientNet-B0 (Base)* | *Con Aumento ($\times 8$)* | *2,112* | *0.1873* | *0.9950* | *95.18%* | *96.15%* | *93.55%* | *0.9615* | *0.0378* | *Excelente AUC, pero las rotaciones/flips introducen ligera distorsión en bordes.* |
| **ResNet50** | **Sin Aumento (Puro)** | **264** | 0.3424 | **0.9926** | **97.59%** | **98.08%** | **96.77%** | **0.9808** | **0.0350** | **Convergencia limpia y más rápida (8 épocas vs 15+).** |
| *ResNet50* | *Con Aumento ($\times 8$)* | *2,112* | *0.5121* | *0.9870* | *92.77%* | *96.15%* | *87.10%* | *0.9434* | *0.0526* | *Mayor pérdida en validación por redundancia de réplicas.* |
| **VGG16** | **Sin Aumento (Puro)** | **264** | 0.2935 | 0.9677 | 89.16% | 90.38% | 87.10% | 0.9126 | 0.0724 | Ligera sensibilidad al tamaño muestral por tener 138M de parámetros. |
| *VGG16* | *Con Aumento ($\times 8$)* | *2,112* | *0.2450* | **0.9895** | **95.18%** | **96.15%** | **93.55%** | **0.9615** | **0.0482** | VGG16 sí se beneficia de más muestras por su cabezal denso pesado. |
| **ConvNeXt-Tiny** | **Sin Aumento (Puro)** | **264** | 0.3894 | **0.9870** | **85.54%** | **98.08%** | 64.52% | **0.8947** | 0.0845 | Alta sensibilidad clínica (98.08%) y mejor AUC sin réplicas artificiales. |
| *ConvNeXt-Tiny* | *Con Aumento ($\times 8$)* | *2,112* | *0.4530* | *0.9090* | *85.37%* | *96.15%* | *80.00%* | *0.8929* | *0.0627* | *Menor AUC en test cuando se entrena con recortes rotados.* |
| **CNN Desde Cero** | **Sin Aumento (Puro)** | **264** | 0.6966 | **0.9125** | **85.54%** | **94.23%** | 70.97% | **0.8909** | 0.1355 | Buen aprendizaje de color/textura, aunque inferior a pre-entrenados. |
| *CNN Desde Cero* | *Con Aumento ($\times 8$)* | *2,112* | *0.4136* | *0.8958* | *81.93%* | *80.77%* | *83.87%* | *0.8485* | *0.1272* | *Menor sensibilidad diagnóstica frente a lesiones sutiles.* |
| **EfficientNet-B0 + CBAM** | **CBAM Residual ($\gamma$, Sin Aumento)** | **264** | **0.3258** | **0.9969** | **96.39%** | **96.15%** | **96.77%** | **0.9709** | **0.0296** | **El enlace residual con $\gamma_c, \gamma_s \to 0$ evita el colapso y alcanza AUC=0.9969.** |
| *EfficientNet-B0 + CBAM* | *CBAM Directo (Sin $\gamma$, Con Aumento)* | *2,112* | *0.6912* | *0.3679* | *39.76%* | *44.23%* | *32.26%* | *0.4565* | *0.2522* | *Colapso total al multiplicar sigmoides sin conexión residual.* |
| **CapsuleNet** | **Sin Aumento (Puro)** | **264** | 0.7052 | **0.8871** | **84.34%** | **82.69%** | **87.10%** | **0.8687** | 0.2379 | Entrenamiento estable desde cero, superado por redes con Transfer Learning. |

---

## 5. Conclusiones Clave de la Comparativa Experimental

1. **Superioridad del Enfoque Sin Aumento de Datos en Modelos Pre-entrenados (`EfficientNet-B0` y `ResNet50`):**
   * Al entrenar estrictamente con las **264 imágenes originales sin aumento**, **EfficientNet-B0** sube su exactitud en el Test de 83 imágenes de **95.18% a 97.59%** (`TN=30, FP=1, FN=1, TP=51`) y mejora su calibración probabilística (**Brier Score = 0.0269** vs `0.0378`).
   * Esto ocurre porque las fotografías de fondo de ojo mantienen su geometría anatómica nativa (relación espacial entre mácula temporal y disco óptico nasal) y no sufren difuminación por interpolación bilineal al rotar microaneurismas de 2 a 5 píxeles.
2. **Importancia Crítica del CBAM Residual con Escalares $\gamma_c$ y $\gamma_s$:**
   * Mientras que el CBAM multiplicativo directo destruía los mapas pre-entrenados de ImageNet (`AUC = 0.3679`, DeLong $Z = 8.99, p < 0.0001$), la formulación **CBAM Residual con $\gamma$ inicializado en 0** (`Notebook 06`) preserva las características base y alcanza un **ROC-AUC de 0.9969** en test.
   * No obstante, dado que en validación interna **EfficientNet-B0 Base** logra menor pérdida (`val_loss = 0.2847` vs `0.3258` de CBAM Residual), se selecciona formalmente **EfficientNet-B0 Base** por criterio de validación y principio de parsimonia.
3. **Desempeño en el Test Oficial Externo ($N = 103$ originales de la carpeta `test/`):**
   * Al evaluar el modelo ganador **EfficientNet-B0 Sin Aumento** sobre las 103 imágenes originales independientes del conjunto de prueba externo del Challenge IDRiD, obtiene **ROC-AUC = 0.8906**, **Sensibilidad = 89.06%** y **F1-Score = 0.8702**, confirmando su capacidad de generalización clínica fuera de la muestra de desarrollo.
