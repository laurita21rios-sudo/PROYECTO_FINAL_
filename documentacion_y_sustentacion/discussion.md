# Discusión Académica y Análisis Clínico

## 1. Interpretación de Resultados y Hallazgos Principales

### A. Priorización Clínica de la Sensibilidad (Recall)
En el cribado (*screening*) de Retinopatía Diabética (RD), la consecuencia médica de un **Falso Negativo (FN)** es crítica: un paciente con retinopatía proliferativa o edema macular no detectado a tiempo puede sufrir pérdida visual irreversible o ceguera. Por el contrario, un **Falso Positivo (FP)** únicamente deriva al paciente a una consulta oftalmológica de confirmación.

Bajo este principio:
* **ConvNeXt-Tiny (Clinical):** Al optimizar el umbral de decisión a $\tau = 0.379$, se alcanzó una **Sensibilidad del 100% ($FN = 0$)** en el conjunto de prueba IDRiD, manteniendo una especificidad del 67.74% y un **AUC-ROC de 0.9870** (PR-AUC 0.9929). Esto lo posiciona como el candidato idóneo para un sistema de triaje o primera línea diagnóstica.
* **ResNet50 (Fine-Tuning v2):** Con un umbral balanceado ($\tau = 0.632$), ResNet50 demostró la mayor robustez armónica global, alcanzando un **F1-Score de 0.9515**, **Accuracy de 93.98%**, Sensibilidad de 94.23% y Especificidad de 93.55%. Es el modelo óptimo cuando se busca minimizar la sobrecarga del sistema de salud sin comprometer la seguridad diagnóstica.

---

## 2. Explicabilidad Médica con Grad-CAM (XAI)

Para garantizar la fiabilidad en entornos clínicos y descartar el aprendizaje de atajos espurios (*shortcut learning*), se implementaron mapas de activación de clase Grad-CAM en la arquitectura convolucional:

1. **Verdaderos Positivos (TP):** La red activa fuertemente regiones anatómicas coherentes con lesiones vasculares retinarias: microaneurismas, exudados duros perimaculares y hemorragias intraretinianas.
2. **Verdaderos Negativos (TN):** Activación homogénea o dispersa de fondo sin fijación focal en zonas sanas de la retina.
3. **Análisis de Errores (FP y FN):**
   * Los falsos positivos se correlacionan frecuentemente con artefactos de iluminación en la periferia de la imagen retiniana o reflexiones fotográficas del polo posterior.
   * La inclusión de auditorías visuales permite al oftalmólogo validar la propuesta del modelo antes de emitir un juicio diagnóstico definitivo.

---

## 3. Discusión de Anomalías: El Fallo del Módulo de Atención (CBAM)

El experimento con **EfficientNetB0 + CBAM** arrojó un rendimiento significativamente inferior (Accuracy 32.53%, AUC-ROC 0.1948). El análisis retrospectivo identifica las siguientes causas:
* **Incompatibilidad de Transfer Learning:** La inserción de bloques de atención CBAM no preentrenados dentro de una columna vertebral convolucional ya convergente altera la propagación del gradiente durante las primeras épocas.
* **Sobreajuste a nivel espacial en muestras reducidas:** Sin una estrategia de inicialización cálida (*warmup*) o descongelamiento por etapas (como la aplicada exitosamente en ResNet50 y ConvNeXt), los pesos del módulo CBAM descalibran la extracción de características finas.

---

## 4. Limitaciones del Estudio

1. **Tamaño Muestral del Dataset:** IDRiD proporciona anotaciones de alta calidad y resolución clínica, pero el número total de ojos evaluados es limitado comparado con consorcios masivos (EyePACS, Messidor).
2. **Validación Cruzada Inter-Poblacional:** El modelo fue entrenado en pacientes de la cohorte india (IDRiD); se requiere validación externa en poblaciones étnicamente diversas para descartar sesgos de pigmentación del fondo de ojo.
3. **Latencia Computacional:** ConvNeXt y ResNet50 requieren hardware acelerado por GPU para inferencia en tiempo real en dispositivos portátiles o cámaras de fondo de ojo no midriáticas de bajo coste.

---

## 5. Trabajo Futuro y Líneas de Investigación

* **Validación Externa:** Probar la capacidad de generalización en los datasets Kaggle Diabetic Retinopathy y Messidor-2 sin re-entrenamiento previo (*zero-shot domain adaptation*).
* **Segmentación de Lesiones Concurrente:** Entrenar modelos multitask que clasifiquen el grado de RD y a la vez segmenten máscaras de microaneurismas y exudados algodonosos.
* **Cuantización y Despliegue:** Conversión de ConvNeXt-Tiny y ResNet50 a formatos INT8 / ONNX / TFLite para ejecución directa en el borde (*Edge AI*).
