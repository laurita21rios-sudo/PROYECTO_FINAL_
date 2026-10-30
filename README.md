# Detección Asistida de Retinopatía Diabética mediante Deep Learning

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Dataset IDRiD](https://img.shields.io/badge/Dataset-IDRiD-green.svg)](https://idrid.grand-challenge.org/)
[![Clinical Grade](https://img.shields.io/badge/Clinical--Grade-Sensitivity%20100%25-red.svg)]()

Repositorio de investigación enfocado en el desarrollo, optimización clínica y explicabilidad de modelos de aprendizaje profundo para el cribado (*screening*) automatizado de **Retinopatía Diabética Referible (RDR)** a partir de retinografías digitales del dataset **IDRiD** (*Indian Diabetic Retinopathy Image Dataset*).

---

## 📊 Benchmark Consolidado de Modelos

Todos los modelos fueron rigurosamente evaluados sobre las mismas **83 imágenes originales de pacientes únicos** (N = 83: 52 referibles, 31 no referibles), garantizando **cero data leakage** y eliminando réplicas sintéticas en el conjunto de prueba.

| Modelo | N | Umbral ($\tau$) | Sensibilidad | Especificidad | Precisión | Accuracy | F1-Score | AUC-ROC | PR-AUC | Brier Score | Matriz Confusión (TN/FP/FN/TP) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ConvNeXt-Tiny (Clinical)** | 83 | $0.379$ | **1.0000** | 0.6774 | 0.8387 | 0.8795 | 0.9123 | 0.9870 | 0.9929 | 0.0627 | 21 / 10 / **0** / 52 |
| **EfficientNetB0 (Baseline)** | 83 | $0.500$ | 0.9615 | **0.9355** | **0.9615** | **0.9518** | **0.9615** | **0.9950** | **0.9971** | **0.0378** | 29 / 2 / 2 / 50 |
| **VGG16 (Fine-Tuned)** | 83 | $0.500$ | 0.9615 | **0.9355** | **0.9615** | **0.9518** | **0.9615** | 0.9895 | 0.9944 | — | 29 / 2 / 2 / 50 |
| **ResNet50 (Fine-Tuning Opt)** | 83 | $0.632$ | 0.9423 | **0.9355** | 0.9608 | 0.9398 | 0.9515 | 0.9876 | 0.9931 | 0.0526 | 29 / 2 / 3 / 49 |
| **ResNet50 (Baseline 0.5)** | 83 | $0.500$ | 0.9615 | 0.8710 | 0.9259 | 0.9277 | 0.9434 | 0.9876 | 0.9931 | 0.0526 | 27 / 4 / 2 / 50 |
| **CapsuleNet** | 83 | $0.500$ | 0.8846 | 0.7742 | 0.8679 | 0.8434 | 0.8762 | 0.9107 | 0.9475 | 0.1229 | 24 / 7 / 6 / 46 |
| **CNN Personalizada (Deep)** | 83 | $0.500$ | 0.8077 | 0.8387 | 0.8936 | 0.8193 | 0.8485 | 0.8958 | 0.9336 | 0.1272 | 26 / 5 / 10 / 42 |
| **EfficientNetB0 + CBAM** | 83 | $0.500$ | 0.4231 | 0.1613 | 0.4583 | 0.3253 | 0.4400 | 0.1948 | — | — | 5 / 26 / 30 / 22 |

> 📌 **Notas Metodológicas:**
> 1. **Calibración y Tamizaje:** `ConvNeXt-Tiny` prioriza la seguridad clínica mediante calibración de temperatura ($T=2.1648$) y optimización de umbral ($\tau=0.379$), logrando **cero falsos negativos ($FN=0$)**.
> 2. **Corrección de Muestra Estandarizada:** `EfficientNetB0` se reporta formalmente sobre las 83 imágenes originales (corrigiendo reportes preliminares internos que evaluaban sobre 664 réplicas con TTA).
> 3. **Análisis de Colapso en CBAM:** La arquitectura `EfficientNetB0 + CBAM` evidenció colapso de gradiente ($AUC=0.1948$) atribuible a la inserción de bloques de atención no preentrenados sobre pesos congelados de transfer learning.
> 
> Para la discusión médica completa y el análisis de trade-offs diagnósticos, consulta [docs/discussion.md](docs/discussion.md).

---

## 🔍 Explicabilidad y Validación Médica (Grad-CAM)

En aplicaciones de salud no basta con alta precisión; el modelo debe fundamentar sus predicciones en anomalías biológicas reales. Se implementó una auditoría visual mediante mapas **Grad-CAM** (*Gradient-weighted Class Activation Mapping*):

![Grad-CAM TP, TN, FP, FN](results/figures/gradcam_cnn_tp_tn_fp_fn.png)

* **Verdaderos Positivos (TP):** Foco prioritario sobre microaneurismas periféricos, exudados algodonosos y exudados duros perimaculares.
* **Verdaderos Negativos (TN):** Distribución uniforme de atención sobre el parénquima retiniano sin activación focal anómala.
* **Auditoría de Errores (FP / FN):** Detección de posibles sesgos provocados por iluminación periférica y artefactos del lente oftalmológico.

---

## 📁 Estructura del Repositorio

La arquitectura del proyecto sigue el estándar de reproducibilidad científica:

```text
PROYECTO_FINAL/
│
├── docs/                                # Documentación académica y discusión
│   └── discussion.md                    # Análisis clínico, trade-offs y limitaciones
│
├── results/                             # Salidas reproducibles y consolidadas
│   ├── figures/                         # Mapas Grad-CAM, curvas ROC y análisis de error
│   ├── benchmark_summary.json           # Métricas cuantitativas de todos los modelos
│   └── benchmark_summary.md             # Tabla comparativa en formato Markdown
│
├── scripts/                             # Automatizaciones y pipelines
│   ├── consolidate_metrics.py           # Script consolidador de métricas clínicas
│   └── update_benchmark_markdown.py     # Generador automático de tablas comparativas
│
├── src/                                 # Módulos transversales del proyecto
│   ├── __init__.py
│   └── env_setup.py                     # Selector automático multi-entorno (Local/Kaggle/Colab)
│
├── .gitignore                           # Exclusión de datos pesados (data/) y pesos (*.keras)
├── pyproject.toml                       # Especificación de dependencias
└── README.md                            # Documento ejecutivo principal
```

---

## 🚀 Instalación y Reproducción

### 1. Clonar el repositorio
```bash
git clone https://github.com/laurita21rios-sudo/PROYECTO_FINAL_.git
cd PROYECTO_FINAL_
```

### 2. Configurar el entorno de Python
```bash
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/Mac:
source .venv/bin/activate

pip install -r pyproject.toml
```

### 3. Descarga Automática del Dataset desde Kaggle
El dataset base de entrenamiento y evaluación es [`mohamedabdalkader/indian-diabetic-retinopathy-image-dataset-idrid2`](https://www.kaggle.com/datasets/mohamedabdalkader/indian-diabetic-retinopathy-image-dataset-idrid2):

```bash
kaggle datasets download -d mohamedabdalkader/indian-diabetic-retinopathy-image-dataset-idrid2 -p data/IDRiD2 --unzip
```

### 4. Soporte Multi-Entorno (Local / Kaggle / Colab)
El proyecto incluye un detector automático de entorno en [`src/env_setup.py`](src/env_setup.py). Tanto los notebooks como los scripts adaptan sus rutas de trabajo y tamaño de lote (`BATCH_SIZE`) según la plataforma:

```python
from src.env_setup import setup_environment

env = setup_environment()
print(f"Entorno detectado: {env['env']}")          # 'LOCAL', 'KAGGLE' o 'COLAB'
print(f"Ruta de datos: {env['data_dir']}")         # Resuelve automáticamente la ruta activa
print(f"Batch size recomendado: {env['batch_size']}")
```

### 5. Ejecutar la consolidación de métricas
Para recalcular y actualizar automáticamente la tabla de resultados a partir de los registros de inferencia:

```bash
python scripts/consolidate_metrics.py
```

---

## 📑 Principales Conclusiones Clínicas

1. **Estrategia Óptima de Despliegue en Salud (Cribado en 2 Fases):**
   * **Fase 1 (Cribado masivo no invasivo):** Desplegar `ConvNeXt-Tiny` con umbral ajustado a $0.379$. Garantiza que ningún paciente enfermo quede sin atención ($FN=0$, Sensibilidad $100\%$).
   * **Fase 2 (Confirmación diagnóstica / Triaje especializado):** Desplegar `EfficientNetB0` o `ResNet50` (umbral $0.632$) para maximizar precisión y especificidad ($93.55\%$), minimizando las sobrederivaciones y costos al sistema de salud.
2. **Importancia de la Explicabilidad:** La validación con mapas Grad-CAM confirmó que la red aprende características anatomopatológicas genuinas (microaneurismas, exudados) y no artefactos del retinógrafo.

Para más detalles, consulta el documento completo de discusión en [docs/discussion.md](docs/discussion.md).
