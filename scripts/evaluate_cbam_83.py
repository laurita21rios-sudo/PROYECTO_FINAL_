import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, confusion_matrix

tf.keras.config.enable_unsafe_deserialization()

# Load model
model_path = 'resultados_efficientnetb0_cbam/modelos/EfficientNetB0_CBAM_TransferLearning_224.keras'
print("Cargando modelo CBAM...")
model = tf.keras.models.load_model(model_path, compile=False)
print("Modelo cargado con éxito.")

# Reference 83 test images from ResNet50
ref_df = pd.read_csv('ResNet50/predicciones_validacion.csv')
test_img_dir = os.path.join('data', 'IDRiD2', 'IDRiD', 'test', 'images')

images = []
y_true = []
valid_names = []

for _, row in ref_df.iterrows():
    img_name = row['Image name']
    found_path = None
    for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
        p = os.path.join(test_img_dir, img_name + ext)
        if os.path.exists(p):
            found_path = p
            break
    
    if found_path is None:
        print(f"Alerta: no se encontró {img_name}")
        continue
    
    img = Image.open(found_path).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    # EfficientNet preprocessing: values in [0, 255] or tf.keras.applications.efficientnet.preprocess_input
    # Let's inspect model first layer to see if it has Rescaling
    images.append(arr)
    y_true.append(row['binary_label'])
    valid_names.append(img_name)

X = np.array(images)
y_true = np.array(y_true)
print(f"Total imágenes cargadas: {len(X)}")

# Predict
preds = model.predict(X, batch_size=16)
y_prob = preds.flatten()

# Save predictions
pred_df = pd.DataFrame({
    'Image name': valid_names,
    'binary_label': y_true,
    'probability': y_prob,
    'pred_label_05': (y_prob >= 0.5).astype(int)
})
pred_df.to_csv('resultados_efficientnetb0_cbam/predicciones_cbam_83.csv', index=False)

# Compute metrics
brier = float(brier_score_loss(y_true, y_prob))
p, r, _ = precision_recall_curve(y_true, y_prob)
pr_auc = float(auc(r, p))
roc_auc = float(roc_auc_score(y_true, y_prob))

cm = confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
tn, fp, fn, tp = cm.ravel()

print(f"CBAM ROC-AUC: {roc_auc:.4f}")
print(f"CBAM PR-AUC: {pr_auc:.4f}")
print(f"CBAM Brier Score: {brier:.4f}")
print(f"Matriz de confusión: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# Update benchmark_summary.json
with open('results/benchmark_summary.json', 'r', encoding='utf-8') as f:
    bench = json.load(f)

if 'EfficientNetB0 + CBAM' in bench:
    bench['EfficientNetB0 + CBAM']['pr_auc'] = pr_auc
    bench['EfficientNetB0 + CBAM']['brier_score'] = brier
    bench['EfficientNetB0 + CBAM']['roc_auc'] = roc_auc
    bench['EfficientNetB0 + CBAM']['accuracy'] = float((tp + tn) / len(y_true))
    bench['EfficientNetB0 + CBAM']['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    bench['EfficientNetB0 + CBAM']['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = bench['EfficientNetB0 + CBAM']['sensitivity']
    bench['EfficientNetB0 + CBAM']['precision'] = precision
    bench['EfficientNetB0 + CBAM']['f1_score'] = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    bench['EfficientNetB0 + CBAM']['confusion_matrix'] = {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }

with open('results/benchmark_summary.json', 'w', encoding='utf-8') as f:
    json.dump(bench, f, indent=2, ensure_ascii=False)

print("Actualización completada!")
