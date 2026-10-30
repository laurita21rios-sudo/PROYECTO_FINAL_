import os
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, confusion_matrix

print("1. Extrayendo model.weights.h5...")
zip_path = 'resultados_efficientnetb0_cbam/modelos/EfficientNetB0_CBAM_TransferLearning_224.keras'
weights_path = 'resultados_efficientnetb0_cbam/modelos/extracted_weights.weights.h5'
with zipfile.ZipFile(zip_path, 'r') as z:
    with open(weights_path, 'wb') as f:
        f.write(z.read('model.weights.h5'))

print("2. Construyendo arquitectura idéntica de CBAM...")
inputs = tf.keras.Input(shape=(224, 224, 3), name='input_layer')

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights=None,
    input_tensor=inputs
)
x = base_model.output

# Channel Attention
gap = tf.keras.layers.GlobalAveragePooling2D(keepdims=True, name='cbam_final_gap')(x)
gmp = tf.keras.layers.GlobalMaxPooling2D(keepdims=True, name='cbam_final_gmp')(x)

dense1 = tf.keras.layers.Dense(160, activation='relu', name='cbam_final_channel_dense1')
dense2 = tf.keras.layers.Dense(1280, activation='linear', name='cbam_final_channel_dense2')

gap_out = dense2(dense1(gap))
gmp_out = dense2(dense1(gmp))

ch_add = tf.keras.layers.Add(name='add')([gap_out, gmp_out])
ch_att = tf.keras.layers.Activation('sigmoid', name='cbam_final_channel_sigmoid')(ch_add)
x_channel = tf.keras.layers.Multiply(name='cbam_final_channel_multiply')([x, ch_att])

# Spatial Attention (use_bias=False!)
avg_pool = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
                                  output_shape=(7, 7, 1), name='cbam_final_spatial_avg')(x_channel)
max_pool = tf.keras.layers.Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
                                  output_shape=(7, 7, 1), name='cbam_final_spatial_max')(x_channel)

concat = tf.keras.layers.Concatenate(axis=-1, name='cbam_final_spatial_concat')([avg_pool, max_pool])
sp_att = tf.keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid', use_bias=False, name='cbam_final_spatial_conv')(concat)
x_spatial = tf.keras.layers.Multiply(name='cbam_final_spatial_multiply')([x_channel, sp_att])

# Classifier Head
gap_head = tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling')(x_spatial)
drop1 = tf.keras.layers.Dropout(0.35, name='dropout')(gap_head)
dense_fc = tf.keras.layers.Dense(256, activation='relu', name='dense_256')(drop1)
drop2 = tf.keras.layers.Dropout(0.35, name='dropout_2')(dense_fc)
output = tf.keras.layers.Dense(1, activation='sigmoid', name='prediction')(drop2)

model = tf.keras.Model(inputs=inputs, outputs=output, name='EfficientNetB0_CBAM')

print("3. Cargando pesos en el modelo...")
model.load_weights(weights_path)
print("¡Pesos cargados exitosamente sin un solo error!")

# 4. Inferencia en las 83 imágenes
ref_df = pd.read_csv('ResNet50/predicciones_validacion.csv')
search_dirs = [
    os.path.join('data', 'IDRiD2', 'IDRiD', 'test', 'images'),
    os.path.join('data', 'IDRiD2', 'IDRiD', 'train', 'images')
]

images = []
y_true = []
valid_names = []

for _, row in ref_df.iterrows():
    img_name = row['Image name']
    found_path = None
    for s_dir in search_dirs:
        for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
            p = os.path.join(s_dir, img_name + ext)
            if os.path.exists(p):
                found_path = p
                break
        if found_path:
            break
    
    img = Image.open(found_path).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    images.append(arr)
    y_true.append(row['binary_label'])
    valid_names.append(img_name)

X = np.array(images)
y_true = np.array(y_true)
print(f"Total imágenes cargadas: {len(X)}")

preds = model.predict(X, batch_size=16)
y_prob = preds.flatten()

# Save predictions
pred_df = pd.DataFrame({
    'Image name': valid_names,
    'binary_label': y_true,
    'probability': y_prob,
    'pred_label_05': (y_prob >= 0.5).astype(int)
})
pred_df.to_csv('resultados_efficientnetb0_cbam/predicciones_cbam_83_exactas.csv', index=False)

# Compute metrics
brier = float(brier_score_loss(y_true, y_prob))
p, r, _ = precision_recall_curve(y_true, y_prob)
pr_auc = float(auc(r, p))
roc_auc = float(roc_auc_score(y_true, y_prob))

cm = confusion_matrix(y_true, (y_prob >= 0.5).astype(int))
tn, fp, fn, tp = cm.ravel()

print(f"=== RESULTADOS EXACTOS CALCULADOS ===")
print(f"AUC-ROC:     {roc_auc:.4f}")
print(f"PR-AUC:      {pr_auc:.4f}")
print(f"Brier Score: {brier:.4f}")
print(f"Matriz de Confusión: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
