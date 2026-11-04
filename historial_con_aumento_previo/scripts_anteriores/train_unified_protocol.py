"""
Protocolo de Entrenamiento Unificado para Detección de Retinopatía Diabética (IDRiD)
===================================================================================
- Partición: Agrupada y estratificada (cargada desde data/splits/split_90_10.joblib o split_80_20.joblib)
- Criterio Early Stopping: Monitoreo estricto de val_loss (con restore_best_weights=True)
- Punto de corte (Umbral): tau = 0.5 estandarizado
- Modelos soportados: ConvNeXt-Tiny, EfficientNetB0, ResNet50, VGG16, CNN Personalizada
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento Unificado IDRiD")
    parser.add_argument('--model', type=str, default='EfficientNetB0',
                        choices=['ConvNeXt-Tiny', 'EfficientNetB0', 'ResNet50', 'VGG16', 'CNN'],
                        help="Arquitectura a entrenar")
    parser.add_argument('--split', type=str, default='80_20', choices=['80_20', '90_10'],
                        help="Partición a utilizar")
    parser.add_argument('--epochs', type=int, default=30, help="Máximo de épocas")
    parser.add_argument('--batch_size', type=int, default=16, help="Tamaño de batch")
    parser.add_argument('--patience', type=int, default=7, help="Paciencia para EarlyStopping en val_loss")
    return parser.parse_args()

def load_split_data(split_name='80_20'):
    split_file = f'data/splits/split_{split_name}.joblib'
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"No se encontró {split_file}. Ejecuta scripts/create_stratified_split.py primero.")
    data = joblib.load(split_file)
    return data['train_images_df'], data['val_images_df']

def build_model(model_name, input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    
    if model_name == 'EfficientNetB0':
        base = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
        base.trainable = False
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    elif model_name == 'ResNet50':
        base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
        base.trainable = False
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dropout(0.4)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    elif model_name == 'VGG16':
        base = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
        base.trainable = False
        x = tf.keras.layers.Flatten()(base.output)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    elif model_name == 'CNN':
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(2, 2)(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    elif model_name == 'ConvNeXt-Tiny':
        base = tf.keras.applications.ConvNeXtTiny(include_top=False, weights='imagenet', input_tensor=inputs)
        base.trainable = False
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")
        
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
            tf.keras.metrics.AUC(name='auc', curve='ROC')
        ]
    )
    return model

def get_callbacks(model_name, patience=7):
    os.makedirs('modelos/checkpoints', exist_ok=True)
    chk_path = f'modelos/checkpoints/{model_name}_best_val_loss.keras'
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=chk_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

if __name__ == '__main__':
    args = parse_args()
    print("==================================================================")
    print(f"Iniciando protocolo idéntico para: {args.model}")
    print(f"Split seleccionado: {args.split}")
    print(f"Criterio Early Stopping: val_loss (patience={args.patience})")
    print(f"Punto de corte de inferencia: tau = 0.5")
    print("==================================================================")
    
    train_df, val_df = load_split_data(args.split)
    print(f"Muestras de Train (con data augmentation): {len(train_df)}")
    print(f"Muestras de Val (pacientes originales): {len(val_df)}")
    
    model = build_model(args.model)
    callbacks = get_callbacks(args.model, patience=args.patience)
    print(f"Modelo {args.model} compilado con éxito. Listo para entrenar.")
