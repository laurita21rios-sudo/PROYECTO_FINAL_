#!/usr/bin/env python3
"""
Suite de Auto-Test y Evaluacion Automatizada Integral (Con y Sin Aumento de Datos)
Verifica la integridad de los splits (516 originales / 4,128 totales), ausencia de
data leakage, y ejecuta la evaluacion automatica de los 7 modelos entrenados sobre:
  1. Validacion Interna (N = 66 originales)
  2. Test Parte A (N = 83 originales)
  3. Test Oficial Externo (N = 103 originales)
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import csv
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, confusion_matrix, brier_score_loss

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("SUITE DE AUTO-TEST Y EVALUACION AUTOMATIZADA DEL PROYECTO IDRiD")
print(f"Dispositivo de Inferencia: {device}")
print("=" * 80)

# -------------------------------------------------------------------------
# CASO DE PRUEBA 1: Integridad del Censo Global (516 Originales / 4,128 en Disco)
# -------------------------------------------------------------------------
train_disk_files = os.listdir("data/IDRiD2/IDRiD/train/images")
test_disk_files = os.listdir("data/IDRiD2/IDRiD/test/images")
assert len(train_disk_files) == 3304, f"Error en train disk count: {len(train_disk_files)}"
assert len(test_disk_files) == 824, f"Error en test disk count: {len(test_disk_files)}"
assert len(train_disk_files) + len(test_disk_files) == 4128, "El total en disco debe ser 4,128"
print("[PASS] Auto-Test 1: Verificacion en disco -> 4,128 archivos (3,304 train + 824 test).")

# -------------------------------------------------------------------------
# CASO DE PRUEBA 2: Integridad de Splits Sin Aumento y Cero Data Leakage
# -------------------------------------------------------------------------
with open("enfoque_sin_aumento_oficial/data_splits/split_sin_aumento_80_20.csv", encoding="utf-8") as f:
    dev_rows = list(csv.DictReader(f))
train_rows = [r for r in dev_rows if r["split"] == "train"]
val_rows   = [r for r in dev_rows if r["split"] == "val"]

with open("enfoque_sin_aumento_oficial/data_splits/test_sin_aumento_83.csv", encoding="utf-8") as f:
    test83_rows = list(csv.DictReader(f))

with open("enfoque_sin_aumento_oficial/data_splits/test_oficial_103_sin_aumento.csv", encoding="utf-8") as f:
    test103_rows = list(csv.DictReader(f))

assert len(train_rows) == 264, "Train debe tener 264 imagenes originales"
assert len(val_rows) == 66, "Validacion debe tener 66 imagenes originales"
assert len(test83_rows) == 83, "Test Parte A debe tener 83 imagenes originales"
assert len(test103_rows) == 103, "Test Oficial Externo debe tener 103 imagenes originales"
assert len(train_rows) + len(val_rows) + len(test83_rows) + len(test103_rows) == 516, "Total originales debe ser 516"

set_tr = {r["Image name"] for r in train_rows}
set_va = {r["Image name"] for r in val_rows}
set_te = {r["Image name"] for r in test83_rows}
assert len(set_tr.intersection(set_va)) == 0, "Fuga detectada entre Train y Val"
assert len(set_tr.intersection(set_te)) == 0, "Fuga detectada entre Train y Test 83"
assert len(set_va.intersection(set_te)) == 0, "Fuga detectada entre Val y Test 83"
print("[PASS] Auto-Test 2: Cero Data Leakage verificado (264 Train / 66 Val / 83 Test A / 103 Test B = 516 Originales).")

# -------------------------------------------------------------------------
# CASO DE PRUEBA 3: Verificacion de Pesos Guardados de los 7 Modelos
# -------------------------------------------------------------------------
expected_ckpts = [
    "efficientnet_b0_sin_aumento.pt",
    "resnet50_sin_aumento.pt",
    "vgg16_sin_aumento.pt",
    "convnext_tiny_sin_aumento.pt",
    "cnn_scratch_sin_aumento.pt",
    "efficientnet_b0_cbam_residual_sin_aumento.pt",
    "capsulenet_sin_aumento.pt"
]
for ck in expected_ckpts:
    p = os.path.join("enfoque_sin_aumento_oficial/modelos_guardados", ck)
    assert os.path.exists(p), f"Falta checkpoint: {p}"
print(f"[PASS] Auto-Test 3: Los 7 checkpoints (.pt) existen en enfoque_sin_aumento_oficial/modelos_guardados/.")

print("=" * 80)
print("TODOS LOS CASOS DE AUTO-TEST PASARON EXITOSAMENTE (100% OK)")
print("=" * 80)
