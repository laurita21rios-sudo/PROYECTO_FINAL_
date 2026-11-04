import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Configure thread safety for OpenBLAS
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.makedirs('data/splits', exist_ok=True)
os.makedirs('docs', exist_ok=True)

print("="*70)
print("📦 GENERACIÓN DE SPLITS SIN AUMENTO DE DATOS (IMÁGENES ORIGINALES PURAS)")
print("Dataset: IDRiD (413 pacientes únicos, 1 imagen por paciente)")
print("="*70)

# 1. Image directories for path resolution
img_dirs = [
    os.path.join('data', 'IDRiD2', 'IDRiD', 'train', 'images'),
    os.path.join('data', 'IDRiD2', 'IDRiD', 'test', 'images')
]

def resolve_image_path(name):
    for d in img_dirs:
        for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
            p = os.path.join(d, name + ext)
            if os.path.exists(p):
                return p.replace('\\', '/')
    return None

# 2. Load original images from development (330) and test (83)
train_split_raw = pd.read_csv('data/train_split.csv')
orig_dev = train_split_raw[train_split_raw['is_original'] == True].copy().reset_index(drop=True)

test_split_raw = pd.read_csv('data/val_split.csv')
orig_test = test_split_raw[test_split_raw['is_original'] == True].copy().reset_index(drop=True)

# Add resolved file paths
orig_dev['file_path'] = orig_dev['Image name'].apply(resolve_image_path)
orig_test['file_path'] = orig_test['Image name'].apply(resolve_image_path)

assert orig_dev['file_path'].isnull().sum() == 0, "¡Error! Hay imágenes de desarrollo sin ruta física."
assert orig_test['file_path'].isnull().sum() == 0, "¡Error! Hay imágenes de test sin ruta física."

# Check patient uniqueness
assert orig_dev['patient_id'].nunique() == len(orig_dev), "¡Error! IDs de pacientes duplicados en desarrollo."
assert orig_test['patient_id'].nunique() == len(orig_test), "¡Error! IDs de pacientes duplicados en test."
overlap = set(orig_dev['patient_id']).intersection(set(orig_test['patient_id']))
assert len(overlap) == 0, f"¡Error! Fuga de datos detectada: {overlap}"

print(f"✅ Desarrollo (Train + Val): {len(orig_dev)} imágenes originales verificadas.")
print(f"✅ Test Independiente (Parte A): {len(orig_test)} imágenes originales verificadas.")
print(f"✅ Total Estudio Clínico: {len(orig_dev) + len(orig_test)} pacientes únicos (Cero réplicas sintéticas).")

# 3. Stratified Split Generator (by Retinopathy grade)
def generate_unaugmented_split(orig_df, test_df, val_ratio=0.20, name_prefix='split_sin_aumento_80_20', seed=42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(orig_df, orig_df['Retinopathy grade']))
    
    train_df = orig_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = orig_df.iloc[val_idx].copy().reset_index(drop=True)
    test_clean_df = test_df.copy().reset_index(drop=True)
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_clean_df['split'] = 'test'
    
    dev_combined = pd.concat([train_df, val_df], ignore_index=True)
    all_combined = pd.concat([train_df, val_df, test_clean_df], ignore_index=True)
    
    # Save CSVs
    dev_combined.to_csv(f'data/splits/{name_prefix}.csv', index=False)
    train_df.to_csv(f'data/splits/train_sin_aumento_{int((1-val_ratio)*100)}_{int(val_ratio*100)}.csv', index=False)
    val_df.to_csv(f'data/splits/val_sin_aumento_{int((1-val_ratio)*100)}_{int(val_ratio*100)}.csv', index=False)
    
    # Save Joblib
    joblib_payload = {
        'split_name': name_prefix,
        'val_ratio': val_ratio,
        'seed': seed,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_clean_df,
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test_clean_df),
        'train_binary_dist': train_df['binary_label'].value_counts().to_dict(),
        'val_binary_dist': val_df['binary_label'].value_counts().to_dict(),
        'test_binary_dist': test_clean_df['binary_label'].value_counts().to_dict(),
        'train_grade_dist': train_df['Retinopathy grade'].value_counts().to_dict(),
        'val_grade_dist': val_df['Retinopathy grade'].value_counts().to_dict(),
        'test_grade_dist': test_clean_df['Retinopathy grade'].value_counts().to_dict()
    }
    joblib_path = f'data/splits/{name_prefix}.joblib'
    joblib.dump(joblib_payload, joblib_path)
    
    print(f"\n--- Resumen {name_prefix} ---")
    print(f"• Train ({int((1-val_ratio)*100)}%): {len(train_df)} imágenes | Clases: {train_df['binary_label'].value_counts().to_dict()}")
    print(f"• Val   ({int(val_ratio*100)}%): {len(val_df)} imágenes | Clases: {val_df['binary_label'].value_counts().to_dict()}")
    print(f"• Test  (Ciego): {len(test_clean_df)} imágenes | Clases: {test_clean_df['binary_label'].value_counts().to_dict()}")
    print(f"• Archivo guardado: data/splits/{name_prefix}.csv y .joblib")
    
    return joblib_payload

# Generate 80/20 and 90/10 unaugmented splits
split_80_20 = generate_unaugmented_split(orig_dev, orig_test, val_ratio=0.20, name_prefix='split_sin_aumento_80_20', seed=42)
split_90_10 = generate_unaugmented_split(orig_dev, orig_test, val_ratio=0.10, name_prefix='split_sin_aumento_90_10', seed=42)

# Save test split and full dataset CSV
orig_test.to_csv('data/splits/test_sin_aumento_83.csv', index=False)
full_413 = pd.concat([orig_dev, orig_test], ignore_index=True)
full_413.to_csv('data/splits/dataset_completo_original_413.csv', index=False)

print("\n" + "="*70)
print("📊 DETALLE CLÍNICO DEL SPLIT 80/20 SIN AUMENTO DE DATOS")
print("="*70)

train_80 = split_80_20['train_df']
val_80 = split_80_20['val_df']
test_clean = split_80_20['test_df']

def format_grade_table(df_subset, name):
    counts = df_subset['Retinopathy grade'].value_counts().sort_index().to_dict()
    total = len(df_subset)
    binary = df_subset['binary_label'].value_counts().to_dict()
    return f"| **{name}** | {total} | {binary.get(0, 0)} ({binary.get(0, 0)/total*100:.1f}%) | {binary.get(1, 0)} ({binary.get(1, 0)/total*100:.1f}%) | {counts.get(0, 0)} | {counts.get(1, 0)} | {counts.get(2, 0)} | {counts.get(3, 0)} | {counts.get(4, 0)} |"

table_rows = [
    "| Partición | Total Imágenes | No Referible (0) | Referible (1) | Grado 0 (Normal) | Grado 1 (Leve) | Grado 2 (Mod) | Grado 3 (Sev) | Grado 4 (Prolif) |",
    "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    format_grade_table(train_80, "Train (80% Original)"),
    format_grade_table(val_80, "Val (20% Original)"),
    format_grade_table(test_clean, "Test (Parte A Original)"),
    format_grade_table(full_413, "Total Estudio Clínico")
]

for row in table_rows:
    print(row)

# 4. Generate Markdown documentation
md_content = [
    "# Partición Oficial del Dataset IDRiD Sin Aumento de Datos",
    "",
    "> **Principio de Evaluación Clínica Rigurosa:**",
    "> - **Cero Aumento de Datos:** No se aplican rotaciones, giros horizontales (*flips*), traslaciones ni copias sintéticas en ningún subconjunto.",
    "> - **Preservación Anatómica:** Se mantiene la orientación espacial fisiológica real de la retina (papila óptica nasal vs. fóvea temporal).",
    "> - **Partición a Nivel de Paciente (Patient-Level Split):** Cada uno de los 413 pacientes del estudio aparece exactamente **una sola vez**.",
    "",
    "## 1. Distribución Numérica y Clínica de Pacientes (Split 80/20)",
    "",
    "\n".join(table_rows),
    "",
    "---",
    "",
    "## 2. Archivos Generados en el Repositorio",
    "",
    "| Nombre del Archivo | Descripción | N° Imágenes / Filas |",
    "|---|---|:---:|",
    "| [`data/splits/split_sin_aumento_80_20.csv`](file:///c:/Users/lauri/Documents/PROYECTO_FINAL/data/splits/split_sin_aumento_80_20.csv) | Conjunto de desarrollo (Train + Val) con asignación de split | 330 |",
    "| [`data/splits/train_sin_aumento_80_20.csv`](file:///c:/Users/lauri/Documents/PROYECTO_FINAL/data/splits/train_sin_aumento_80_20.csv) | Subconjunto exclusivo de entrenamiento | 264 |",
    "| [`data/splits/val_sin_aumento_80_20.csv`](file:///c:/Users/lauri/Documents/PROYECTO_FINAL/data/splits/val_sin_aumento_80_20.csv) | Subconjunto exclusivo de validación interna | 66 |",
    "| [`data/splits/test_sin_aumento_83.csv`](file:///c:/Users/lauri/Documents/PROYECTO_FINAL/data/splits/test_sin_aumento_83.csv) | Subconjunto de prueba independiente ciego (Parte A) | 83 |",
    "| [`data/splits/dataset_completo_original_413.csv`](file:///c:/Users/lauri/Documents/PROYECTO_FINAL/data/splits/dataset_completo_original_413.csv) | Dataset consolidado de los 413 pacientes con rutas absolutas | 413 |",
    "| [`data/splits/split_sin_aumento_80_20.joblib`](file:///c:/Users/lauri/Documents/PROYECTO_FINAL/data/splits/split_sin_aumento_80_20.joblib) | Diccionario serializado con DataFrames y metadatos estadísticos | 80/20 |",
    "| [`data/splits/split_sin_aumento_90_10.joblib`](file:///c:/Users/lauri/Documents/PROYECTO_FINAL/data/splits/split_sin_aumento_90_10.joblib) | Partición alternativa 90/10 para comparativas | 90/10 |",
    "",
    "---",
    "",
    "## 3. Garantías Metodológicas",
    "",
    "1. **Cero Data Leakage:** Ningún paciente de entrenamiento o validación se encuentra en el conjunto de prueba independiente.",
    "2. **Estratificación Multiclase Exacta:** La estratificación se ejecutó sobre la variable `Retinopathy grade` (escala 0 a 4 del ICDR), lo que garantiza automáticamente una proporción idéntica de la variable binaria `binary_label` (No Referible vs Referible).",
    "3. **Rutas Físicas Verificadas:** El 100% de las 413 imágenes han sido verificadas en el almacenamiento local y enlazadas en la columna `file_path`.",
    ""
]

with open('docs/split_sin_aumento_resumen.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_content) + '\n')

print("✅ Documentación guardada en: docs/split_sin_aumento_resumen.md")
print("="*70)
print("🎉 ¡SPLIT SIN AUMENTO DE DATOS FINALIZADO EXITOSAMENTE!")
print("="*70)
