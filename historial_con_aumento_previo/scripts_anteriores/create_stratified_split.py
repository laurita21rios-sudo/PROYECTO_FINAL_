import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# Ensure output directory exists
os.makedirs('data/splits', exist_ok=True)

# Load full train_split
full_train_df = pd.read_csv('data/train_split.csv')
orig_df = full_train_df[full_train_df['is_original'] == True].copy().reset_index(drop=True)

print(f"Total pacientes originales en train: {len(orig_df)}")

def make_split(orig_df, full_df, val_ratio, name_prefix, seed=42):
    # Stratify by Retinopathy grade for fine-grained balance (also perfectly balances binary_label)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(orig_df, orig_df['Retinopathy grade']))
    
    train_patients = orig_df.iloc[train_idx].copy().reset_index(drop=True)
    val_patients = orig_df.iloc[val_idx].copy().reset_index(drop=True)
    
    train_patients['split_assignment'] = 'train'
    val_patients['split_assignment'] = 'val'
    
    combined_patients = pd.concat([train_patients, val_patients], ignore_index=True)
    
    # Map back to full augmented images:
    # Any image belonging to a train patient goes to train_images
    train_patient_ids = set(train_patients['patient_id'])
    val_patient_ids = set(val_patients['patient_id'])
    
    train_images = full_df[full_df['patient_id'].isin(train_patient_ids)].copy().reset_index(drop=True)
    # Validation strictly uses original unaugmented images to avoid inflation
    val_images = val_patients.copy().reset_index(drop=True)
    
    print(f"\n--- Split {name_prefix} (Val ratio: {val_ratio}) ---")
    print(f"Pacientes Train: {len(train_patients)} | Imágenes aumentadas Train: {len(train_images)}")
    print(f"Pacientes Val (Originales): {len(val_patients)} | Imágenes Val: {len(val_images)}")
    print("Distribución de clases en Train:")
    print(train_patients['binary_label'].value_counts(normalize=True).round(3).to_dict())
    print("Distribución de clases en Val:")
    print(val_patients['binary_label'].value_counts(normalize=True).round(3).to_dict())
    
    # Save CSVs
    csv_patient_path = f'data/splits/{name_prefix}_patients.csv'
    combined_patients.to_csv(csv_patient_path, index=False)
    
    csv_train_path = f'data/splits/{name_prefix}_train_augmented.csv'
    train_images.to_csv(csv_train_path, index=False)
    
    csv_val_path = f'data/splits/{name_prefix}_val_original.csv'
    val_images.to_csv(csv_val_path, index=False)
    
    # Save Joblib dictionary
    joblib_data = {
        'split_name': name_prefix,
        'val_ratio': val_ratio,
        'seed': seed,
        'train_patient_ids': list(train_patient_ids),
        'val_patient_ids': list(val_patient_ids),
        'train_patients_df': train_patients,
        'val_patients_df': val_patients,
        'train_images_df': train_images,
        'val_images_df': val_images,
        'metrics_info': {
            'n_train_patients': len(train_patients),
            'n_train_images': len(train_images),
            'n_val_patients': len(val_patients),
            'train_class_dist': train_patients['binary_label'].value_counts().to_dict(),
            'val_class_dist': val_patients['binary_label'].value_counts().to_dict(),
        }
    }
    
    joblib_path = f'data/splits/{name_prefix}.joblib'
    joblib.dump(joblib_data, joblib_path)
    print(f"Guardado CSV: {csv_patient_path}")
    print(f"Guardado Joblib: {joblib_path}")
    return joblib_data

# Create 90/10 split
split_90_10 = make_split(orig_df, full_train_df, val_ratio=0.10, name_prefix='split_90_10', seed=42)

# Create 80/20 split
split_80_20 = make_split(orig_df, full_train_df, val_ratio=0.20, name_prefix='split_80_20', seed=42)

# Save independent test split (83 originals)
test_df = pd.read_csv('data/val_split.csv')
test_df.to_csv('data/splits/test_images_original.csv', index=False)
joblib.dump({'test_df': test_df, 'n_test': len(test_df), 'class_dist': test_df['binary_label'].value_counts().to_dict()},
            'data/splits/test_split.joblib')
print("\n¡Splits generados y guardados exitosamente en data/splits/!")
