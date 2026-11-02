import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, brier_score_loss, confusion_matrix
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo activo de entrenamiento: {device}")
if device.type == 'cuda':
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    print(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# Ensure output directories
os.makedirs('modelos/checkpoints_gpu', exist_ok=True)
os.makedirs('results/gpu_predictions', exist_ok=True)

# 1. Dataset class
class IDRiDDataset(Dataset):
    def __init__(self, df, img_dirs, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dirs = img_dirs
        self.transform = transform
        self.valid_indices = []
        self.paths = []
        
        for idx, row in self.df.iterrows():
            name = str(row['Image name'])
            found = None
            for d in self.img_dirs:
                for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                    p = os.path.join(d, name + ext)
                    if os.path.exists(p):
                        found = p
                        break
                if found:
                    break
            if found:
                self.valid_indices.append(idx)
                self.paths.append(found)
        
        self.df = self.df.iloc[self.valid_indices].reset_index(drop=True)
        print(f"Dataset cargado: {len(self.df)} imágenes válidas encontradas.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert('RGB')
        label = float(self.df.iloc[idx]['binary_label'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32), self.df.iloc[idx]['Image name']

# 2. Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Model builder
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def build_model(model_name):
    if model_name == 'ResNet50':
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Freeze initial layers
        for param in list(m.parameters())[:-15]:
            param.requires_grad = False
        m.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(m.fc.in_features, 1)
        )
        return m
    elif model_name == 'EfficientNetB0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        for param in list(m.parameters())[:-15]:
            param.requires_grad = False
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.classifier[1].in_features, 1)
        )
        return m
    elif model_name == 'VGG16':
        m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        for param in list(m.parameters())[:-10]:
            param.requires_grad = False
        m.classifier[6] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(4096, 1)
        )
        return m
    elif model_name == 'ConvNeXt-Tiny':
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        for param in list(m.parameters())[:-15]:
            param.requires_grad = False
        m.classifier[2] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(m.classifier[2].in_features, 1)
        )
        return m
    elif model_name == 'CNN':
        return CustomCNN()
    else:
        raise ValueError(f"Modelo no reconocido: {model_name}")

# 4. Bootstrap CI 95%
def compute_bootstrap_metrics(y_true, y_prob, threshold=0.5, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = recall_score(y_true == 0, y_pred == 0, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except:
        roc = 0.5
    try:
        p_c, r_c, _ = precision_recall_curve(y_true, y_prob)
        pr = auc(r_c, p_c)
    except:
        pr = 0.5
    brier = brier_score_loss(y_true, y_prob)
    
    b_acc, b_sens, b_spec, b_f1, b_roc = [], [], [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        yt = y_true[idx]
        yp = y_prob[idx]
        yh = (yp >= threshold).astype(int)
        
        b_acc.append(accuracy_score(yt, yh))
        b_sens.append(recall_score(yt, yh, zero_division=0))
        b_spec.append(recall_score(yt == 0, yh == 0, zero_division=0))
        b_f1.append(f1_score(yt, yh, zero_division=0))
        try:
            b_roc.append(roc_auc_score(yt, yp))
        except:
            pass
            
    def get_ci(arr):
        if len(arr) == 0:
            return (0.0, 1.0)
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
        
    return {
        'accuracy': {'val': float(acc), 'ci95': get_ci(b_acc)},
        'sensitivity': {'val': float(sens), 'ci95': get_ci(b_sens)},
        'specificity': {'val': float(spec), 'ci95': get_ci(b_spec)},
        'precision': float(prec),
        'f1_score': {'val': float(f1), 'ci95': get_ci(b_f1)},
        'roc_auc': {'val': float(roc), 'ci95': get_ci(b_roc)},
        'pr_auc': float(pr),
        'brier_score': float(brier),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

# 5. Training loop with Early Stopping on val_loss
def train_and_evaluate_model(model_name, train_loader, val_loader, test_loader, epochs=15, patience=5):
    print(f"\n=======================================================")
    print(f"🚀 Entrenando {model_name} en GPU con Protocolo Idéntico")
    print(f"Criterio Early Stopping: val_loss (patience={patience})")
    print(f"=======================================================")
    
    model = build_model(model_name).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = f'modelos/checkpoints_gpu/{model_name}_best.pth'
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        val_probs, val_trues = [], []
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_probs.extend(probs)
                val_trues.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        val_acc = accuracy_score(val_trues, (np.array(val_probs) >= 0.5).astype(int))
        elapsed = time.time() - t0
        print(f"Época [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ⭐ Nuevo mejor val_loss: {val_loss:.4f}. Checkpoint guardado.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  🛑 Early Stopping activado en época {epoch} (sin mejora en val_loss por {patience} épocas).")
                break
                
    # Load best weights
    print(f"Cargando mejores pesos para evaluación final desde {best_model_path}...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # Final eval on Validation set
    v_probs, v_trues = [], []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            v_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            v_trues.extend(labels.numpy())
            
    val_results = compute_bootstrap_metrics(np.array(v_trues), np.array(v_probs), threshold=0.5)
    
    # Final eval on Test set (N=83)
    t_probs, t_trues, t_names = [], [], []
    with torch.no_grad():
        for images, labels, names in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            t_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            t_trues.extend(labels.numpy())
            t_names.extend(names)
            
    test_results = compute_bootstrap_metrics(np.array(t_trues), np.array(t_probs), threshold=0.5)
    
    # Save predictions CSV
    preds_df = pd.DataFrame({
        'Image name': t_names,
        'binary_label': t_trues,
        'probability': t_probs,
        'pred_label_05': (np.array(t_probs) >= 0.5).astype(int)
    })
    preds_df.to_csv(f'results/gpu_predictions/{model_name}_test_preds.csv', index=False)
    
    return {
        'best_val_loss': float(best_val_loss),
        'validation': val_results,
        'test': test_results
    }

def main():
    print("Iniciando pipeline de entrenamiento con GPU...")
    # Load splits
    split_data = joblib.load('data/splits/split_80_20.joblib')
    train_df = split_data['train_images_df']
    val_df = split_data['val_images_df']
    test_df = pd.read_csv('data/splits/test_images_original.csv')
    
    img_dirs = [
        os.path.join('data', 'IDRiD2', 'IDRiD', 'train', 'images'),
        os.path.join('data', 'IDRiD2', 'IDRiD', 'test', 'images')
    ]
    
    train_ds = IDRiDDataset(train_df, img_dirs, transform=train_transform)
    val_ds = IDRiDDataset(val_df, img_dirs, transform=eval_transform)
    test_ds = IDRiDDataset(test_df, img_dirs, transform=eval_transform)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    
    models_to_train = [
        'ResNet50',
        'EfficientNetB0',
        'VGG16',
        'CNN',
        'ConvNeXt-Tiny'
    ]
    
    all_results = {}
    for name in models_to_train:
        t_start = time.time()
        res = train_and_evaluate_model(name, train_loader, val_loader, test_loader, epochs=12, patience=4)
        all_results[name] = res
        print(f"Modelo {name} completado en {time.time() - t_start:.1f}s.")
        
    # Save benchmark JSON
    with open('results/benchmark_gpu_val_test_ic95.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
        
    # Generate Markdown Table
    md = [
        "# Benchmark Consolidado GPU: Validación y Test con IC 95%",
        f"> **Hardware Utilizado:** {torch.cuda.get_device_name(0)} (NVIDIA RTX 4050 6GB)",
        "> **Protocolo Idéntico:** Early Stopping en `val_loss` (patience=4), umbral fijo $\\tau = 0.5$. Data augmentation SOLO en train.",
        "",
        "## 1. Conjunto de Test Independiente (N = 83 originales)",
        "",
        "| Modelo | Accuracy [IC 95%] | Sensibilidad [IC 95%] | Especificidad [IC 95%] | F1-Score [IC 95%] | AUC-ROC [IC 95%] | PR-AUC | Brier |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]
    
    for name, d in all_results.items():
        t = d['test']
        acc = f"{t['accuracy']['val']:.4f} [{t['accuracy']['ci95'][0]:.3f}, {t['accuracy']['ci95'][1]:.3f}]"
        sens = f"{t['sensitivity']['val']:.4f} [{t['sensitivity']['ci95'][0]:.3f}, {t['sensitivity']['ci95'][1]:.3f}]"
        spec = f"{t['specificity']['val']:.4f} [{t['specificity']['ci95'][0]:.3f}, {t['specificity']['ci95'][1]:.3f}]"
        f1 = f"{t['f1_score']['val']:.4f} [{t['f1_score']['ci95'][0]:.3f}, {t['f1_score']['ci95'][1]:.3f}]"
        roc = f"{t['roc_auc']['val']:.4f} [{t['roc_auc']['ci95'][0]:.3f}, {t['roc_auc']['ci95'][1]:.3f}]"
        pr = f"{t['pr_auc']:.4f}"
        brier = f"{t['brier_score']:.4f}"
        md.append(f"| **{name}** | {acc} | {sens} | {spec} | {f1} | {roc} | {pr} | {brier} |")
        
    md.extend([
        "",
        "## 2. Conjunto de Validación Interna (Época Óptima por `val_loss`)",
        "",
        "| Modelo | val_loss | Accuracy [IC 95%] | Sensibilidad [IC 95%] | Especificidad [IC 95%] | F1-Score [IC 95%] | AUC-ROC [IC 95%] | PR-AUC |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ])
    
    for name, d in all_results.items():
        v = d['validation']
        vloss = f"{d['best_val_loss']:.4f}"
        acc = f"{v['accuracy']['val']:.4f} [{v['accuracy']['ci95'][0]:.3f}, {v['accuracy']['ci95'][1]:.3f}]"
        sens = f"{v['sensitivity']['val']:.4f} [{v['sensitivity']['ci95'][0]:.3f}, {v['sensitivity']['ci95'][1]:.3f}]"
        spec = f"{v['specificity']['val']:.4f} [{v['specificity']['ci95'][0]:.3f}, {v['specificity']['ci95'][1]:.3f}]"
        f1 = f"{v['f1_score']['val']:.4f} [{v['f1_score']['ci95'][0]:.3f}, {v['f1_score']['ci95'][1]:.3f}]"
        roc = f"{v['roc_auc']['val']:.4f} [{v['roc_auc']['ci95'][0]:.3f}, {v['roc_auc']['ci95'][1]:.3f}]"
        pr = f"{v['pr_auc']:.4f}"
        md.append(f"| **{name}** | {vloss} | {acc} | {sens} | {spec} | {f1} | {roc} | {pr} |")
        
    with open('results/benchmark_gpu_val_test_ic95.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
        
    print("\n🎉 ¡ENTRENAMIENTO Y EVALUACIÓN GPU COMPLETADOS CON ÉXITO TOTAL!")
    print("Resultados guardados en results/benchmark_gpu_val_test_ic95.md y .json")

if __name__ == '__main__':
    main()
