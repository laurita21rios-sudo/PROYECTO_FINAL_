import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Benchmark Clínico: Detección de Retinopatía Diabética (IDRiD, N=83 Originales)', fontsize=16, fontweight='bold', y=0.98)

# Read data
with open('results/benchmark_summary.json', 'r', encoding='utf-8') as f:
    bench = json.load(f)

# Select key models
models = [
    'ConvNeXt-Tiny (Clinical)',
    'EfficientNetB0 (Baseline, N=83)',
    'VGG16 (Fine-Tuned)',
    'ResNet50 (Fine-Tuning Opt)',
    'CapsuleNet',
    'CNN Personalizada (Custom Deep)',
    'EfficientNetB0 + CBAM'
]

short_names = [
    'ConvNeXt-Tiny\n(Clinical)',
    'EfficientNetB0\n(Baseline)',
    'VGG16\n(Fine-Tuned)',
    'ResNet50\n(Opt)',
    'CapsuleNet',
    'CNN\nPersonalizada',
    'EfficientNetB0\n+ CBAM'
]

sens = [bench[m]['sensitivity'] for m in models]
spec = [bench[m]['specificity'] for m in models]
f1 = [bench[m]['f1_score'] for m in models]
auc_roc = [bench[m]['roc_auc'] for m in models]
pr_auc = [bench[m]['pr_auc'] if bench[m].get('pr_auc') else 0 for m in models]
fn_counts = [bench[m]['confusion_matrix']['fn'] for m in models]
brier = [bench[m]['brier_score'] if bench[m].get('brier_score') is not None else np.nan for m in models]

colors = ['#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b', '#e377c2', '#d62728']

# 1. Trade-off Clínico: Sensibilidad vs Especificidad
ax1 = axes[0, 0]
for i, m in enumerate(short_names):
    ax1.scatter(spec[i], sens[i], color=colors[i], s=180, zorder=5, label=m.replace('\n', ' '))
    ax1.annotate(m.replace('\n', ' '), (spec[i], sens[i]), textcoords="offset points", xytext=(0, 10),
                 ha='center', fontsize=9, fontweight='bold', color=colors[i])

ax1.axhline(0.95, color='gray', linestyle='--', alpha=0.5, label='Umbral Alta Sensibilidad (95%)')
ax1.axvline(0.90, color='gray', linestyle=':', alpha=0.5, label='Umbral Alta Especificidad (90%)')
ax1.fill_between([0.85, 1.0], 0.90, 1.0, color='green', alpha=0.08, label='Zona Óptima Confirmación')
ax1.fill_between([0.50, 0.85], 0.98, 1.02, color='blue', alpha=0.08, label='Zona Screening (Cero FN)')
ax1.set_xlabel('Especificidad (Reducción de Falsos Positivos)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Sensibilidad / Recall (Detección de Patológicos)', fontsize=11, fontweight='bold')
ax1.set_title('A) Trade-off Clínico: Screening vs. Confirmación', fontsize=12, fontweight='bold')
ax1.set_xlim(0.1, 1.05)
ax1.set_ylim(0.3, 1.05)

# 2. AUC-ROC vs PR-AUC
ax2 = axes[0, 1]
x = np.arange(len(models))
width = 0.35
ax2.bar(x - width/2, auc_roc, width, label='AUC-ROC', color='#1f77b4', alpha=0.85)
ax2.bar(x + width/2, pr_auc, width, label='PR-AUC', color='#ff7f0e', alpha=0.85)
ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Línea de Azar (0.5)')
ax2.set_xticks(x)
ax2.set_xticklabels(short_names, fontsize=8, fontweight='bold')
ax2.set_ylabel('Área bajo la Curva', fontsize=11, fontweight='bold')
ax2.set_title('B) Capacidad Discriminativa (AUC-ROC y PR-AUC)', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.legend(loc='lower left', fontsize=9)

# 3. Falsos Negativos (Riesgo Clínico Crítico)
ax3 = axes[1, 0]
bars = ax3.bar(short_names, fn_counts, color=['#2ca02c' if fn == 0 else ('#d62728' if fn > 10 else '#ff7f0e') for fn in fn_counts], alpha=0.85)
for bar in bars:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{int(yval)}", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.set_ylabel('Casos Patológicos Omitidos (FN)', fontsize=11, fontweight='bold')
ax3.set_title('C) Seguridad de Paciente: Falsos Negativos (Menor es Mejor)', fontsize=12, fontweight='bold')
ax3.set_xticklabels(short_names, fontsize=8, fontweight='bold')
ax3.axhline(0, color='green', linewidth=2)

# 4. F1-Score y Precisión Balanceada
ax4 = axes[1, 1]
b_bars = ax4.bar(short_names, f1, color='#2ca02c', alpha=0.85)
for bar in b_bars:
    yval = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
ax4.set_ylabel('F1-Score Balanceado', fontsize=11, fontweight='bold')
ax4.set_title('D) Desempeño F1-Score Global', fontsize=12, fontweight='bold')
ax4.set_xticklabels(short_names, fontsize=8, fontweight='bold')
ax4.set_ylim(0, 1.15)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_fig = 'results/figures/comparativa_modelos_benchmark.png'
plt.savefig(out_fig, dpi=300)
print(f"Figura guardada exitosamente en: {out_fig}")
