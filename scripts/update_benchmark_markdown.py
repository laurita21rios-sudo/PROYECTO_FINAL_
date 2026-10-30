import json

with open('results/benchmark_summary.json', 'r', encoding='utf-8') as f:
    bench = json.load(f)

lines = [
    '# Benchmark Consolidado — Retinopatía Diabética (IDRiD)',
    '> Todos los modelos fueron evaluados rigurosamente sobre las **83 imágenes originales** de pacientes únicos (sin data augmentation en validación/test).',
    '',
    '| Modelo | N | Umbral (τ) | Sensibilidad | Especificidad | Accuracy | F1-Score | AUC-ROC | PR-AUC | Brier Score | CM (TN/FP/FN/TP) |',
    '|---|---|---|---|---|---|---|---|---|---|---|'
]

for model, d in bench.items():
    cm = d.get('confusion_matrix', {})
    cm_str = f"{cm.get('tn','-')}/{cm.get('fp','-')}/{cm.get('fn','-')}/{cm.get('tp','-')}"
    brier = f"{d['brier_score']:.4f}" if d.get('brier_score') is not None else '-'
    pr = f"{d['pr_auc']:.4f}" if d.get('pr_auc') is not None else '-'
    roc = f"{d['roc_auc']:.4f}" if d.get('roc_auc') is not None else '-'
    acc = f"{d['accuracy']:.4f}" if d.get('accuracy') is not None else '-'
    sens = f"{d['sensitivity']:.4f}" if d.get('sensitivity') is not None else '-'
    spec = f"{d['specificity']:.4f}" if d.get('specificity') is not None else '-'
    f1 = f"{d['f1_score']:.4f}" if d.get('f1_score') is not None else '-'
    thresh = f"{d['threshold']:.3f}" if d.get('threshold') is not None else '-'
    n = d.get('total_samples', 83)
    lines.append(f"| **{model}** | {n} | {thresh} | {sens} | {spec} | {acc} | {f1} | {roc} | {pr} | {brier} | {cm_str} |")

with open('results/benchmark_summary.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')

print('results/benchmark_summary.md generado exitosamente.')
