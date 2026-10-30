import os
import json
import csv

def compute_metrics_from_csv(csv_path, threshold=0.5):
    """Calcula métricas binarias puras sin dependencias externas."""
    y_true = []
    y_prob = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(int(float(row["binary_label"])))
            y_prob.append(float(row["prob_referible"]))
    
    tp = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 1 and yp >= threshold)
    fn = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 1 and yp < threshold)
    tn = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 0 and yp < threshold)
    fp = sum(1 for yt, yp in zip(y_true, y_prob) if yt == 0 and yp >= threshold)
    total = len(y_true)

    acc = (tp + tn) / total if total > 0 else 0.0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
    bal_acc = (sens + spec) / 2.0

    return {
        "threshold": float(threshold),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "f1_score": float(f1),
        "roc_auc": None,
        "pr_auc": None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "total_samples": total
    }

def format_markdown_table(rows, headers):
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = []
    for r in rows:
        data_lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join([header_line, separator_line] + data_lines)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results = {}

    # 1. ConvNeXt Tiny Clinical
    convnext_path = os.path.join(base_dir, "ConvNeXt_Tiny_Clinical", "test_metrics.json")
    if os.path.exists(convnext_path):
        with open(convnext_path, "r", encoding="utf-8") as f:
            cdata = json.load(f)
            m = cdata.get("metrics", {})
            ci = cdata.get("confidence_intervals_95", {})
            results["ConvNeXt-Tiny (Clinical)"] = {
                "threshold": m.get("threshold"),
                "accuracy": m.get("accuracy"),
                "balanced_accuracy": m.get("balanced_accuracy"),
                "sensitivity": m.get("sensitivity"),
                "specificity": m.get("specificity"),
                "precision": m.get("precision_ppv"),
                "f1_score": m.get("f1_score"),
                "roc_auc": m.get("roc_auc"),
                "pr_auc": m.get("pr_auc"),
                "brier_score": m.get("brier_score"),
                "confusion_matrix": {"tn": m.get("tn"), "fp": m.get("fp"), "fn": m.get("fn"), "tp": m.get("tp")},
                "confidence_intervals_95": ci,
                "note": "Screening clínico: Sensibilidad 100% (FN=0)"
            }

    # 2. ResNet50 (Fine-Tuning v2)
    resnet_path = os.path.join(base_dir, "ResNet50", "metricas_validacion.json")
    if os.path.exists(resnet_path):
        with open(resnet_path, "r", encoding="utf-8") as f:
            rdata = json.load(f)
            m_opt = rdata.get("metrics_selected_threshold", {})
            results["ResNet50 (Fine-Tuning Opt)"] = {
                "threshold": m_opt.get("threshold"),
                "accuracy": m_opt.get("accuracy"),
                "balanced_accuracy": (m_opt.get("sensitivity_recall", 0) + m_opt.get("specificity", 0)) / 2,
                "sensitivity": m_opt.get("sensitivity_recall"),
                "specificity": m_opt.get("specificity"),
                "precision": m_opt.get("precision"),
                "f1_score": m_opt.get("f1_score"),
                "roc_auc": m_opt.get("roc_auc"),
                "pr_auc": m_opt.get("pr_auc"),
                "brier_score": None,
                "confusion_matrix": {
                    "tn": m_opt.get("true_negatives"),
                    "fp": m_opt.get("false_positives"),
                    "fn": m_opt.get("false_negatives"),
                    "tp": m_opt.get("true_positives")
                },
                "note": "Mayor F1-Score (0.9515) y balance general"
            }
            m_base = rdata.get("metrics_threshold_0_5", {})
            results["ResNet50 (Baseline 0.5)"] = {
                "threshold": 0.5,
                "accuracy": m_base.get("accuracy"),
                "balanced_accuracy": (m_base.get("sensitivity_recall", 0) + m_base.get("specificity", 0)) / 2,
                "sensitivity": m_base.get("sensitivity_recall"),
                "specificity": m_base.get("specificity"),
                "precision": m_base.get("precision"),
                "f1_score": m_base.get("f1_score"),
                "roc_auc": m_base.get("roc_auc"),
                "pr_auc": m_base.get("pr_auc"),
                "brier_score": None,
                "confusion_matrix": {
                    "tn": m_base.get("true_negatives"),
                    "fp": m_base.get("false_positives"),
                    "fn": m_base.get("false_negatives"),
                    "tp": m_base.get("true_positives")
                },
                "note": "Umbral estándar 0.5"
            }

    # 3. CNN Personalizada
    cnn_path = os.path.join(base_dir, "resultados_cnn_personalizada", "metricas_completas_cnn.csv")
    if os.path.exists(cnn_path):
        with open(cnn_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results["CNN Personalizada (Custom Deep)"] = {
                    "threshold": float(row["Umbral"]),
                    "accuracy": float(row["Accuracy"]),
                    "balanced_accuracy": float(row["Balanced Accuracy"]),
                    "sensitivity": float(row["Sensibilidad"]),
                    "specificity": float(row["Especificidad"]),
                    "precision": float(row["Precisión"]),
                    "f1_score": float(row["F1-score"]),
                    "roc_auc": float(row["AUC-ROC"]),
                    "pr_auc": float(row["AUC-PR"]),
                    "brier_score": float(row["Brier Score"]),
                    "confusion_matrix": {"tn": int(row["TN"]), "fp": int(row["FP"]), "fn": int(row["FN"]), "tp": int(row["TP"])},
                    "note": "Baseline con explicabilidad Grad-CAM auditada"
                }
                break

    # 4. CapsuleNet
    capsule_path = os.path.join(base_dir, "modelos", "CapsuleNet_sin_aumento_corregida", "predicciones_validacion.csv")
    if os.path.exists(capsule_path):
        cap_metrics = compute_metrics_from_csv(capsule_path, threshold=0.5)
        cap_metrics["note"] = "Red de cápsulas para relaciones espaciales"
        results["CapsuleNet"] = cap_metrics

    # 5. EfficientNetB0 + CBAM
    cbam_path = os.path.join(base_dir, "resultados_efficientnetb0_cbam", "metricas_cbam.csv")
    if os.path.exists(cbam_path):
        with open(cbam_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results["EfficientNetB0 + CBAM"] = {
                    "threshold": float(row["Umbral"]),
                    "accuracy": float(row["Accuracy"]),
                    "balanced_accuracy": (float(row["Sensibilidad"]) + float(row["Especificidad"])) / 2,
                    "sensitivity": float(row["Sensibilidad"]),
                    "specificity": float(row["Especificidad"]),
                    "precision": float(row["Precisión"]),
                    "f1_score": float(row["F1-score"]),
                    "roc_auc": float(row["AUC-ROC"]),
                    "pr_auc": None,
                    "brier_score": None,
                    "confusion_matrix": {"tn": int(row["TN"]), "fp": int(row["FP"]), "fn": int(row["FN"]), "tp": int(row["TP"])},
                    "note": "Colapso/bajo desempeño de atención CBAM"
                }
                break

    # Crear carpeta results si no existe
    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    # Guardar JSON consolidado
    summary_json_path = os.path.join(out_dir, "benchmark_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[OK] Guardado resumen JSON en: {summary_json_path}")

    # Generar tabla Markdown
    headers = ["Modelo", "Umbral", "Sensibilidad", "Especificidad", "Accuracy", "F1-Score", "AUC-ROC", "PR-AUC", "Propósito / Diagnóstico Clínico"]
    rows = []
    for model_name, m in results.items():
        sens_str = f"{m['sensitivity']:.4f}" if m.get("sensitivity") is not None else "-"
        spec_str = f"{m['specificity']:.4f}" if m.get("specificity") is not None else "-"
        acc_str = f"{m['accuracy']:.4f}" if m.get("accuracy") is not None else "-"
        f1_str = f"{m['f1_score']:.4f}" if m.get("f1_score") is not None else "-"
        roc_str = f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "-"
        pr_str = f"{m['pr_auc']:.4f}" if m.get("pr_auc") is not None else "-"
        thresh_str = f"{m['threshold']:.3f}" if m.get("threshold") is not None else "0.500"
        
        rows.append({
            "Modelo": f"**{model_name}**",
            "Umbral": thresh_str,
            "Sensibilidad": sens_str,
            "Especificidad": spec_str,
            "Accuracy": acc_str,
            "F1-Score": f1_str,
            "AUC-ROC": roc_str,
            "PR-AUC": pr_str,
            "Propósito / Diagnóstico Clínico": m.get("note", "")
        })

    md_table = format_markdown_table(rows, headers)
    summary_md_path = os.path.join(out_dir, "benchmark_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Consolidado de Modelos - Detección de Retinopatía Diabética (IDRiD)\n\n")
        f.write(md_table)
        f.write("\n")
    print(f"[OK] Guardado reporte Markdown en: {summary_md_path}\n")
    print(md_table)

if __name__ == "__main__":
    main()
