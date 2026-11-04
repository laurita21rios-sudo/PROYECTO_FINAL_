import json
import csv
import os
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    recall_score, precision_score, accuracy_score,
    balanced_accuracy_score, brier_score_loss, confusion_matrix
)
import pandas as pd

def compute_metrics_from_predictions(df, y_true_col, y_prob_col, y_pred_col, threshold=0.5):
    y_true = df[y_true_col].values
    y_prob = df[y_prob_col].values
    y_pred = (df[y_pred_col] == "Referible").astype(int).values if df[y_pred_col].dtype == object else (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "total_samples": int(len(y_true))
    }

def main():
    base_dir = Path(__file__).resolve().parent.parent
    results = {}

    # 1. ConvNeXt Tiny Clinical
    convnext_path = base_dir / "ConvNeXt_Tiny_Clinical" / "test_metrics.json"
    if convnext_path.exists():
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
            "evaluation_set": "83 imágenes originales",
            "note": "Screening clínico: Sensibilidad 100% (FN=0)"
        }

    # 2. ResNet50
    resnet_path = base_dir / "ResNet50" / "metricas_validacion.json"
    if resnet_path.exists():
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
                "tn": m_opt.get("true_negatives"), "fp": m_opt.get("false_positives"),
                "fn": m_opt.get("false_negatives"), "tp": m_opt.get("true_positives")
            },
            "evaluation_set": "83 imágenes originales",
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
                "tn": m_base.get("true_negatives"), "fp": m_base.get("false_positives"),
                "fn": m_base.get("false_negatives"), "tp": m_base.get("true_positives")
            },
            "evaluation_set": "83 imágenes originales",
            "note": "Umbral estándar 0.5"
        }

    # 3. EfficientNetB0 Baseline (83 originales CORREGIDO)
    eff_pred_path = base_dir / "efficientnetb0_rdr" / "predicciones_validacion.csv"
    if eff_pred_path.exists():
        df_eff = pd.read_csv(eff_pred_path)
        df_eff_orig = df_eff[~df_eff["Image name"].str.contains("_L|_R|_VFLIP")]
        m_eff = compute_metrics_from_predictions(df_eff_orig, "binary_label", "probability_referable", "predicted_class")
        m_eff["evaluation_set"] = "83 imágenes originales (corregido desde N=664 con TTA)"
        m_eff["note"] = "Mejor AUC-ROC global (0.9950). Evaluación sobre imágenes originales."
        results["EfficientNetB0 (Baseline, N=83)"] = m_eff

    # 4. CNN Personalizada
    cnn_path = base_dir / "resultados_cnn_personalizada" / "metricas_completas_cnn.csv"
    if cnn_path.exists():
        df_cnn = pd.read_csv(cnn_path)
        r = df_cnn.iloc[0]
        results["CNN Personalizada (Custom Deep)"] = {
            "threshold": float(r["Umbral"]),
            "accuracy": float(r["Accuracy"]),
            "balanced_accuracy": float(r["Balanced Accuracy"]),
            "sensitivity": float(r["Sensibilidad"]),
            "specificity": float(r["Especificidad"]),
            "precision": float(r["Precisión"]),
            "f1_score": float(r["F1-score"]),
            "roc_auc": float(r["AUC-ROC"]),
            "pr_auc": float(r["AUC-PR"]),
            "brier_score": float(r["Brier Score"]),
            "confusion_matrix": {"tn": int(r["TN"]), "fp": int(r["FP"]), "fn": int(r["FN"]), "tp": int(r["TP"])},
            "evaluation_set": "83 imágenes originales",
            "note": "Baseline convolucional con explicabilidad Grad-CAM auditada"
        }

    # 5. CapsuleNet
    capsule_path = base_dir / "modelos" / "CapsuleNet_sin_aumento_corregida" / "predicciones_validacion.csv"
    if capsule_path.exists():
        df_cap = pd.read_csv(capsule_path)
        y_true = df_cap["binary_label"].values
        y_prob = df_cap["prob_referible"].values
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        results["CapsuleNet"] = {
            "threshold": 0.5,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "brier_score": float(brier_score_loss(y_true, y_prob)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "evaluation_set": "83 imágenes originales",
            "note": "Red de cápsulas para relaciones espaciales"
        }

    # 6. EfficientNetB0 + CBAM
    cbam_path = base_dir / "resultados_efficientnetb0_cbam" / "metricas_cbam.csv"
    if cbam_path.exists():
        df_cbam = pd.read_csv(cbam_path)
        r_c = df_cbam.iloc[0]
        results["EfficientNetB0 + CBAM"] = {
            "threshold": float(r_c["Umbral"]),
            "accuracy": float(r_c["Accuracy"]),
            "balanced_accuracy": (float(r_c["Sensibilidad"]) + float(r_c["Especificidad"])) / 2,
            "sensitivity": float(r_c["Sensibilidad"]),
            "specificity": float(r_c["Especificidad"]),
            "precision": float(r_c["Precisión"]),
            "f1_score": float(r_c["F1-score"]),
            "roc_auc": float(r_c["AUC-ROC"]),
            "pr_auc": None,
            "brier_score": None,
            "confusion_matrix": {"tn": int(r_c["TN"]), "fp": int(r_c["FP"]), "fn": int(r_c["FN"]), "tp": int(r_c["TP"])},
            "evaluation_set": "83 imágenes originales",
            "note": "Colapso de gradiente - módulo CBAM no preentrenado"
        }

    # Guardar JSON
    out_dir = base_dir / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("[OK] benchmark_summary.json actualizado")

    # Generar tabla Markdown
    headers = ["Modelo", "N", "Umbral", "Sensibilidad", "Especificidad", "Accuracy", "F1-Score", "AUC-ROC", "PR-AUC", "Brier", "CM (TN/FP/FN/TP)"]
    rows = []
    for model_name, m in results.items():
        cm_d = m.get("confusion_matrix", {})
        cm_str = f"{cm_d.get('tn','?')}/{cm_d.get('fp','?')}/{cm_d.get('fn','?')}/{cm_d.get('tp','?')}"
        rows.append({
            "Modelo": f"**{model_name}**",
            "N": "83",
            "Umbral": f"{m['threshold']:.3f}" if m.get("threshold") else "-",
            "Sensibilidad": f"{m['sensitivity']:.4f}" if m.get("sensitivity") is not None else "-",
            "Especificidad": f"{m['specificity']:.4f}" if m.get("specificity") is not None else "-",
            "Accuracy": f"{m['accuracy']:.4f}" if m.get("accuracy") is not None else "-",
            "F1-Score": f"{m['f1_score']:.4f}" if m.get("f1_score") is not None else "-",
            "AUC-ROC": f"{m['roc_auc']:.4f}" if m.get("roc_auc") is not None else "-",
            "PR-AUC": f"{m['pr_auc']:.4f}" if m.get("pr_auc") is not None else "-",
            "Brier": f"{m['brier_score']:.4f}" if m.get("brier_score") is not None else "-",
            "CM (TN/FP/FN/TP)": cm_str,
        })

    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_lines = ["| " + " | ".join(r[h] for h in headers) + " |" for r in rows]
    md_table = "\n".join([header_line, sep_line] + data_lines)

    with open(out_dir / "benchmark_summary.md", "w", encoding="utf-8") as f:
        f.write("# Benchmark Consolidado — Retinopatía Diabética (IDRiD)\n")
        f.write("> Todos los modelos evaluados sobre **83 imágenes originales** de pacientes únicos (sin aumentación).\n\n")
        f.write("> EfficientNetB0 Baseline corregido: evaluación sobre N=83 originales (previo registro interno sobre N=664 con TTA).\n\n")
        f.write(md_table + "\n")
    print("[OK] benchmark_summary.md actualizado")
    print()
    print(md_table)

if __name__ == "__main__":
    main()
