import json
import glob
from pathlib import Path
import pandas as pd

print("=== BUSCANDO TODAS LAS MATRICES DE CONFUSION ===")

for jf in glob.glob("**/*.json", recursive=True):
    if ".venv" in jf or ".git" in jf:
        continue
    try:
        with open(jf, "r", encoding="utf-8") as f:
            d = json.load(f)
        def search_cm(obj, prefix=""):
            if isinstance(obj, dict):
                if all(k in obj for k in ["tn", "fp", "fn", "tp"]):
                    total = obj["tn"] + obj["fp"] + obj["fn"] + obj["tp"]
                    print(f"JSON {jf} [{prefix}]: TN={obj['tn']}, FP={obj['fp']}, FN={obj['fn']}, TP={obj['tp']} => TOTAL={total}")
                elif all(k in obj for k in ["true_negatives", "false_positives", "false_negatives", "true_positives"]):
                    total = obj["true_negatives"] + obj["false_positives"] + obj["false_negatives"] + obj["true_positives"]
                    print(f"JSON {jf} [{prefix}]: TN={obj['true_negatives']}, FP={obj['false_positives']}, FN={obj['false_negatives']}, TP={obj['true_positives']} => TOTAL={total}")
                for k, v in obj.items():
                    search_cm(v, prefix=f"{prefix}.{k}" if prefix else k)
        search_cm(d)
    except Exception:
        pass

for cf in glob.glob("**/*.csv", recursive=True):
    if ".venv" in cf or ".git" in cf or "data" in cf:
        continue
    try:
        df = pd.read_csv(cf)
        cols_lower = [c.lower() for c in df.columns]
        if any(c in cols_lower for c in ["tn", "true_negatives"]):
            for idx, row in df.iterrows():
                row_dict = {str(k).lower(): v for k, v in row.items()}
                tn = row_dict.get("tn", row_dict.get("true_negatives"))
                fp = row_dict.get("fp", row_dict.get("false_positives"))
                fn = row_dict.get("fn", row_dict.get("false_negatives"))
                tp = row_dict.get("tp", row_dict.get("true_positives"))
                model = row_dict.get("modelo", row_dict.get("model", f"Row {idx}"))
                if pd.notna(tn) and pd.notna(tp):
                    total = int(tn) + int(fp) + int(fn) + int(tp)
                    print(f"CSV {cf} [{model}]: TN={int(tn)}, FP={int(fp)}, FN={int(fn)}, TP={int(tp)} => TOTAL={total}")
    except Exception:
        pass
