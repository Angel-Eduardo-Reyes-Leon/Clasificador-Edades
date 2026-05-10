"""
evaluate.py — Evalúa modelos en test set con métricas + audit de sesgo demográfico.

Genera confusion matrices, accuracy por clase, y accuracy por etnia/género.

Uso:
    python src/evaluate.py
"""

import argparse
import json
import logging
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SUB_TO_BUCKET = {
    "baby": "young", "child": "young", "teen": "young", "young_adult": "young",
    "middle_adult": "adult", "mature_adult": "adult", "old": "old",
}


def get_X(df):
    meta = {"filename", "age", "age_bucket", "age_sub", "gender", "ethnicity", "source", "split"}
    feat_cols = [c for c in df.columns if c not in meta]
    return df[feat_cols].values


def plot_cm(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Conteo"); axes[0].set_xlabel("Predicción"); axes[0].set_ylabel("Real")
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("Porcentaje (recall)"); axes[1].set_xlabel("Predicción")
    plt.suptitle(title, fontweight="bold"); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


def main():
    p = argparse.ArgumentParser(description="Evalúa modelos en test set.")
    p.add_argument("--manifest", default="data/final/manifest.csv", type=Path)
    p.add_argument("--features", default="data/final/features.csv", type=Path)
    p.add_argument("--models-dir", default="models", type=Path)
    p.add_argument("--output-dir", default="reports", type=Path)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar datos
    manifest = pd.read_csv(args.manifest)
    features = pd.read_csv(args.features)
    df = manifest.merge(features, on="filename")
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    logger.info(f"Test set: {len(df_test):,}")

    # Preprocesador
    pre = joblib.load(args.models_dir / "preprocessor.joblib")
    X_test = pre["scaler"].transform(get_X(df_test))
    if pre["pca"]:
        X_test = pre["pca"].transform(X_test)
    target = pre["target"]

    # Evaluar cada modelo
    model_files = [m for m in sorted(args.models_dir.glob("*.joblib")) if m.name != "preprocessor.joblib"]
    all_results = []

    for mpath in model_files:
        name = mpath.stem
        model = joblib.load(mpath)
        logger.info(f"\n--- {name} ---")

        y_pred = model.predict(X_test)
        dt = df_test.copy()

        if target == "age_sub":
            dt["pred_bucket"] = [SUB_TO_BUCKET[p] for p in y_pred]
        else:
            dt["pred_bucket"] = y_pred

        acc = accuracy_score(dt["age_bucket"], dt["pred_bucket"])
        logger.info(f"  Accuracy (3 buckets): {acc:.4f}")
        print(classification_report(dt["age_bucket"], dt["pred_bucket"],
                                    labels=["young", "adult", "old"], zero_division=0))

        # Confusion matrix
        plot_cm(dt["age_bucket"], dt["pred_bucket"], ["young", "adult", "old"],
                f"Confusion Matrix — {name}", args.output_dir / f"cm_{name}.png")

        # Audit de sesgo
        logger.info("  Audit por etnia:")
        audit_eth = dt.groupby("ethnicity").apply(
            lambda g: pd.Series({"n": len(g), "acc": (g["pred_bucket"] == g["age_bucket"]).mean()})).round(3)
        print(audit_eth.to_string())

        logger.info("\n  Audit por género:")
        audit_gen = dt.groupby("gender").apply(
            lambda g: pd.Series({"n": len(g), "acc": (g["pred_bucket"] == g["age_bucket"]).mean()})).round(3)
        print(audit_gen.to_string())

        # Heatmap etnia × edad
        pivot = dt.groupby(["ethnicity", "age_bucket"]).apply(
            lambda g: (g["pred_bucket"] == g["age_bucket"]).mean()).unstack()
        plt.figure(figsize=(7, 4))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.3, vmax=1.0)
        plt.title(f"Accuracy por etnia × edad — {name}"); plt.tight_layout()
        plt.savefig(args.output_dir / f"audit_{name}.png", dpi=120); plt.close()

        all_results.append({"model": name, "accuracy": acc})

    # Resumen
    logger.info("\n=== Comparación ===")
    print(pd.DataFrame(all_results).sort_values("accuracy", ascending=False).to_string(index=False))

    with open(args.output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Gráficas en: {args.output_dir}/")


if __name__ == "__main__":
    main()
