"""
evaluate.py
============

PASO 8 del pipeline (final del proyecto, generación de reportes).

QUÉ HACE:
    1. Carga los modelos entrenados y el preprocesador
    2. Evalúa cada modelo en el TEST set (que no se tocó hasta ahora)
    3. Genera:
       - Confusion matrix por modelo
       - Classification report (precision, recall, F1 por clase)
       - AUDIT DE SESGO: accuracy por etnia, género y combinaciones
       - Gráficas guardadas en reports/
    4. Imprime tabla comparativa de modelos

POR QUÉ EL AUDIT DE SESGO ES IMPORTANTE:
    El profe pidió "validar con personas conocidas", pero eso es un sanity
    check, no evaluación seria. Una evaluación profesional reporta:
    - Accuracy global (lo obvio)
    - Accuracy por clase (¿el modelo es peor en 'old'?)
    - Accuracy por demografía (¿el modelo es peor en mujeres negras?)
    
    Ese análisis demográfico es lo que distingue un proyecto de papers
    serios (FairFace, FaceNet) de uno amateur. Súbelo al reporte y le
    sube la nota al equipo.

USO:
    python src/evaluate.py \\
        --manifest data/final/manifest.csv \\
        --features data/final/features.csv \\
        --models-dir models \\
        --output-dir reports
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Reusa el mapeo de train.py
SUB_TO_BUCKET = {
    "baby": "young", "child": "young", "teen": "young", "young_adult": "young",
    "middle_adult": "adult", "mature_adult": "adult", "old": "old",
}


# =============================================================================
# Helpers (similares a train.py, pero re-implementados acá para no acoplar)
# =============================================================================
def get_X_y(df, target_col):
    manifest_cols = {"filename", "age", "age_bucket", "age_sub",
                     "gender", "ethnicity", "source", "split", "is_duplicate"}
    feature_cols = [c for c in df.columns if c not in manifest_cols]
    return df[feature_cols].values, df[target_col].values


def map_sub_to_bucket(predictions):
    return np.array([SUB_TO_BUCKET[p] for p in predictions])


# =============================================================================
# Visualización
# =============================================================================
def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Guarda confusion matrix bonita usando seaborn."""
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar=False)
    axes[0].set_title("Counts")
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Real")

    # Percentages
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar=False)
    axes[1].set_title("Porcentaje por fila (recall)")
    axes[1].set_xlabel("Predicción")
    axes[1].set_ylabel("Real")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"  Guardado: {save_path}")


def plot_demographic_audit(df_test, model_name, save_path):
    """Heatmap de accuracy por (age_bucket, ethnicity)."""
    pivot = df_test.groupby(["ethnicity", "age_bucket"]).apply(
        lambda g: (g["pred_bucket"] == g["age_bucket"]).mean()
    ).unstack()

    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0.3, vmax=1.0, cbar_kws={"label": "Accuracy"})
    plt.title(f"Accuracy por etnia × edad — {model_name}")
    plt.xlabel("Bucket real")
    plt.ylabel("Etnia")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"  Guardado: {save_path}")


# =============================================================================
# Evaluación de un modelo
# =============================================================================
def evaluate_model(model_name, model, X_test, df_test, output_dir, target):
    """Evalúa un modelo y genera todos los reportes."""
    logger.info(f"\n--- Evaluando: {model_name} ---")

    # Predicciones (en el espacio del target con el que se entrenó)
    y_pred = model.predict(X_test)
    df_test = df_test.copy()
    df_test["pred"] = y_pred

    # Si entrenamos en age_sub, mapear a age_bucket para reportar
    if target == "age_sub":
        df_test["pred_bucket"] = map_sub_to_bucket(y_pred)
    else:
        df_test["pred_bucket"] = y_pred

    # ---- Métricas globales en buckets (3 clases) ----
    bucket_classes = ["young", "adult", "old"]
    acc_bucket = accuracy_score(df_test["age_bucket"], df_test["pred_bucket"])
    logger.info(f"  Accuracy global (3 buckets): {acc_bucket:.4f}")

    print(f"\n  Classification report ({model_name}, 3 buckets):")
    print(classification_report(df_test["age_bucket"], df_test["pred_bucket"],
                                labels=bucket_classes, zero_division=0))

    # ---- Confusion matrix de buckets ----
    plot_confusion_matrix(
        df_test["age_bucket"], df_test["pred_bucket"], bucket_classes,
        title=f"Confusion Matrix (3 buckets) — {model_name}",
        save_path=output_dir / f"cm_buckets_{model_name}.png",
    )

    # ---- Si predijo age_sub, también CM de sub-buckets ----
    if target == "age_sub":
        sub_classes = ["baby", "child", "teen", "young_adult",
                       "middle_adult", "mature_adult", "old"]
        plot_confusion_matrix(
            df_test["age_sub"], df_test["pred"], sub_classes,
            title=f"Confusion Matrix (7 sub-buckets) — {model_name}",
            save_path=output_dir / f"cm_subs_{model_name}.png",
        )

    # ---- AUDIT DE SESGO ----
    logger.info(f"\n  AUDIT DE SESGO (accuracy por subgrupo):")

    # Por etnia
    audit_eth = df_test.groupby("ethnicity").apply(
        lambda g: pd.Series({
            "n":       len(g),
            "acc":     (g["pred_bucket"] == g["age_bucket"]).mean(),
        })
    ).round(3)
    print(f"\n  Por etnia:")
    print(audit_eth.to_string())

    # Por género
    audit_gender = df_test.groupby("gender").apply(
        lambda g: pd.Series({
            "n":       len(g),
            "acc":     (g["pred_bucket"] == g["age_bucket"]).mean(),
        })
    ).round(3)
    print(f"\n  Por género:")
    print(audit_gender.to_string())

    # Por bucket × etnia
    audit_cross = df_test.groupby(["age_bucket", "ethnicity"]).apply(
        lambda g: (g["pred_bucket"] == g["age_bucket"]).mean()
    ).unstack().round(3)
    print(f"\n  Accuracy por bucket × etnia:")
    print(audit_cross.to_string())

    # Heatmap del audit
    plot_demographic_audit(df_test, model_name,
                          save_path=output_dir / f"audit_{model_name}.png")

    return {
        "model":        model_name,
        "acc_global":   acc_bucket,
        "audit_eth":    audit_eth.to_dict(),
        "audit_gender": audit_gender.to_dict(),
    }


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Evalúa modelos en el test set.")
    p.add_argument("--manifest", default="data/final/manifest.csv", type=Path)
    p.add_argument("--features", default="data/final/features.csv", type=Path)
    p.add_argument("--models-dir", default="models", type=Path)
    p.add_argument("--output-dir", default="reports", type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Cargar datos -----------------------------------------------------
    logger.info("=== Cargando datos ===")
    manifest = pd.read_csv(args.manifest)
    features = pd.read_csv(args.features)
    df = manifest.merge(features, on="filename", how="inner")

    df_test = df[df["split"] == "test"].reset_index(drop=True)
    logger.info(f"Test set: {len(df_test):,} imágenes")

    # ---- Cargar preprocesador --------------------------------------------
    pre = joblib.load(args.models_dir / "preprocessor.joblib")
    scaler = pre["scaler"]
    pca = pre["pca"]
    target = pre["target"]
    logger.info(f"Preprocesador cargado. Target original: {target}")

    # Separar features y aplicar preprocesamiento
    X_test, _ = get_X_y(df_test, target_col=target)
    X_test = scaler.transform(X_test)
    if pca is not None:
        X_test = pca.transform(X_test)

    # ---- Evaluar cada modelo ---------------------------------------------
    model_files = sorted(args.models_dir.glob("*.joblib"))
    model_files = [m for m in model_files if m.name != "preprocessor.joblib"]

    if not model_files:
        raise SystemExit(f"No encontré modelos en {args.models_dir}")

    all_results = []
    for model_path in model_files:
        model_name = model_path.stem
        model = joblib.load(model_path)
        result = evaluate_model(model_name, model, X_test, df_test,
                                args.output_dir, target)
        all_results.append(result)

    # ---- Resumen comparativo ---------------------------------------------
    logger.info("\n=== COMPARACIÓN DE MODELOS ===")
    summary = pd.DataFrame([{
        "model":      r["model"],
        "acc_global": r["acc_global"],
    } for r in all_results]).sort_values("acc_global", ascending=False)
    print(summary.to_string(index=False))

    # Guardar JSON con todos los resultados
    results_path = args.output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResultados guardados: {results_path}")
    logger.info(f"Gráficas en: {args.output_dir}/")


if __name__ == "__main__":
    main()
