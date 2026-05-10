"""
train.py — Entrena SVM, XGBoost y Logistic Regression sobre las features.

Entrena con age_sub (7 clases finas) y reporta accuracy en age_bucket (3 clases).
Esto mejora la clase 'young' porque internamente distingue bebé de adolescente.

Uso:
    python src/train.py
"""

import argparse
import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SUB_TO_BUCKET = {
    "baby": "young", "child": "young", "teen": "young", "young_adult": "young",
    "middle_adult": "adult", "mature_adult": "adult", "old": "old",
}


def get_X_y(df, target):
    meta = {"filename", "age", "age_bucket", "age_sub", "gender", "ethnicity", "source", "split"}
    feat_cols = [c for c in df.columns if c not in meta]
    return df[feat_cols].values, df[target].values


def main():
    p = argparse.ArgumentParser(description="Entrena clasificadores de edad.")
    p.add_argument("--manifest", default="data/final/manifest.csv", type=Path)
    p.add_argument("--features", default="data/final/features.csv", type=Path)
    p.add_argument("--output-dir", default="models", type=Path)
    p.add_argument("--target", default="age_sub", choices=["age_bucket", "age_sub"])
    p.add_argument("--pca", type=int, default=200, help="Componentes PCA (0 = sin PCA)")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar
    manifest = pd.read_csv(args.manifest)
    features = pd.read_csv(args.features)
    df = manifest.merge(features, on="filename")
    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"]
    logger.info(f"Train: {len(df_train):,}  Val: {len(df_val):,}")

    X_train, y_train = get_X_y(df_train, args.target)
    X_val, y_val = get_X_y(df_val, args.target)

    # Escalar
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # PCA
    pca = None
    if args.pca > 0:
        pca = PCA(n_components=min(args.pca, X_train.shape[1]), random_state=42)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        logger.info(f"PCA: {pca.n_components_} componentes, {pca.explained_variance_ratio_.sum()*100:.1f}% varianza")

    # Modelos
    models = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=42),
        "svm_rbf": SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
                       probability=True, random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                           n_jobs=-1, random_state=42, verbosity=0)
    except ImportError:
        logger.warning("XGBoost no instalado, saltando")

    results = {}
    for name, model in models.items():
        logger.info(f"\n--- {name} ---")
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        logger.info(f"  Train: {train_acc:.4f}  Val: {val_acc:.4f}")

        if args.target == "age_sub":
            pred_bucket = np.array([SUB_TO_BUCKET[p] for p in model.predict(X_val)])
            real_bucket = np.array([SUB_TO_BUCKET[p] for p in y_val])
            bucket_acc = accuracy_score(real_bucket, pred_bucket)
            logger.info(f"  Val (3 buckets): {bucket_acc:.4f}")
            results[name] = {"train": train_acc, "val": val_acc, "val_buckets": bucket_acc}
        else:
            results[name] = {"train": train_acc, "val": val_acc}

        print(classification_report(y_val, model.predict(X_val), zero_division=0))
        joblib.dump(model, args.output_dir / f"{name}.joblib")

    # Guardar preprocesador
    joblib.dump({"scaler": scaler, "pca": pca, "target": args.target},
                args.output_dir / "preprocessor.joblib")

    with open(args.output_dir / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n=== Resumen ===")
    print(pd.DataFrame(results).T.to_string())


if __name__ == "__main__":
    main()
