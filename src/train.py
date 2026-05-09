"""
train.py
=========

PASO 7 del pipeline (después del merge, cualquiera del equipo lo corre).

QUÉ HACE:
    1. Carga manifest.csv y features.csv y los une por 'filename'
    2. Filtra split == 'train' para entrenar y 'val' para tunear
    3. Estandariza features (StandardScaler) — crítico para SVM
    4. Reduce dimensionalidad con PCA (opcional, ayuda mucho a SVM)
    5. Entrena 3 modelos: Logistic Regression (baseline), SVM RBF, XGBoost
    6. Evalúa en val set y reporta accuracy
    7. Guarda los modelos y el preprocesador para que evaluate.py los use

POR QUÉ ENTRENAR EN age_sub Y EVALUAR EN age_bucket:
    'young' (0-25) abarca desde bebés hasta jóvenes adultos. Visualmente
    son muy distintos. Si entrenamos directo en 3 clases, el modelo se
    confunde porque mezcla bebés con personas de 25 en la misma etiqueta.

    Estrategia: entrenamos con 7 sub-clases finas (baby, child, teen,
    young_adult, middle_adult, mature_adult, old). Al evaluar, mapeamos
    las predicciones del sub-bucket al bucket de 3 clases:

        baby, child, teen, young_adult → young
        middle_adult, mature_adult     → adult
        old                            → old

    Esto suele dar +5-10% accuracy en la clase 'young'.

USO:
    python src/train.py \\
        --manifest data/final/manifest.csv \\
        --features data/final/features.csv \\
        --output-dir models
"""

from __future__ import annotations
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Mapeo de sub-buckets (lo que predice el modelo) a buckets finales (3 clases)
SUB_TO_BUCKET = {
    "baby":         "young",
    "child":        "young",
    "teen":         "young",
    "young_adult":  "young",
    "middle_adult": "adult",
    "mature_adult": "adult",
    "old":          "old",
}


# =============================================================================
# Carga de datos
# =============================================================================
def load_data(manifest_path: Path, features_path: Path) -> pd.DataFrame:
    """Carga y une manifest + features por 'filename'."""
    manifest = pd.read_csv(manifest_path)
    features = pd.read_csv(features_path)
    df = manifest.merge(features, on="filename", how="inner")
    logger.info(f"Datos cargados: {len(df):,} filas, "
                f"{features.shape[1] - 1} features por imagen")
    return df


def get_X_y(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Separa features (X) de target (y). Las columnas de features son las
    que NO pertenecen al manifest.
    """
    manifest_cols = {"filename", "age", "age_bucket", "age_sub",
                     "gender", "ethnicity", "source", "split", "is_duplicate"}
    feature_cols = [c for c in df.columns if c not in manifest_cols]
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y


# =============================================================================
# Pipelines de modelos
# =============================================================================
def build_models() -> dict:
    """
    Devuelve un dict {nombre: modelo_sklearn}.

    Justificación de cada uno:
    - LogisticRegression: baseline obligado. Si los modelos complejos no
      le ganan claramente, probablemente algo está mal.
    - SVC con kernel RBF: clásico ganador en age estimation con LBP/HOG.
      class_weight='balanced' compensa desbalance entre clases.
    - XGBoost: top performer en datos tabulares en general.
    """
    models = {
        "logreg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
    }

    # XGBoost es opcional (puede no estar instalado en algunas máquinas)
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softprob",
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        logger.warning("XGBoost no instalado, saltando ese modelo.")

    return models


# =============================================================================
# Entrenamiento y evaluación
# =============================================================================
def train_and_evaluate(model_name: str, model, X_train, y_train,
                       X_val, y_val, label_encoder=None):
    """
    Entrena un modelo y reporta su accuracy en train y val.
    Devuelve el modelo entrenado y un dict con métricas.
    """
    logger.info(f"\n--- Entrenando: {model_name} ---")
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    logger.info(f"  Train accuracy: {train_acc:.4f}")
    logger.info(f"  Val accuracy:   {val_acc:.4f}")
    if train_acc - val_acc > 0.15:
        logger.warning(f"  ⚠ Posible overfitting (gap = {train_acc - val_acc:.3f})")

    # Reporte detallado en val
    logger.info(f"\n  Classification report en val ({model_name}):")
    print(classification_report(y_val, y_val_pred, zero_division=0))

    return model, {"train_acc": train_acc, "val_acc": val_acc}


def map_sub_to_bucket(predictions: np.ndarray) -> np.ndarray:
    """Convierte predicciones de sub-bucket (7 clases) a bucket (3 clases)."""
    return np.array([SUB_TO_BUCKET[p] for p in predictions])


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena clasificadores de edad.")
    p.add_argument("--manifest", default="data/final/manifest.csv", type=Path)
    p.add_argument("--features", default="data/final/features.csv", type=Path)
    p.add_argument("--output-dir", default="models", type=Path)
    p.add_argument("--config", default="configs/spec.yaml", type=Path)
    p.add_argument("--target", choices=["age_bucket", "age_sub"], default="age_sub",
                   help="Qué predecir. Default: age_sub (7 clases finas), "
                        "luego se colapsa a 3 al evaluar.")
    p.add_argument("--pca-components", type=int, default=200,
                   help="Cuántos componentes principales usar (0 = sin PCA)")
    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Cargar datos ----------------------------------------------------
    logger.info("=== Cargando datos ===")
    df = load_data(args.manifest, args.features)

    # Filtrar splits
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    logger.info(f"  Train: {len(df_train):,}    Val: {len(df_val):,}")

    # Extraer X, y
    X_train, y_train = get_X_y(df_train, target_col=args.target)
    X_val, y_val = get_X_y(df_val, target_col=args.target)
    logger.info(f"  Shape X_train: {X_train.shape}   y_train: {y_train.shape}")

    # ----- Estandarizar features ------------------------------------------
    logger.info("\n=== Estandarizando features ===")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # ----- PCA opcional ----------------------------------------------------
    pca = None
    if args.pca_components > 0:
        logger.info(f"\n=== PCA a {args.pca_components} componentes ===")
        pca = PCA(n_components=args.pca_components, random_state=42)
        X_train_s = pca.fit_transform(X_train_s)
        X_val_s = pca.transform(X_val_s)
        explained = pca.explained_variance_ratio_.sum() * 100
        logger.info(f"  Varianza explicada: {explained:.1f}%")

    # ----- Entrenar todos los modelos -------------------------------------
    logger.info("\n=== Entrenando modelos ===")
    models = build_models()
    results = {}
    for name, model in models.items():
        trained_model, metrics = train_and_evaluate(
            name, model, X_train_s, y_train, X_val_s, y_val,
        )

        # Si entrenamos sobre age_sub, también reportamos accuracy en age_bucket
        if args.target == "age_sub":
            y_val_pred_sub = trained_model.predict(X_val_s)
            y_val_pred_bucket = map_sub_to_bucket(y_val_pred_sub)
            y_val_bucket = map_sub_to_bucket(y_val)
            bucket_acc = accuracy_score(y_val_bucket, y_val_pred_bucket)
            metrics["val_acc_buckets"] = bucket_acc
            logger.info(f"  Val accuracy (3 buckets): {bucket_acc:.4f}")

        results[name] = metrics

        # Guardar modelo
        model_path = args.output_dir / f"{name}.joblib"
        joblib.dump(trained_model, model_path)
        logger.info(f"  Modelo guardado: {model_path}")

    # ----- Guardar preprocesador (scaler + pca) ---------------------------
    preprocessor = {"scaler": scaler, "pca": pca, "target": args.target}
    pre_path = args.output_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, pre_path)
    logger.info(f"\nPreprocesador guardado: {pre_path}")

    # ----- Resumen final --------------------------------------------------
    logger.info("\n=== RESUMEN ===")
    summary = pd.DataFrame(results).T
    print(summary.to_string())

    summary_path = args.output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Resumen guardado: {summary_path}")


if __name__ == "__main__":
    main()
