"""
merge_team.py — Junta los outputs de todo el equipo en un dataset unificado.

Solo el líder técnico corre esto, después de que todos subieron sus archivos.

Qué hace:
  1. Concatena todos los partial_manifest_*.csv
  2. Concatena todos los partial_features_*.csv
  3. Detecta duplicados con pHash
  4. Asigna splits estratificados (train 70%, val 15%, test 15%)
  5. Guarda manifest.csv y features.csv finales

Uso:
    python src/merge_team.py
"""

import argparse
import logging
from pathlib import Path
import imagehash
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def concat_csvs(directory, prefix):
    files = sorted(directory.glob(f"{prefix}*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay archivos {prefix}*.csv en {directory}")
    logger.info(f"Encontrados {len(files)} archivos {prefix}*.csv")
    for f in files:
        logger.info(f"  - {f.name}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def detect_duplicates(images_dir, filenames, threshold=5):
    logger.info("Calculando hashes para detectar duplicados...")
    hashes = {}
    for fname in tqdm(filenames, desc="pHash"):
        path = images_dir / fname
        if path.exists():
            try:
                hashes[fname] = imagehash.phash(Image.open(path))
            except Exception:
                pass

    items = list(hashes.items())
    duplicates = set()
    for i in tqdm(range(len(items)), desc="Comparando"):
        if items[i][0] in duplicates:
            continue
        for j in range(i + 1, len(items)):
            if items[j][0] in duplicates:
                continue
            if items[i][1] - items[j][1] < threshold:
                duplicates.add(items[j][0])
    return duplicates


def stratified_split(df, train_pct, val_pct, seed=42):
    df = df.copy()
    df["_strat"] = df["age_bucket"] + "_" + df["ethnicity"] + "_" + df["gender"]
    counts = df["_strat"].value_counts()
    rare = counts[counts < 10].index
    if len(rare):
        df.loc[df["_strat"].isin(rare), "_strat"] = "rare"

    train_idx, valtest_idx = train_test_split(
        df.index, train_size=train_pct, stratify=df["_strat"], random_state=seed)
    valtest = df.loc[valtest_idx]
    val_ratio = val_pct / (1 - train_pct)
    val_idx, test_idx = train_test_split(
        valtest.index, train_size=val_ratio, stratify=valtest["_strat"], random_state=seed)

    splits = pd.Series(index=df.index, dtype="object")
    splits.loc[train_idx] = "train"
    splits.loc[val_idx] = "val"
    splits.loc[test_idx] = "test"
    return splits


def main():
    p = argparse.ArgumentParser(description="Une los outputs del equipo.")
    p.add_argument("--partial-dir", default="data/partial_outputs", type=Path)
    p.add_argument("--images", default="data/processed", type=Path)
    p.add_argument("--output-dir", default="data/final", type=Path)
    p.add_argument("--config", default="configs/spec.yaml", type=Path)
    p.add_argument("--skip-dedup", action="store_true", help="Saltar detección de duplicados")
    args = p.parse_args()

    with open(args.config) as f:
        spec = yaml.safe_load(f)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Concatenar
    logger.info("=== Concatenando manifests ===")
    manifest = concat_csvs(args.partial_dir, "partial_manifest_")
    logger.info(f"Total: {len(manifest):,} filas")

    logger.info("\n=== Concatenando features ===")
    features = concat_csvs(args.partial_dir, "partial_features_")

    # Intersección por si hay discrepancias
    common = set(manifest["filename"]) & set(features["filename"])
    manifest = manifest[manifest["filename"].isin(common)].reset_index(drop=True)
    features = features[features["filename"].isin(common)].reset_index(drop=True)
    logger.info(f"Tras intersección: {len(manifest):,} filas")

    # 2. Duplicados
    if not args.skip_dedup:
        logger.info("\n=== Detectando duplicados ===")
        dupes = detect_duplicates(args.images, manifest["filename"].tolist(),
                                  threshold=spec["duplicates"]["threshold"])
        logger.info(f"Duplicados: {len(dupes):,}")
        manifest = manifest[~manifest["filename"].isin(dupes)].reset_index(drop=True)
        features = features[features["filename"].isin(manifest["filename"])].reset_index(drop=True)

    # 3. Splits
    logger.info("\n=== Asignando splits ===")
    s = spec["splits"]
    manifest["split"] = stratified_split(manifest, s["train"], s["val"], s["random_seed"])

    # 4. Guardar
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    features.to_csv(args.output_dir / "features.csv", index=False)
    logger.info(f"\nManifest: {args.output_dir / 'manifest.csv'} ({len(manifest):,} filas)")
    logger.info(f"Features: {args.output_dir / 'features.csv'}")

    # 5. Reporte
    logger.info("\n=== Balance final ===")
    print(pd.crosstab(manifest["split"], manifest["age_bucket"], margins=True).to_string())
    print()
    print(pd.crosstab(manifest["split"], manifest["ethnicity"], margins=True).to_string())
    print()
    logger.info("Por source:")
    print(manifest["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
