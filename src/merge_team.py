"""
merge_team.py
==============

PASO 6 del pipeline (UNA persona del equipo, después de que todos terminaron).

QUÉ HACE:
    1. Concatena todos los partial_manifest_*.csv en un manifest único.
    2. Concatena todos los partial_features_*.csv en un features único.
    3. Detecta duplicados con perceptual hashing (pHash) entre TODAS las
       imágenes (incluyendo cross-dataset, donde más probablemente haya
       repetidos: ej. celebridades en IMDB-Wiki y AgeDB).
    4. Genera split estratificado train/val/test usando age_bucket+ethnicity
       como llaves para garantizar representación demográfica balanceada.
    5. Imprime tabla de balance final por (split, age_bucket, ethnicity).

POR QUÉ EL SPLIT ES DESPUÉS Y NO ANTES:
    Si cada miembro asignara sus propios splits, el balance demográfico
    se rompería (FairFace tiene mucha gente negra, IMDB-Wiki casi nada).
    Splitando sobre el conjunto completo, garantizamos que test contenga
    una mezcla representativa.

POR QUÉ pHash:
    Detecta imágenes visualmente similares incluso si fueron recomprimidas
    o redimensionadas. Crítico porque IMDB-Wiki tiene la misma celebridad
    cientos de veces, y algunas también aparecen en AgeDB.

USO:
    python src/merge_team.py \\
        --partial-dir data/partial_outputs \\
        --images data/images \\
        --output-dir data/final
"""

from __future__ import annotations
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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Concatenación
# =============================================================================
def concat_partials(partial_dir: Path, prefix: str) -> pd.DataFrame:
    """Concatena todos los CSVs que empiezan con `prefix` en partial_dir."""
    files = sorted(partial_dir.glob(f"{prefix}*.csv"))
    if not files:
        raise FileNotFoundError(f"No encontré archivos {prefix}*.csv en {partial_dir}")
    logger.info(f"Encontrados {len(files)} archivos {prefix}*.csv:")
    for f in files:
        logger.info(f"  - {f.name}")

    dfs = [pd.read_csv(f) for f in files]
    out = pd.concat(dfs, ignore_index=True)
    logger.info(f"Concatenado: {len(out):,} filas en total")
    return out


# =============================================================================
# Detección de duplicados con pHash
# =============================================================================
def compute_hashes(images_dir: Path, filenames: list[str]) -> dict[str, str]:
    """
    Calcula pHash de cada imagen. Devuelve dict {filename: hash_string}.
    """
    hashes = {}
    for fname in tqdm(filenames, desc="Calculando pHashes"):
        img_path = images_dir / fname
        if not img_path.exists():
            logger.warning(f"No existe: {img_path}")
            continue
        try:
            img = Image.open(img_path)
            phash = str(imagehash.phash(img))
            hashes[fname] = phash
        except Exception as e:
            logger.warning(f"Error con {fname}: {e}")
    return hashes


def detect_duplicates(hashes: dict[str, str], threshold: int = 5) -> set[str]:
    """
    Detecta imágenes duplicadas: pares con distancia Hamming < threshold.

    Retorna el conjunto de filenames a marcar como duplicados (mantenemos
    la primera ocurrencia, marcamos las demás).
    """
    items = list(hashes.items())
    duplicates = set()

    # Convertir hashes a objetos imagehash para poder comparar
    hash_objs = [(fname, imagehash.hex_to_hash(h)) for fname, h in items]

    for i in tqdm(range(len(hash_objs)), desc="Comparando hashes"):
        if hash_objs[i][0] in duplicates:
            continue   # Ya marcado, saltar
        for j in range(i + 1, len(hash_objs)):
            if hash_objs[j][0] in duplicates:
                continue
            distance = hash_objs[i][1] - hash_objs[j][1]
            if distance < threshold:
                duplicates.add(hash_objs[j][0])   # Marcar el segundo

    return duplicates


# =============================================================================
# Split estratificado
# =============================================================================
def stratified_split(df: pd.DataFrame, train_pct: float, val_pct: float,
                     test_pct: float, seed: int = 42) -> pd.Series:
    """
    Asigna train/val/test estratificando por age_bucket+ethnicity+gender.

    Usamos una columna combinada como key de estratificación para que
    cada split mantenga proporciones similares en TODAS las dimensiones.
    """
    assert abs((train_pct + val_pct + test_pct) - 1.0) < 1e-6, \
        f"Splits deben sumar 1.0, dieron {train_pct + val_pct + test_pct}"

    # Llave combinada de estratificación
    df = df.copy()
    df["_strat"] = (df["age_bucket"].astype(str) + "_" +
                    df["ethnicity"].astype(str) + "_" +
                    df["gender"].astype(str))

    # Algunos estratos pueden tener muy pocas muestras; sklearn falla si
    # hay estrato con <2 muestras. Los marcamos como 'other_strat'.
    counts = df["_strat"].value_counts()
    rare = counts[counts < 10].index
    if len(rare):
        logger.warning(f"{len(rare)} estratos con <10 muestras se agrupan como 'rare'")
        df.loc[df["_strat"].isin(rare), "_strat"] = "rare"

    # Primer split: train vs (val+test)
    train_idx, valtest_idx = train_test_split(
        df.index, train_size=train_pct, stratify=df["_strat"],
        random_state=seed,
    )

    # Segundo split: val vs test
    valtest_df = df.loc[valtest_idx]
    val_size = val_pct / (val_pct + test_pct)
    val_idx, test_idx = train_test_split(
        valtest_df.index, train_size=val_size, stratify=valtest_df["_strat"],
        random_state=seed,
    )

    splits = pd.Series(index=df.index, dtype="object")
    splits.loc[train_idx] = "train"
    splits.loc[val_idx] = "val"
    splits.loc[test_idx] = "test"
    return splits


# =============================================================================
# Reporte
# =============================================================================
def print_balance_report(manifest: pd.DataFrame):
    """Imprime tabla cruzada (split × age_bucket × ethnicity)."""
    logger.info("=" * 70)
    logger.info("BALANCE FINAL")
    logger.info("=" * 70)

    # Total por split
    logger.info("\nTotal por split:")
    print(manifest["split"].value_counts().to_string())

    # Cruzada split × age_bucket
    logger.info("\nSplit × age_bucket:")
    print(pd.crosstab(manifest["split"], manifest["age_bucket"], margins=True).to_string())

    # Cruzada split × ethnicity
    logger.info("\nSplit × ethnicity:")
    print(pd.crosstab(manifest["split"], manifest["ethnicity"], margins=True).to_string())

    # age_bucket × ethnicity (sin separar por split)
    logger.info("\nage_bucket × ethnicity (totales):")
    print(pd.crosstab(manifest["age_bucket"], manifest["ethnicity"], margins=True).to_string())

    # Cuántas filas por dataset llegaron al final
    logger.info("\nFilas por source:")
    print(manifest["source"].value_counts().to_string())


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Une los outputs parciales del equipo en el dataset final.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--partial-dir", default="data/partial_outputs", type=Path,
                   help="Directorio donde están los partial_manifest_*.csv y partial_features_*.csv")
    p.add_argument("--images", default="data/images", type=Path,
                   help="Directorio con TODAS las imágenes ya procesadas")
    p.add_argument("--output-dir", default="data/final", type=Path,
                   help="Directorio de salida")
    p.add_argument("--config", default="configs/spec.yaml", type=Path)
    p.add_argument("--skip-dedup", action="store_true",
                   help="Saltar detección de duplicados (rápido pero peligroso)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Paso 1: concatenar manifests parciales --------------------------
    logger.info("=== PASO 1: Concatenando manifests parciales ===")
    manifest = concat_partials(args.partial_dir, "partial_manifest_")

    # ----- Paso 2: concatenar features parciales ---------------------------
    logger.info("\n=== PASO 2: Concatenando features parciales ===")
    features = concat_partials(args.partial_dir, "partial_features_")

    # Verificar consistencia
    if len(manifest) != len(features):
        logger.warning(f"Manifest tiene {len(manifest):,} filas pero features tiene "
                       f"{len(features):,}. Quedándonos con la intersección.")
        common_files = set(manifest["filename"]) & set(features["filename"])
        manifest = manifest[manifest["filename"].isin(common_files)].reset_index(drop=True)
        features = features[features["filename"].isin(common_files)].reset_index(drop=True)
        logger.info(f"Tras intersección: {len(manifest):,} filas")

    # ----- Paso 3: detectar duplicados -------------------------------------
    if args.skip_dedup:
        logger.warning("⚠ SALTANDO detección de duplicados (--skip-dedup)")
        manifest["is_duplicate"] = False
    else:
        logger.info("\n=== PASO 3: Detectando duplicados con pHash ===")
        hashes = compute_hashes(args.images, manifest["filename"].tolist())
        threshold = spec["duplicates"]["threshold"]
        duplicates = detect_duplicates(hashes, threshold=threshold)
        manifest["is_duplicate"] = manifest["filename"].isin(duplicates)
        logger.info(f"Duplicados detectados: {manifest['is_duplicate'].sum():,} "
                    f"({manifest['is_duplicate'].mean() * 100:.1f}%)")

    # Filtrar duplicados antes del split (para no contaminar test con train)
    n_before = len(manifest)
    keep_mask = ~manifest["is_duplicate"]
    manifest = manifest[keep_mask].drop(columns=["is_duplicate"]).reset_index(drop=True)
    features = features[features["filename"].isin(manifest["filename"])].reset_index(drop=True)
    logger.info(f"Tras eliminar duplicados: {len(manifest):,} (de {n_before:,})")

    # ----- Paso 4: split estratificado -------------------------------------
    logger.info("\n=== PASO 4: Generando splits estratificados ===")
    splits_cfg = spec["splits"]
    manifest["split"] = stratified_split(
        manifest,
        train_pct=splits_cfg["train"],
        val_pct=splits_cfg["val"],
        test_pct=splits_cfg["test"],
        seed=splits_cfg["random_seed"],
    )

    # ----- Paso 5: guardar archivos finales --------------------------------
    logger.info("\n=== PASO 5: Guardando archivos finales ===")
    manifest_path = args.output_dir / "manifest.csv"
    features_path = args.output_dir / "features.csv"
    manifest.to_csv(manifest_path, index=False)
    features.to_csv(features_path, index=False)
    logger.info(f"  Manifest: {manifest_path} ({len(manifest):,} filas)")
    logger.info(f"  Features: {features_path} "
                f"({features_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # ----- Paso 6: reporte de balance --------------------------------------
    print_balance_report(manifest)


if __name__ == "__main__":
    main()
