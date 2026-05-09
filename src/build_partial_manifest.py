"""
build_partial_manifest.py
==========================

PASO 3 del pipeline (cada miembro del equipo lo corre con su dataset).

QUÉ HACE:
    1. Usa el adaptador del dataset para leer las etiquetas originales
    2. Une las etiquetas con el mapeo generado en preprocess.py
       (porque las imágenes ya tienen nombres nuevos)
    3. Asigna age_bucket (3 clases) y age_sub (7 clases internas)
    4. Genera el partial_manifest_{dataset}.csv

POR QUÉ:
    - Cada dataset tiene formato distinto, pero al final del día queremos
      un único formato común para que el merge concatene sin problemas.
    - El age_sub fino ayuda a entrenar mejor la clase 'young' (que abarca
      desde bebés hasta jóvenes adultos, muy heterogénea).

USO:
    python src/build_partial_manifest.py \\
        --dataset fairface \\
        --raw data/raw/fairface \\
        --mapping data/partial_outputs/mapping_fairface.csv \\
        --output data/partial_outputs/partial_manifest_fairface.csv

OUTPUT:
    CSV con columnas: filename, age, age_bucket, age_sub, gender, ethnicity, source
    NOTA: NO incluye 'split' (eso se asigna en el merge final).
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import pandas as pd
import yaml

from dataset_adapters import get_adapter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Asignación de buckets de edad
# =============================================================================
def assign_bucket(age: int, buckets: dict) -> str:
    """
    Asigna una edad a su bucket. `buckets` es un dict {nombre: [min, max]}
    Devuelve el nombre del bucket o 'unknown' si no encaja.
    """
    for bucket_name, (low, high) in buckets.items():
        if low <= age <= high:
            return bucket_name
    return "unknown"


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Construye el manifest parcial de un dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--raw", required=True, type=Path,
                   help="Directorio raw original del dataset")
    p.add_argument("--mapping", required=True, type=Path,
                   help="CSV de mapeo generado por preprocess.py")
    p.add_argument("--output", required=True, type=Path,
                   help="CSV de salida (partial_manifest_{dataset}.csv)")
    p.add_argument("--config", default="configs/spec.yaml", type=Path)
    return p.parse_args()


def main():
    args = parse_args()

    # Cargar configuración
    with open(args.config, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    age_buckets = spec["age_buckets"]
    age_sub_buckets = spec["age_sub_buckets"]

    logger.info(f"=== Construyendo manifest parcial: {args.dataset} ===")

    # Paso 1: leer etiquetas originales con el adaptador correspondiente
    adapter = get_adapter(args.dataset, args.raw)
    labels = adapter.read_labels()
    logger.info(f"Etiquetas originales leídas: {len(labels):,} filas")

    # Paso 2: leer mapeo nuevo↔original generado por preprocess.py
    mapping = pd.read_csv(args.mapping)
    logger.info(f"Mapeo de imágenes: {len(mapping):,} filas")

    # Paso 3: unir etiquetas con mapeo (las imágenes ya tienen nombres nuevos)
    # El join se hace por 'original_filename' que aparece en ambos.
    merged = mapping.merge(labels, on="original_filename", how="inner")
    logger.info(f"Tras el merge: {len(merged):,} filas con etiquetas válidas")

    if len(merged) == 0:
        logger.error("El merge dio cero filas. Probable causa: los nombres en "
                     "el mapeo y en las etiquetas no coinciden.")
        logger.error("Ejemplo de mapping: %s", mapping["original_filename"].iloc[:3].tolist())
        logger.error("Ejemplo de labels:  %s", labels["original_filename"].iloc[:3].tolist())
        raise SystemExit(1)

    # Paso 4: asignar buckets de edad
    merged["age_bucket"] = merged["age"].apply(lambda a: assign_bucket(a, age_buckets))
    merged["age_sub"] = merged["age"].apply(lambda a: assign_bucket(a, age_sub_buckets))

    # Filtrar filas con bucket 'unknown' (edad fuera de rango definido)
    n_before = len(merged)
    merged = merged[merged["age_bucket"] != "unknown"].reset_index(drop=True)
    n_after = len(merged)
    if n_before != n_after:
        logger.warning(f"Descartadas {n_before - n_after} filas con edad fuera de buckets")

    # Paso 5: agregar columna 'source' (de qué dataset viene)
    merged["source"] = args.dataset

    # Paso 6: ordenar columnas en orden estándar
    output = merged[[
        "new_filename", "age", "age_bucket", "age_sub",
        "gender", "ethnicity", "source"
    ]].rename(columns={"new_filename": "filename"})

    # Paso 7: guardar
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    logger.info(f"Manifest parcial guardado: {args.output}")

    # Reporte de estadísticas
    logger.info("=== Distribución por age_bucket ===")
    for bucket, count in output["age_bucket"].value_counts().items():
        pct = count / len(output) * 100
        logger.info(f"  {bucket:10s}: {count:6,} ({pct:5.1f}%)")

    logger.info("=== Distribución por etnia ===")
    for eth, count in output["ethnicity"].value_counts().items():
        pct = count / len(output) * 100
        logger.info(f"  {eth:10s}: {count:6,} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
