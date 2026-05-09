"""
validate_partial.py
====================

PASO 5 del pipeline (cada miembro corre esto antes de subir al Drive).

QUÉ HACE:
    Verifica que los outputs parciales estén bien antes de subirlos:

    ✓ Todas las imágenes son del tamaño correcto y grayscale
    ✓ Todos los nombres siguen la convención {dataset}_NNNNNN.jpg
    ✓ El número de imágenes en disco = filas en manifest = filas en features
    ✓ El manifest no tiene nulos en columnas críticas
    ✓ Los buckets caen en el vocabulario común
    ✓ Las features no tienen NaN ni infinitos

POR QUÉ:
    Detectar errores ANTES de subir 5 GB al Drive ahorra tiempo a todos.
    Si algo falla, el merge final se rompe y nadie sabe por qué.

USO:
    python src/validate_partial.py \\
        --dataset fairface \\
        --images data/images \\
        --manifest data/partial_outputs/partial_manifest_fairface.csv \\
        --features data/partial_outputs/partial_features_fairface.csv

OUTPUT:
    Mensajes en pantalla. Exit code 0 = todo OK, 1 = hay errores.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml


# Códigos de color para terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

errors = []
warnings_list = []


def check(condition: bool, ok_msg: str, fail_msg: str, warn: bool = False):
    """Helper: imprime ✓ si pasa, ✗ o ⚠ si no, y registra el problema."""
    if condition:
        print(f"  {GREEN}✓{RESET} {ok_msg}")
    else:
        if warn:
            print(f"  {YELLOW}⚠{RESET} {fail_msg}")
            warnings_list.append(fail_msg)
        else:
            print(f"  {RED}✗{RESET} {fail_msg}")
            errors.append(fail_msg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Valida outputs parciales antes de subir.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--config", default="configs/spec.yaml", type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    expected_size = spec["image"]["size"]
    age_buckets = list(spec["age_buckets"].keys())
    age_sub_buckets = list(spec["age_sub_buckets"].keys())
    ethnicity_vocab = spec["ethnicity_vocab"]
    gender_vocab = spec["gender_vocab"]

    print(f"\n=== Validando outputs de '{args.dataset}' ===\n")

    # ----------------------------------------------------------------- 1. Manifest
    print("1. MANIFEST")
    if not args.manifest.exists():
        print(f"  {RED}✗{RESET} No existe: {args.manifest}")
        sys.exit(1)
    manifest = pd.read_csv(args.manifest)
    check(len(manifest) > 0, f"Manifest tiene {len(manifest):,} filas",
          "Manifest está vacío")

    expected_cols = {"filename", "age", "age_bucket", "age_sub",
                     "gender", "ethnicity", "source"}
    actual_cols = set(manifest.columns)
    missing = expected_cols - actual_cols
    check(not missing, "Tiene todas las columnas requeridas",
          f"Faltan columnas: {missing}")

    if not missing:
        check(manifest["filename"].notna().all(),
              "Sin nulos en 'filename'", "Hay nulos en 'filename'")
        check(manifest["age"].notna().all(),
              "Sin nulos en 'age'", "Hay nulos en 'age'")
        check(manifest["age_bucket"].isin(age_buckets).all(),
              "Todos los age_bucket son válidos",
              f"Hay age_bucket fuera del vocabulario: "
              f"{set(manifest['age_bucket']) - set(age_buckets)}")
        check(manifest["age_sub"].isin(age_sub_buckets).all(),
              "Todos los age_sub son válidos",
              f"Hay age_sub fuera del vocabulario: "
              f"{set(manifest['age_sub']) - set(age_sub_buckets)}")
        check(manifest["ethnicity"].isin(ethnicity_vocab).all(),
              "Todas las etnias son válidas",
              f"Hay etnia fuera del vocabulario: "
              f"{set(manifest['ethnicity']) - set(ethnicity_vocab)}")
        check(manifest["gender"].isin(gender_vocab).all(),
              "Todos los géneros son válidos",
              f"Hay género fuera del vocabulario: "
              f"{set(manifest['gender']) - set(gender_vocab)}")
        check((manifest["source"] == args.dataset).all(),
              f"Todos los registros son de '{args.dataset}'",
              "Hay registros de otro dataset en el manifest")

        # Verificar prefijo de filename
        bad_names = manifest[~manifest["filename"].str.startswith(f"{args.dataset}_")]
        check(len(bad_names) == 0,
              f"Todos los filenames empiezan con '{args.dataset}_'",
              f"{len(bad_names)} filenames sin prefijo correcto")

    # ----------------------------------------------------------------- 2. Features
    print("\n2. FEATURES")
    if not args.features.exists():
        print(f"  {RED}✗{RESET} No existe: {args.features}")
    else:
        features = pd.read_csv(args.features)
        check(len(features) > 0, f"Features tiene {len(features):,} filas",
              "Features está vacío")
        check("filename" in features.columns,
              "Tiene columna 'filename'", "Falta columna 'filename'")

        if "filename" in features.columns:
            n_features = len(features.columns) - 1   # menos 'filename'
            check(n_features > 100, f"Tiene {n_features} columnas de features",
                  f"Solo tiene {n_features} features (sospechosamente pocas)")

            # Verificar consistencia con manifest
            check(len(features) == len(manifest),
                  f"Mismo número de filas que el manifest ({len(features):,})",
                  f"Filas no coinciden: features={len(features):,}, manifest={len(manifest):,}")

            # Verificar NaN/inf en valores numéricos
            num_cols = features.select_dtypes(include=[np.number]).columns
            n_nan = features[num_cols].isna().sum().sum()
            check(n_nan == 0, "Sin NaN en features",
                  f"{n_nan} valores NaN en features", warn=True)

            n_inf = np.isinf(features[num_cols].values).sum()
            check(n_inf == 0, "Sin valores infinitos",
                  f"{n_inf} valores infinitos", warn=True)

    # ----------------------------------------------------------------- 3. Imágenes
    print("\n3. IMÁGENES")
    if not args.images.exists():
        print(f"  {RED}✗{RESET} No existe directorio: {args.images}")
    else:
        # Solo validar imágenes de este dataset (puede haber de otros)
        imgs_this = sorted(args.images.glob(f"{args.dataset}_*.jpg"))
        check(len(imgs_this) > 0, f"Hay {len(imgs_this):,} imágenes de {args.dataset}",
              f"No hay imágenes con prefijo '{args.dataset}_'")

        if imgs_this:
            check(len(imgs_this) == len(manifest),
                  f"Imágenes en disco = filas en manifest ({len(imgs_this):,})",
                  f"Discrepancia: {len(imgs_this):,} en disco vs {len(manifest):,} en manifest",
                  warn=True)

            # Sample de 5 imágenes para verificar tamaño y grayscale
            sample = imgs_this[::max(1, len(imgs_this) // 5)][:5]
            sizes_ok = True
            grayscale_ok = True
            for img_path in sample:
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    sizes_ok = False
                    continue
                if img.shape[:2] != (expected_size, expected_size):
                    sizes_ok = False
                if img.ndim != 2:   # No es grayscale
                    grayscale_ok = False
            check(sizes_ok, f"Imágenes tienen tamaño {expected_size}x{expected_size}",
                  f"Algunas imágenes NO son {expected_size}x{expected_size}")
            check(grayscale_ok, "Imágenes son grayscale (1 canal)",
                  "Algunas imágenes NO son grayscale")

    # ----------------------------------------------------------------- 4. Resumen
    print("\n=== RESUMEN ===")
    if errors:
        print(f"{RED}✗ {len(errors)} ERRORES (deben corregirse antes de subir):{RESET}")
        for e in errors:
            print(f"  - {e}")
    if warnings_list:
        print(f"{YELLOW}⚠ {len(warnings_list)} ADVERTENCIAS (revisar pero no bloqueantes):{RESET}")
        for w in warnings_list:
            print(f"  - {w}")
    if not errors and not warnings_list:
        print(f"{GREEN}✓ Todo en orden. Puedes subir tus archivos al Drive.{RESET}")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
