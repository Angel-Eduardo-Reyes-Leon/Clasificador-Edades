"""
validate_output.py — Verifica que tus outputs estén listos para subir al Drive.

Corre esto DESPUÉS de extract_features.py y ANTES de subir al Drive.

Uso:
    python src/validate_output.py \\
        --name fairface \\
        --images data/processed \\
        --manifest data/partial_outputs/partial_manifest_fairface.csv \\
        --features data/partial_outputs/partial_features_fairface.csv
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

VALID_BUCKETS = {"young", "adult", "old"}
VALID_SUBS = {"baby", "child", "teen", "young_adult", "middle_adult", "mature_adult", "old"}
VALID_ETH = {"white", "black", "asian", "latino", "other", "unknown"}
VALID_GEN = {"male", "female", "unknown"}


def check(ok, ok_msg, fail_msg, errors):
    if ok:
        print(f"  {GREEN}✓{RESET} {ok_msg}")
    else:
        print(f"  {RED}✗{RESET} {fail_msg}")
        errors.append(fail_msg)


def main():
    p = argparse.ArgumentParser(description="Valida outputs antes de subir al Drive.")
    p.add_argument("--name", required=True)
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--features", required=True, type=Path)
    args = p.parse_args()

    errors = []
    print(f"\n=== Validando outputs de '{args.name}' ===\n")

    # 1. Manifest
    print("1. MANIFEST")
    manifest = pd.read_csv(args.manifest)
    expected = {"filename", "age", "age_bucket", "age_sub", "gender", "ethnicity", "source"}
    missing = expected - set(manifest.columns)
    check(not missing, "Columnas correctas", f"Faltan columnas: {missing}", errors)
    check(len(manifest) > 0, f"{len(manifest):,} filas", "Está vacío", errors)

    if not missing and len(manifest) > 0:
        check(manifest["age_bucket"].isin(VALID_BUCKETS).all(),
              "age_bucket válidos", f"Valores inválidos en age_bucket", errors)
        check(manifest["ethnicity"].isin(VALID_ETH).all(),
              "ethnicity válidos", f"Valores inválidos en ethnicity", errors)
        check(manifest["gender"].isin(VALID_GEN).all(),
              "gender válidos", f"Valores inválidos en gender", errors)
        check(manifest["filename"].str.startswith(args.name).all(),
              f"Filenames con prefijo '{args.name}_'", "Hay filenames sin prefijo correcto", errors)

    # 2. Features
    print("\n2. FEATURES")
    features = pd.read_csv(args.features)
    n_feat = len(features.columns) - 1
    check(n_feat > 50, f"{n_feat} columnas de features", f"Solo {n_feat} features (muy pocas)", errors)
    check(len(features) == len(manifest),
          f"Mismas filas que manifest ({len(features):,})",
          f"Discrepancia: features={len(features):,} vs manifest={len(manifest):,}", errors)

    num_cols = features.select_dtypes(include=[np.number]).columns
    n_nan = features[num_cols].isna().sum().sum()
    if n_nan > 0:
        print(f"  {YELLOW}⚠{RESET} {n_nan} valores NaN en features")

    # 3. Imágenes
    print("\n3. IMÁGENES")
    imgs = sorted(args.images.glob(f"{args.name}_*.jpg"))
    check(len(imgs) > 0, f"{len(imgs):,} imágenes procesadas", "No hay imágenes", errors)
    check(len(imgs) == len(manifest),
          "Misma cantidad que manifest",
          f"Imágenes: {len(imgs):,} vs manifest: {len(manifest):,}", errors)

    # Verificar muestra
    sample = imgs[::max(1, len(imgs) // 3)][:3]
    for img_path in sample:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None or img.shape != (128, 128) or img.ndim != 2:
            check(False, "", f"{img_path.name} no es 128x128 grayscale", errors)
            break
    else:
        check(True, "Muestra: 128x128 grayscale", "", errors)

    # Resumen
    print("\n=== RESUMEN ===")
    if errors:
        print(f"{RED}✗ {len(errors)} errores. Corrige antes de subir.{RESET}")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"{GREEN}✓ Todo en orden. Puedes subir al Drive.{RESET}")
        print(f"\nSube estos 3 archivos:")
        print(f"  1. Imágenes (ZIP de data/processed/{args.name}_*.jpg)")
        print(f"  2. {args.manifest}")
        print(f"  3. {args.features}")
        sys.exit(0)


if __name__ == "__main__":
    main()
