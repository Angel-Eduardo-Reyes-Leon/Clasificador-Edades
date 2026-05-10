"""
validate_input.py — Verifica que tu carpeta estandarizada esté bien.

Corre esto ANTES de preprocess.py. Si algo falla, te dice qué corregir.

Uso:
    python src/validate_input.py --input data/raw/mi_dataset
"""

import argparse
import sys
from pathlib import Path
import csv

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

VALID_GENDERS = {"male", "female", "unknown", ""}
VALID_ETHNICITIES = {"white", "black", "asian", "latino", "other", "unknown", ""}


def check(ok, ok_msg, fail_msg, errors):
    if ok:
        print(f"  {GREEN}✓{RESET} {ok_msg}")
    else:
        print(f"  {RED}✗{RESET} {fail_msg}")
        errors.append(fail_msg)


def main():
    p = argparse.ArgumentParser(description="Verifica tu carpeta estandarizada.")
    p.add_argument("--input", required=True, type=Path,
                   help="Ruta a tu carpeta (debe contener images/ y metadata.csv)")
    args = p.parse_args()

    errors = []
    print(f"\n=== Validando: {args.input} ===\n")

    # --- 1. Estructura de carpetas ---
    print("1. ESTRUCTURA")
    images_dir = args.input / "images"
    metadata_path = args.input / "metadata.csv"

    check(args.input.is_dir(),
          f"Carpeta existe: {args.input}",
          f"No existe la carpeta: {args.input}", errors)

    check(images_dir.is_dir(),
          f"Subcarpeta images/ existe",
          f"No existe {images_dir}. Crea la carpeta 'images/' dentro de {args.input} y pon tus fotos ahí.", errors)

    check(metadata_path.is_file(),
          f"metadata.csv existe",
          f"No existe {metadata_path}. Revisa docs/DATASET_GUIDE.md para saber cómo crearlo.", errors)

    if errors:
        print(f"\n{RED}Corrige los errores de arriba antes de continuar.{RESET}")
        sys.exit(1)

    # --- 2. Contenido de images/ ---
    print("\n2. IMÁGENES")
    img_extensions = {".jpg", ".jpeg", ".png"}
    all_images = [f for f in images_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in img_extensions]

    check(len(all_images) > 0,
          f"Hay {len(all_images):,} imágenes en images/",
          "No hay imágenes en images/. Copia tus fotos ahí.", errors)

    image_names = {f.name for f in all_images}

    # --- 3. Contenido de metadata.csv ---
    print("\n3. METADATA.CSV")

    with open(metadata_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    # Columnas requeridas
    check("filename" in headers,
          "Tiene columna 'filename'",
          "Falta columna 'filename'. La primera columna del CSV debe llamarse 'filename'.", errors)

    check("age" in headers,
          "Tiene columna 'age'",
          "Falta columna 'age'. Necesitas una columna con la edad numérica.", errors)

    if "filename" not in headers or "age" not in headers:
        print(f"\n{RED}Corrige las columnas del CSV antes de continuar.{RESET}")
        sys.exit(1)

    check(len(rows) > 0,
          f"CSV tiene {len(rows):,} filas",
          "El CSV está vacío (solo tiene headers, ninguna fila de datos).", errors)

    # Columnas opcionales
    has_gender = "gender" in headers
    has_ethnicity = "ethnicity" in headers
    if not has_gender:
        print(f"  {YELLOW}⚠{RESET} No tiene columna 'gender' (se llenará con 'unknown')")
    if not has_ethnicity:
        print(f"  {YELLOW}⚠{RESET} No tiene columna 'ethnicity' (se llenará con 'unknown')")

    # --- 4. Validar contenido fila por fila ---
    print("\n4. CONTENIDO")
    missing_files = []
    bad_ages = []
    bad_genders = []
    bad_ethnicities = []

    for i, row in enumerate(rows):
        fname = row.get("filename", "").strip()
        age_str = row.get("age", "").strip()

        # ¿El archivo existe en images/?
        if fname not in image_names:
            missing_files.append((i + 2, fname))  # +2 por header + 0-index

        # ¿La edad es un número razonable?
        try:
            age = int(float(age_str))
            if age < 0 or age > 120:
                bad_ages.append((i + 2, age_str))
        except (ValueError, TypeError):
            bad_ages.append((i + 2, age_str))

        # ¿El género es válido?
        if has_gender:
            g = row.get("gender", "").strip().lower()
            if g and g not in VALID_GENDERS:
                bad_genders.append((i + 2, g))

        # ¿La etnia es válida?
        if has_ethnicity:
            e = row.get("ethnicity", "").strip().lower()
            if e and e not in VALID_ETHNICITIES:
                bad_ethnicities.append((i + 2, e))

    # Reportar
    check(len(missing_files) == 0,
          "Todas las imágenes del CSV existen en images/",
          f"{len(missing_files)} imágenes del CSV no se encontraron en images/. "
          f"Primeras 5: {missing_files[:5]}", errors)

    check(len(bad_ages) == 0,
          "Todas las edades son números válidos (0-120)",
          f"{len(bad_ages)} filas con edad inválida. "
          f"Primeras 5: {bad_ages[:5]}. La edad debe ser un número entero.", errors)

    if bad_genders:
        print(f"  {YELLOW}⚠{RESET} {len(bad_genders)} filas con género no reconocido: "
              f"{bad_genders[:5]}. Valores válidos: male, female, unknown")

    if bad_ethnicities:
        print(f"  {YELLOW}⚠{RESET} {len(bad_ethnicities)} filas con etnia no reconocida: "
              f"{bad_ethnicities[:5]}. Valores válidos: white, black, asian, latino, other, unknown")

    # Imágenes huérfanas (están en images/ pero no en el CSV)
    csv_filenames = {row["filename"].strip() for row in rows}
    orphans = image_names - csv_filenames
    if orphans:
        print(f"  {YELLOW}⚠{RESET} {len(orphans)} imágenes en images/ que no están en el CSV. "
              f"No se van a procesar, pero no es un error grave.")

    # --- Resumen ---
    print("\n=== RESUMEN ===")
    if errors:
        print(f"{RED}✗ {len(errors)} errores. Corrígelos antes de correr preprocess.py:{RESET}")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"{GREEN}✓ Todo en orden. Puedes correr preprocess.py{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
