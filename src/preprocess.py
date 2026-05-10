"""
preprocess.py — Detecta caras, recorta, convierte a grayscale, redimensiona.

Lee tu carpeta estandarizada (images/ + metadata.csv) y produce:
  - Imágenes procesadas en data/processed/ con prefijo de tu dataset
  - Un partial_manifest en data/partial_outputs/

Uso:
    python src/preprocess.py --input data/raw/mi_dataset --name fairface --output data/processed
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# --- Detección de caras con MTCNN ---
class FaceProcessor:
    def __init__(self, image_size=128, margin=0.2):
        from mtcnn import MTCNN
        self.detector = MTCNN()
        self.image_size = image_size
        self.margin = margin
        logger.info("MTCNN listo")

    def process(self, img_bgr):
        """Detecta cara, recorta con margen, grayscale, resize. Devuelve None si falla."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            detections = self.detector.detect_faces(img_rgb)
        except Exception:
            return None

        if not detections:
            return None

        best = max(detections, key=lambda d: d["confidence"])
        if best["confidence"] < 0.90:
            return None

        x, y, w, h = best["box"]
        mx, my = int(w * self.margin), int(h * self.margin)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(img_bgr.shape[1], x + w + mx)
        y2 = min(img_bgr.shape[0], y + h + my)

        face = img_bgr[y1:y2, x1:x2]
        if face.size == 0:
            return None

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)


# --- Asignación de buckets ---
def assign_bucket(age, buckets):
    for name, (low, high) in buckets.items():
        if low <= age <= high:
            return name
    return "unknown"


def main():
    p = argparse.ArgumentParser(description="Preprocesa imágenes: detecta cara, recorta, grayscale, resize.")
    p.add_argument("--input", required=True, type=Path, help="Carpeta con images/ y metadata.csv")
    p.add_argument("--name", required=True, help="Nombre del dataset (se usa como prefijo)")
    p.add_argument("--output", default="data/processed", type=Path, help="Carpeta de salida para imágenes procesadas")
    p.add_argument("--config", default="configs/spec.yaml", type=Path)
    args = p.parse_args()

    # Cargar configuración
    with open(args.config) as f:
        spec = yaml.safe_load(f)

    img_cfg = spec["image"]
    age_buckets = spec["age_buckets"]
    age_sub_buckets = spec["age_sub_buckets"]

    # Validar estructura de entrada
    images_dir = args.input / "images"
    metadata_path = args.input / "metadata.csv"
    if not images_dir.is_dir() or not metadata_path.is_file():
        logger.error("La carpeta debe contener images/ y metadata.csv. Corre validate_input.py primero.")
        sys.exit(1)

    # Leer metadata
    with open(metadata_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    logger.info(f"Imágenes a procesar: {len(rows):,}")

    # Preparar output
    args.output.mkdir(parents=True, exist_ok=True)
    partial_dir = Path("data/partial_outputs")
    partial_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = partial_dir / f"partial_manifest_{args.name}.csv"

    # Inicializar detector
    processor = FaceProcessor(image_size=img_cfg["size"], margin=img_cfg["face_margin"])

    # Procesar imagen por imagen
    from tqdm import tqdm
    n_ok = 0
    n_fail = 0
    manifest_rows = []

    for row in tqdm(rows, desc=f"Procesando {args.name}"):
        fname = row["filename"].strip()
        img_path = images_dir / fname

        if not img_path.exists():
            n_fail += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            n_fail += 1
            continue

        face = processor.process(img)
        if face is None:
            n_fail += 1
            continue

        # Guardar con nombre prefijado
        new_name = f"{args.name}_{n_ok + 1:06d}.jpg"
        cv2.imwrite(str(args.output / new_name), face, [cv2.IMWRITE_JPEG_QUALITY, img_cfg["quality"]])

        # Leer metadatos de la fila
        try:
            age = int(float(row["age"]))
        except (ValueError, TypeError):
            n_fail += 1
            continue

        gender = row.get("gender", "unknown").strip().lower() or "unknown"
        ethnicity = row.get("ethnicity", "unknown").strip().lower() or "unknown"

        manifest_rows.append({
            "filename": new_name,
            "age": age,
            "age_bucket": assign_bucket(age, age_buckets),
            "age_sub": assign_bucket(age, age_sub_buckets),
            "gender": gender,
            "ethnicity": ethnicity,
            "source": args.name,
        })
        n_ok += 1

    # Guardar manifest parcial
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "age", "age_bucket", "age_sub",
                                                "gender", "ethnicity", "source"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info(f"Terminado. OK: {n_ok:,} | Fallidas: {n_fail:,} ({n_fail/(n_ok+n_fail)*100:.1f}%)")
    logger.info(f"Manifest guardado: {manifest_path}")


if __name__ == "__main__":
    main()
