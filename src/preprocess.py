"""
preprocess.py
=============

PASO 2 del pipeline (cada miembro del equipo lo corre con su dataset).

QUÉ HACE:
    1. Recorre las imágenes crudas del dataset asignado
    2. Detecta la cara con MTCNN
    3. La recorta con un margen del 20%
    4. Convierte a escala de grises
    5. Redimensiona a 128x128
    6. Guarda como JPG con nombre prefijado: {dataset}_{idx:06d}.jpg

POR QUÉ ESTOS PASOS:
    - Detección de cara: elimina ruido de fondo. La edad está en la cara.
    - Grayscale: LBP/HOG no usan color, ahorra 3x espacio y 3x tiempo.
    - 128x128: balance entre detalle y velocidad para features clásicas.
    - Prefijo en el nombre: evita colisiones al juntar datasets en una sola carpeta.

USO:
    python src/preprocess.py \\
        --dataset fairface \\
        --input data/raw/fairface \\
        --output data/images \\
        --mapping-output data/partial_outputs/mapping_fairface.csv

OUTPUT:
    - Imágenes en data/images/{dataset}_000001.jpg, _000002.jpg, ...
    - CSV de mapeo {nombre_nuevo: nombre_original} para trazabilidad.
"""

from __future__ import annotations
import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# Silenciar warnings de TensorFlow (MTCNN usa TF)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# Configuración de logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Detección de caras
# =============================================================================
class FaceProcessor:
    """
    Encapsula MTCNN. Lo cargamos una sola vez (es caro inicializar)
    y lo reusamos para todas las imágenes.
    """

    def __init__(self, image_size: int = 128, margin: float = 0.2):
        from mtcnn import MTCNN
        self.detector = MTCNN()
        self.image_size = image_size
        self.margin = margin
        logger.info("MTCNN inicializado (puede tardar la primera vez)")

    def process(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Procesa una imagen y devuelve el rostro recortado en grayscale 128x128.
        Devuelve None si no se detectó cara o si hubo problemas.
        """
        # MTCNN espera RGB, OpenCV carga en BGR
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            detections = self.detector.detect_faces(img_rgb)
        except Exception as e:
            logger.debug(f"Error en MTCNN: {e}")
            return None

        if not detections:
            return None

        # Tomar la cara con mayor confianza si hay varias
        best = max(detections, key=lambda d: d["confidence"])
        if best["confidence"] < 0.90:
            return None   # Detección poco confiable

        x, y, w, h = best["box"]

        # Aplicar margen alrededor de la cara
        margin_x = int(w * self.margin)
        margin_y = int(h * self.margin)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(img_bgr.shape[1], x + w + margin_x)
        y2 = min(img_bgr.shape[0], y + h + margin_y)

        face = img_bgr[y1:y2, x1:x2]
        if face.size == 0:
            return None

        # Convertir a grayscale (LBP/HOG no usan color)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Redimensionar al tamaño objetivo
        resized = cv2.resize(gray, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_AREA)
        return resized


# =============================================================================
# Pipeline principal
# =============================================================================
def find_images(input_dir: Path) -> list[Path]:
    """Busca todas las imágenes (jpg/jpeg/png) recursivamente."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    images = []
    for ext in extensions:
        images.extend(input_dir.rglob(ext))
    return sorted(images)


def process_dataset(
    dataset_name: str,
    input_dir: Path,
    output_dir: Path,
    mapping_path: Path,
    image_size: int,
    margin: float,
    quality: int,
) -> Tuple[int, int]:
    """
    Procesa todas las imágenes de un dataset.
    Devuelve (n_procesadas_ok, n_falladas).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)

    images = find_images(input_dir)
    if not images:
        logger.error(f"No encontré imágenes en {input_dir}")
        sys.exit(1)
    logger.info(f"Encontradas {len(images):,} imágenes en {input_dir}")

    processor = FaceProcessor(image_size=image_size, margin=margin)

    n_ok = 0
    n_fail = 0

    # Abrir CSV de mapeo (para trazabilidad: qué imagen original es cuál)
    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["new_filename", "original_filename"])

        for img_path in tqdm(images, desc=f"Procesando {dataset_name}"):
            img = cv2.imread(str(img_path))
            if img is None:
                n_fail += 1
                continue

            face = processor.process(img)
            if face is None:
                n_fail += 1
                continue

            # Generar nombre con prefijo de dataset y padding numérico
            new_name = f"{dataset_name}_{n_ok + 1:06d}.jpg"
            out_path = output_dir / new_name

            # Guardar con calidad configurada
            cv2.imwrite(str(out_path), face, [cv2.IMWRITE_JPEG_QUALITY, quality])

            # Registrar mapeo
            original_rel = str(img_path.relative_to(input_dir))
            writer.writerow([new_name, original_rel])

            n_ok += 1

    return n_ok, n_fail


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocesa un dataset: detecta caras, recorta, grayscale, resize.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", required=True,
                   help="Nombre del dataset (fairface, utkface, agedb, appa_real, imdb_wiki)")
    p.add_argument("--input", required=True, type=Path,
                   help="Directorio raw del dataset")
    p.add_argument("--output", default="data/images", type=Path,
                   help="Directorio de salida para imágenes procesadas")
    p.add_argument("--mapping-output", type=Path, default=None,
                   help="CSV de salida con mapeo nombre_nuevo→nombre_original. "
                        "Default: data/partial_outputs/mapping_{dataset}.csv")
    p.add_argument("--config", default="configs/spec.yaml", type=Path,
                   help="Archivo de configuración")
    return p.parse_args()


def main():
    args = parse_args()

    # Cargar configuración
    with open(args.config, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    img_cfg = spec["image"]

    if args.mapping_output is None:
        args.mapping_output = Path("data/partial_outputs") / f"mapping_{args.dataset}.csv"

    logger.info(f"=== Preprocesando dataset: {args.dataset} ===")
    logger.info(f"  Input:           {args.input}")
    logger.info(f"  Output imgs:     {args.output}")
    logger.info(f"  Output mapping:  {args.mapping_output}")
    logger.info(f"  Image size:      {img_cfg['size']}x{img_cfg['size']} grayscale")
    logger.info(f"  Face margin:     {img_cfg['face_margin']*100:.0f}%")

    n_ok, n_fail = process_dataset(
        dataset_name=args.dataset,
        input_dir=args.input,
        output_dir=args.output,
        mapping_path=args.mapping_output,
        image_size=img_cfg["size"],
        margin=img_cfg["face_margin"],
        quality=img_cfg["quality"],
    )

    logger.info(f"=== Terminado ===")
    logger.info(f"  Procesadas OK:   {n_ok:,}")
    logger.info(f"  Falladas:        {n_fail:,} ({n_fail / (n_ok + n_fail) * 100:.1f}%)")
    logger.info(f"  Mapeo guardado:  {args.mapping_output}")


if __name__ == "__main__":
    main()
