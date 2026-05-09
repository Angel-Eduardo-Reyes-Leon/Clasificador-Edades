"""
extract_features.py
====================

PASO 4 del pipeline (cada miembro del equipo lo corre con su dataset).

QUÉ HACE:
    Para cada imagen del manifest parcial, extrae un vector de features
    compuesto por la concatenación de:

    1. LBP (Local Binary Patterns)  — ~10 valores con method='uniform'
       Captura textura local de la piel: arrugas, poros, irregularidades.
       Es la feature más predictiva de edad porque la piel cambia con los años.

    2. HOG (Histogram of Oriented Gradients)  — ~324 valores
       Captura bordes y orientaciones. Detecta patrones estructurales como
       arrugas profundas, líneas de expresión, contornos.

    3. Landmark ratios  — ~10 valores
       Distancias y proporciones entre puntos clave de la cara (ojos, nariz,
       boca). Especialmente útil para distinguir niños de adultos: las
       proporciones cambian con el desarrollo craneofacial.

POR QUÉ ESTAS Y NO OTRAS:
    LBP + HOG es la combinación clásica en papers de age estimation con ML
    clásico (ver Bereta et al. 2013, Ylioinas et al. 2013). Los landmarks
    añaden información geométrica que LBP/HOG no capturan.

USO:
    python src/extract_features.py \\
        --images data/images \\
        --manifest data/partial_outputs/partial_manifest_fairface.csv \\
        --output data/partial_outputs/partial_features_fairface.csv

OUTPUT:
    CSV con columnas: filename, lbp_0..lbp_N, hog_0..hog_M, lm_0..lm_K
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import yaml
from skimage.feature import local_binary_pattern, hog
from tqdm import tqdm

# Silenciar warnings ruidosos
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Extracción de LBP
# =============================================================================
def extract_lbp(gray_img: np.ndarray, P: int = 8, R: int = 1,
                method: str = "uniform") -> np.ndarray:
    """
    Calcula el histograma de Local Binary Patterns.

    LBP recorre cada píxel y compara su valor con sus P vecinos a distancia R.
    Genera un código binario por píxel. El histograma de esos códigos es
    una signature de textura local.

    Con method='uniform' obtenemos P+2 = 10 bins (P=8). Es la versión
    rotation-invariant más usada.
    """
    lbp = local_binary_pattern(gray_img, P=P, R=R, method=method)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


# =============================================================================
# Extracción de HOG
# =============================================================================
def extract_hog(gray_img: np.ndarray, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), orientations=9) -> np.ndarray:
    """
    Calcula HOG (Histogram of Oriented Gradients).

    HOG divide la imagen en celdas, calcula el gradiente en cada píxel,
    y genera un histograma de orientaciones por celda. Las celdas se
    agrupan en bloques con normalización para ser robustas a iluminación.

    Con imagen 128x128, cell=16x16, block=2x2, orient=9:
        celdas = 8x8 = 64
        bloques = 7x7 = 49 (con stride=1)
        features = 49 * 4 * 9 = 1764... pero por bloques con stride=1 da 324
    """
    return hog(
        gray_img,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


# =============================================================================
# Extracción de landmarks faciales
# =============================================================================
class LandmarkExtractor:
    """
    Extrae landmarks con MediaPipe Face Mesh (468 puntos) y calcula
    distancias/ratios útiles para diferenciar edades (sobre todo niño vs adulto).

    Los ratios geométricos son ROBUSTOS a la escala (no importa el tamaño
    de la cara), porque son cocientes de distancias.
    """

    # Índices de landmarks importantes en MediaPipe Face Mesh (468 puntos)
    # Referencia: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    LM_LEFT_EYE_OUTER  = 33
    LM_RIGHT_EYE_OUTER = 263
    LM_LEFT_EYE_INNER  = 133
    LM_RIGHT_EYE_INNER = 362
    LM_NOSE_TIP        = 1
    LM_NOSE_BRIDGE     = 168
    LM_MOUTH_LEFT      = 61
    LM_MOUTH_RIGHT     = 291
    LM_CHIN            = 152
    LM_FOREHEAD        = 10

    def __init__(self):
        import mediapipe as mp
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.3,   # Bajo, ya recortamos la cara
        )
        self.n_features = 10   # Cantidad de ratios que devolvemos

    def extract(self, gray_img: np.ndarray) -> np.ndarray:
        """
        Devuelve un vector de ratios. Si no hay landmarks, devuelve ceros.
        """
        # MediaPipe espera RGB
        rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return np.zeros(self.n_features, dtype=np.float32)

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = gray_img.shape

        def pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])

        def dist(a, b):
            return float(np.linalg.norm(a - b))

        # Extraer puntos clave
        left_eye_outer  = pt(self.LM_LEFT_EYE_OUTER)
        right_eye_outer = pt(self.LM_RIGHT_EYE_OUTER)
        left_eye_inner  = pt(self.LM_LEFT_EYE_INNER)
        right_eye_inner = pt(self.LM_RIGHT_EYE_INNER)
        nose_tip        = pt(self.LM_NOSE_TIP)
        nose_bridge     = pt(self.LM_NOSE_BRIDGE)
        mouth_left      = pt(self.LM_MOUTH_LEFT)
        mouth_right     = pt(self.LM_MOUTH_RIGHT)
        chin            = pt(self.LM_CHIN)
        forehead        = pt(self.LM_FOREHEAD)

        # Distancias base
        eye_distance     = dist(left_eye_outer, right_eye_outer)
        face_height      = dist(forehead, chin)
        nose_length      = dist(nose_bridge, nose_tip)
        mouth_width      = dist(mouth_left, mouth_right)
        eye_to_eye_inner = dist(left_eye_inner, right_eye_inner)

        # Evitar división por cero
        eps = 1e-6
        eye_distance = max(eye_distance, eps)
        face_height = max(face_height, eps)

        # Ratios (10 features). Son adimensionales → robustos a escala.
        features = np.array([
            face_height / eye_distance,                   # Cara alargada vs redonda
            nose_length / eye_distance,                   # Largo de nariz relativo
            mouth_width / eye_distance,                   # Boca ancha vs angosta
            eye_to_eye_inner / eye_distance,              # Separación interna de ojos
            dist(forehead, nose_bridge) / face_height,    # Frente relativa
            dist(nose_tip, chin) / face_height,           # Mentón relativo
            dist(left_eye_outer, mouth_left) / face_height,
            dist(right_eye_outer, mouth_right) / face_height,
            dist(nose_tip, mouth_left) / mouth_width,
            dist(nose_tip, mouth_right) / mouth_width,
        ], dtype=np.float32)

        return features


# =============================================================================
# Pipeline de extracción
# =============================================================================
def extract_all_features(image_path: Path, lbp_cfg: dict, hog_cfg: dict,
                         lm_extractor: LandmarkExtractor) -> Optional[np.ndarray]:
    """Extrae todas las features de una imagen y las concatena."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    lbp_feat = extract_lbp(img, P=lbp_cfg["P"], R=lbp_cfg["R"],
                           method=lbp_cfg["method"])
    hog_feat = extract_hog(img,
                           pixels_per_cell=tuple(hog_cfg["pixels_per_cell"]),
                           cells_per_block=tuple(hog_cfg["cells_per_block"]),
                           orientations=hog_cfg["orientations"])
    lm_feat = lm_extractor.extract(img)

    return np.concatenate([lbp_feat, hog_feat, lm_feat])


def make_column_names(n_lbp: int, n_hog: int, n_lm: int) -> list[str]:
    """Genera nombres de columna como ['lbp_0', 'lbp_1', ..., 'hog_0', ..., 'lm_0', ...]"""
    cols = ["filename"]
    cols += [f"lbp_{i}" for i in range(n_lbp)]
    cols += [f"hog_{i}" for i in range(n_hog)]
    cols += [f"lm_{i}"  for i in range(n_lm)]
    return cols


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extrae features (LBP + HOG + landmarks) de imágenes procesadas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--images", required=True, type=Path,
                   help="Directorio con imágenes procesadas")
    p.add_argument("--manifest", required=True, type=Path,
                   help="Manifest parcial (define qué imágenes procesar)")
    p.add_argument("--output", required=True, type=Path,
                   help="CSV de salida con features")
    p.add_argument("--config", default="configs/spec.yaml", type=Path)
    return p.parse_args()


def main():
    args = parse_args()

    # Cargar configuración
    with open(args.config, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    feat_cfg = spec["features"]

    logger.info(f"=== Extrayendo features ===")
    logger.info(f"  Manifest: {args.manifest}")
    logger.info(f"  Imágenes: {args.images}")

    # Leer manifest para saber qué imágenes procesar
    manifest = pd.read_csv(args.manifest)
    logger.info(f"Imágenes a procesar: {len(manifest):,}")

    # Inicializar extractor de landmarks (carga MediaPipe una vez)
    lm_extractor = LandmarkExtractor()

    # Procesar la primera imagen para descubrir cuántas features tiene cada tipo
    first_img_path = args.images / manifest.iloc[0]["filename"]
    sample_features = extract_all_features(first_img_path, feat_cfg["lbp"],
                                            feat_cfg["hog"], lm_extractor)
    if sample_features is None:
        raise SystemExit(f"No pude leer la primera imagen: {first_img_path}")

    # Calcular cuántas features tiene cada tipo (para nombres de columna)
    sample_lbp = extract_lbp(cv2.imread(str(first_img_path), cv2.IMREAD_GRAYSCALE),
                              **feat_cfg["lbp"])
    sample_hog = extract_hog(cv2.imread(str(first_img_path), cv2.IMREAD_GRAYSCALE),
                              pixels_per_cell=tuple(feat_cfg["hog"]["pixels_per_cell"]),
                              cells_per_block=tuple(feat_cfg["hog"]["cells_per_block"]),
                              orientations=feat_cfg["hog"]["orientations"])
    n_lbp = len(sample_lbp)
    n_hog = len(sample_hog)
    n_lm = lm_extractor.n_features
    total_features = n_lbp + n_hog + n_lm

    logger.info(f"  Features por imagen: {total_features} "
                f"(LBP={n_lbp}, HOG={n_hog}, landmarks={n_lm})")

    column_names = make_column_names(n_lbp, n_hog, n_lm)

    # Procesar todas las imágenes
    rows = []
    n_failed = 0
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Features"):
        img_path = args.images / row["filename"]
        feats = extract_all_features(img_path, feat_cfg["lbp"],
                                     feat_cfg["hog"], lm_extractor)
        if feats is None:
            n_failed += 1
            continue
        rows.append([row["filename"]] + feats.tolist())

    if n_failed:
        logger.warning(f"Falló extracción en {n_failed} imágenes")

    # Guardar
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows, columns=column_names)
    df_out.to_csv(args.output, index=False)
    logger.info(f"Features guardadas: {args.output}")
    logger.info(f"Tamaño del archivo: ~{args.output.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
