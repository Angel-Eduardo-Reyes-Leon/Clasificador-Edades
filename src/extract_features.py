"""
extract_features.py — Extrae LBP + HOG + landmarks de cada imagen procesada.

Genera un CSV con un vector numérico por imagen (~340 columnas).
Ese CSV es lo que los modelos (SVM, XGBoost) van a usar para aprender.

Qué extrae y por qué:
  - LBP (Local Binary Patterns): textura de piel. Arrugas = cambio de textura.
  - HOG (Histogram of Oriented Gradients): bordes. Líneas de expresión.
  - Landmarks: proporciones faciales. Distingue niños de adultos (proporciones distintas).

Uso:
    python src/extract_features.py \\
        --images data/processed \\
        --manifest data/partial_outputs/partial_manifest_fairface.csv \\
        --output data/partial_outputs/partial_features_fairface.csv
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from skimage.feature import local_binary_pattern, hog
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# --- LBP: textura local ---
def extract_lbp(gray, P=8, R=1, method="uniform"):
    lbp = local_binary_pattern(gray, P=P, R=R, method=method)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


# --- HOG: bordes y orientaciones ---
def extract_hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=9):
    return hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block, block_norm="L2-Hys",
               feature_vector=True).astype(np.float32)


# --- Landmarks: proporciones faciales ---
class LandmarkExtractor:
    # Índices importantes de MediaPipe Face Mesh (468 puntos)
    POINTS = {
        "left_eye_outer": 33, "right_eye_outer": 263,
        "left_eye_inner": 133, "right_eye_inner": 362,
        "nose_tip": 1, "nose_bridge": 168,
        "mouth_left": 61, "mouth_right": 291,
        "chin": 152, "forehead": 10,
    }

    def __init__(self):
        import mediapipe as mp
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=False, min_detection_confidence=0.3)
        self.n_features = 10

    def extract(self, gray):
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return np.zeros(self.n_features, dtype=np.float32)

        lm = results.multi_face_landmarks[0].landmark
        h, w = gray.shape

        def pt(name):
            l = lm[self.POINTS[name]]
            return np.array([l.x * w, l.y * h])

        def dist(a, b):
            return float(np.linalg.norm(a - b))

        eye_dist = max(dist(pt("left_eye_outer"), pt("right_eye_outer")), 1e-6)
        face_h = max(dist(pt("forehead"), pt("chin")), 1e-6)
        nose_len = dist(pt("nose_bridge"), pt("nose_tip"))
        mouth_w = dist(pt("mouth_left"), pt("mouth_right"))
        eye_inner = dist(pt("left_eye_inner"), pt("right_eye_inner"))

        return np.array([
            face_h / eye_dist,
            nose_len / eye_dist,
            mouth_w / eye_dist,
            eye_inner / eye_dist,
            dist(pt("forehead"), pt("nose_bridge")) / face_h,
            dist(pt("nose_tip"), pt("chin")) / face_h,
            dist(pt("left_eye_outer"), pt("mouth_left")) / face_h,
            dist(pt("right_eye_outer"), pt("mouth_right")) / face_h,
            dist(pt("nose_tip"), pt("mouth_left")) / max(mouth_w, 1e-6),
            dist(pt("nose_tip"), pt("mouth_right")) / max(mouth_w, 1e-6),
        ], dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description="Extrae features de imágenes procesadas.")
    p.add_argument("--images", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--config", default="configs/spec.yaml", type=Path)
    args = p.parse_args()

    with open(args.config) as f:
        spec = yaml.safe_load(f)
    feat_cfg = spec["features"]

    manifest = pd.read_csv(args.manifest)
    logger.info(f"Imágenes a procesar: {len(manifest):,}")

    lm_ext = LandmarkExtractor()

    # Descubrir tamaño de features con la primera imagen
    first_img = cv2.imread(str(args.images / manifest.iloc[0]["filename"]), cv2.IMREAD_GRAYSCALE)
    sample_lbp = extract_lbp(first_img, **feat_cfg["lbp"])
    sample_hog = extract_hog(first_img,
                              pixels_per_cell=tuple(feat_cfg["hog"]["pixels_per_cell"]),
                              cells_per_block=tuple(feat_cfg["hog"]["cells_per_block"]),
                              orientations=feat_cfg["hog"]["orientations"])
    n_lbp, n_hog, n_lm = len(sample_lbp), len(sample_hog), lm_ext.n_features
    logger.info(f"Features por imagen: {n_lbp + n_hog + n_lm} (LBP={n_lbp}, HOG={n_hog}, landmarks={n_lm})")

    cols = ["filename"] + [f"lbp_{i}" for i in range(n_lbp)] + \
           [f"hog_{i}" for i in range(n_hog)] + [f"lm_{i}" for i in range(n_lm)]

    rows = []
    n_fail = 0
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Features"):
        img = cv2.imread(str(args.images / row["filename"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            n_fail += 1
            continue

        lbp = extract_lbp(img, **feat_cfg["lbp"])
        h = extract_hog(img, pixels_per_cell=tuple(feat_cfg["hog"]["pixels_per_cell"]),
                        cells_per_block=tuple(feat_cfg["hog"]["cells_per_block"]),
                        orientations=feat_cfg["hog"]["orientations"])
        lm = lm_ext.extract(img)
        rows.append([row["filename"]] + np.concatenate([lbp, h, lm]).tolist())

    if n_fail:
        logger.warning(f"{n_fail} imágenes fallaron")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_csv(args.output, index=False)
    logger.info(f"Features guardadas: {args.output} ({args.output.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
