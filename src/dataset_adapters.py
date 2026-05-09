"""
dataset_adapters.py
====================

Cada dataset (FairFace, UTKFace, AgeDB, APPA-REAL, IMDB-Wiki) tiene su propio
formato de etiquetas. Este módulo unifica todo bajo una interfaz común.

¿POR QUÉ? Para que el resto del pipeline (build_partial_manifest.py, etc.)
no tenga que saber cómo se llaman las columnas en cada dataset, ni cómo se
codifican raza/género. Cada adaptador devuelve un DataFrame estandarizado.

Cada adaptador implementa el método `read_labels()` que devuelve un DataFrame
con columnas:

    - original_filename: nombre del archivo en el dataset original
    - age: edad numérica entera
    - gender: 'male' | 'female' | 'unknown'
    - ethnicity: del vocabulario común (ver spec.yaml)

CÓMO AGREGAR UN NUEVO DATASET:
    1. Crear una clase que herede de DatasetAdapter
    2. Implementar read_labels()
    3. Registrarla en ADAPTERS al final del archivo
"""

from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np


# =============================================================================
# Mapeos de etnia: cada dataset usa sus propias categorías. Aquí las
# normalizamos al vocabulario común (white, black, asian, latino, other, unknown)
# =============================================================================

# FairFace usa 7 categorías
FAIRFACE_ETHNICITY_MAP: Dict[str, str] = {
    "White":            "white",
    "Black":            "black",
    "Latino_Hispanic":  "latino",
    "East Asian":       "asian",
    "Southeast Asian":  "asian",
    "Indian":           "asian",       # Decisión de equipo: agrupar South Asian con Asian
    "Middle Eastern":   "other",
}

# UTKFace usa códigos numéricos 0-4
UTKFACE_ETHNICITY_MAP: Dict[int, str] = {
    0: "white",
    1: "black",
    2: "asian",
    3: "asian",      # Indian → asian (consistencia con FairFace)
    4: "other",      # 'Others' (incluye Latino, Middle Eastern, etc.)
}

# UTKFace género: 0=male, 1=female
UTKFACE_GENDER_MAP: Dict[int, str] = {0: "male", 1: "female"}


# =============================================================================
# Clase base
# =============================================================================
class DatasetAdapter:
    """Interfaz común para todos los adaptadores de dataset."""

    name: str = "base"

    def __init__(self, raw_path: str | Path):
        self.raw_path = Path(raw_path)
        if not self.raw_path.exists():
            raise FileNotFoundError(f"No existe el directorio raw: {self.raw_path}")

    def read_labels(self) -> pd.DataFrame:
        """
        Devuelve un DataFrame con columnas estandarizadas:
        original_filename, age, gender, ethnicity.
        """
        raise NotImplementedError("Cada adaptador debe implementar read_labels()")


# =============================================================================
# FairFace
# =============================================================================
class FairFaceAdapter(DatasetAdapter):
    """
    FairFace viene con dos CSVs (train y val) y dos carpetas de imágenes.
    Etiquetas: file (path relativo), age (rango string), gender, race, service_test.

    Edad viene como string ("0-2", "3-9", "20-29", ..., "more than 70").
    La convertimos a edad numérica usando el punto medio del rango.
    """
    name = "fairface"

    # Mapeo de rangos de edad de FairFace al punto medio (edad numérica)
    AGE_RANGE_MIDPOINTS = {
        "0-2": 1, "3-9": 6, "10-19": 14, "20-29": 24,
        "30-39": 34, "40-49": 44, "50-59": 54, "60-69": 64,
        "more than 70": 75,
    }

    def read_labels(self) -> pd.DataFrame:
        # FairFace tiene CSVs separados para train y val. Los unimos.
        dfs = []
        for split_csv in ["fairface_label_train.csv", "fairface_label_val.csv"]:
            csv_path = self.raw_path / split_csv
            if csv_path.exists():
                dfs.append(pd.read_csv(csv_path))
        if not dfs:
            raise FileNotFoundError(
                f"No encontré CSVs de FairFace en {self.raw_path}. "
                f"Esperaba: fairface_label_train.csv y fairface_label_val.csv"
            )
        df = pd.concat(dfs, ignore_index=True)

        # Construir DataFrame normalizado
        out = pd.DataFrame()
        out["original_filename"] = df["file"]               # Ej: "train/1.jpg"
        out["age"] = df["age"].map(self.AGE_RANGE_MIDPOINTS).astype("Int64")
        out["gender"] = df["gender"].str.lower().fillna("unknown")
        out["ethnicity"] = df["race"].map(FAIRFACE_ETHNICITY_MAP).fillna("unknown")

        # Eliminar filas sin edad válida
        out = out.dropna(subset=["age"]).reset_index(drop=True)
        return out


# =============================================================================
# UTKFace
# =============================================================================
class UTKFaceAdapter(DatasetAdapter):
    """
    UTKFace codifica las etiquetas EN EL NOMBRE DEL ARCHIVO:
    [age]_[gender]_[race]_[date&time].jpg
    Ejemplo: 25_0_3_20170104232355426.jpg.chip.jpg
    """
    name = "utkface"

    # Regex para parsear nombres de archivo
    FILENAME_PATTERN = re.compile(r"^(\d+)_(\d+)_(\d+)_.+\.jpg(\.chip\.jpg)?$")

    def read_labels(self) -> pd.DataFrame:
        records = []
        # UTKFace puede venir todo plano o en subcarpetas
        for img_path in self.raw_path.rglob("*.jpg"):
            m = self.FILENAME_PATTERN.match(img_path.name)
            if not m:
                continue   # Saltar archivos con formato inesperado

            age = int(m.group(1))
            gender_code = int(m.group(2))
            race_code = int(m.group(3))

            # Filtros de cordura: edades imposibles indican mala etiqueta
            if age < 0 or age > 116:
                continue

            records.append({
                "original_filename": str(img_path.relative_to(self.raw_path)),
                "age": age,
                "gender": UTKFACE_GENDER_MAP.get(gender_code, "unknown"),
                "ethnicity": UTKFACE_ETHNICITY_MAP.get(race_code, "unknown"),
            })

        if not records:
            raise RuntimeError(
                f"No encontré imágenes con formato válido en {self.raw_path}. "
                "Esperaba archivos como '25_0_3_20170104232355426.jpg'."
            )
        return pd.DataFrame(records)


# =============================================================================
# AgeDB
# =============================================================================
class AgeDBAdapter(DatasetAdapter):
    """
    AgeDB también codifica etiquetas en el nombre del archivo:
    {id}_{NombreApellido}_{age}_{gender}.jpg
    Ejemplo: 0_MariaCallas_35_f.jpg

    AgeDB NO incluye etiqueta de etnia → todas quedan como 'unknown'.
    Esto es esperado y está bien: el manifest tolera 'unknown'.
    """
    name = "agedb"

    FILENAME_PATTERN = re.compile(r"^(\d+)_([A-Za-z]+)_(\d+)_([mfMF])\.jpg$")
    GENDER_MAP = {"m": "male", "M": "male", "f": "female", "F": "female"}

    def read_labels(self) -> pd.DataFrame:
        records = []
        for img_path in self.raw_path.rglob("*.jpg"):
            m = self.FILENAME_PATTERN.match(img_path.name)
            if not m:
                continue

            age = int(m.group(3))
            gender_code = m.group(4)

            if age < 0 or age > 116:
                continue

            records.append({
                "original_filename": str(img_path.relative_to(self.raw_path)),
                "age": age,
                "gender": self.GENDER_MAP.get(gender_code, "unknown"),
                "ethnicity": "unknown",   # AgeDB no anota etnia
            })

        if not records:
            raise RuntimeError(f"No encontré imágenes en formato AgeDB en {self.raw_path}")
        return pd.DataFrame(records)


# =============================================================================
# APPA-REAL
# =============================================================================
class APPARealAdapter(DatasetAdapter):
    """
    APPA-REAL viene con CSVs (gt_train.csv, gt_valid.csv, gt_test.csv)
    Columnas: file_name, real_age, apparent_age_avg, ...

    Usamos 'real_age' como ground truth (es la edad real, no la aparente).
    No incluye etnia ni género en el CSV principal.
    """
    name = "appa_real"

    def read_labels(self) -> pd.DataFrame:
        dfs = []
        for split_csv in ["gt_train.csv", "gt_valid.csv", "gt_test.csv",
                          "gt_avg_train.csv", "gt_avg_valid.csv", "gt_avg_test.csv"]:
            csv_path = self.raw_path / split_csv
            if csv_path.exists():
                dfs.append(pd.read_csv(csv_path))
        if not dfs:
            raise FileNotFoundError(f"No encontré CSVs de APPA-REAL en {self.raw_path}")

        df = pd.concat(dfs, ignore_index=True)

        out = pd.DataFrame()
        out["original_filename"] = df["file_name"]
        out["age"] = df["real_age"].astype("Int64")
        out["gender"] = "unknown"      # APPA-REAL no incluye género en gt principal
        out["ethnicity"] = "unknown"   # Tampoco etnia

        out = out.dropna(subset=["age"]).reset_index(drop=True)
        return out


# =============================================================================
# IMDB-Wiki
# =============================================================================
class IMDBWikiAdapter(DatasetAdapter):
    """
    IMDB-Wiki viene en formato MATLAB (.mat) con metadatos:
    - dob (matlab serial date) → fecha de nacimiento
    - photo_taken (year) → año de la foto
    - gender (0=female, 1=male, NaN=unknown)
    - full_path (ruta de la imagen)
    - face_score (calidad de detección, descartar < 1.0)
    - second_face_score (si tiene 2 caras detectadas, descartar)

    Edad = photo_taken - año(dob)

    NO incluye etnia (todas las celebridades son mayoritariamente blancas
    de Hollywood, MUY sesgado). Marcamos ethnicity='unknown' para no mentir.

    Las etiquetas son RUIDOSAS (extraídas automáticamente). Filtramos por
    face_score para quedarnos con las más confiables.
    """
    name = "imdb_wiki"

    def read_labels(self) -> pd.DataFrame:
        from scipy.io import loadmat

        # Buscar el .mat (puede ser imdb.mat o wiki.mat)
        mat_files = list(self.raw_path.rglob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(f"No encontré .mat en {self.raw_path}")

        all_records = []
        for mat_file in mat_files:
            mat = loadmat(str(mat_file))
            # La key principal es 'imdb' o 'wiki' dependiendo del subset
            key = "imdb" if "imdb" in mat else "wiki"
            data = mat[key][0, 0]

            dob = data["dob"][0]                      # Matlab serial date
            photo_taken = data["photo_taken"][0]      # Año
            gender = data["gender"][0]                # 0=female, 1=male, NaN=unknown
            full_path = [p[0] for p in data["full_path"][0]]
            face_score = data["face_score"][0]
            second_face_score = data["second_face_score"][0]

            for i in range(len(dob)):
                # Convertir serial date de Matlab a año Python
                # Matlab: días desde 0000-01-01; Python: ofset 366 días
                try:
                    birth_year = int((dob[i] - 366) / 365.25) + 1
                    age = int(photo_taken[i]) - birth_year
                except (ValueError, OverflowError):
                    continue

                if age < 0 or age > 116:
                    continue

                # Filtrar por calidad de detección facial
                if face_score[i] < 1.0:
                    continue
                if not np.isnan(second_face_score[i]):
                    continue   # Hay 2 caras, ambiguo

                gender_str = "unknown"
                if not np.isnan(gender[i]):
                    gender_str = "male" if gender[i] == 1 else "female"

                all_records.append({
                    "original_filename": full_path[i],
                    "age": age,
                    "gender": gender_str,
                    "ethnicity": "unknown",   # IMDB-Wiki es mayoritariamente blanco, no anotamos
                })

        if not all_records:
            raise RuntimeError("No quedaron registros válidos tras filtrar IMDB-Wiki")
        return pd.DataFrame(all_records)


# =============================================================================
# Registro de adaptadores: nombre → clase
# =============================================================================
ADAPTERS: Dict[str, type] = {
    "fairface":  FairFaceAdapter,
    "utkface":   UTKFaceAdapter,
    "agedb":     AgeDBAdapter,
    "appa_real": APPARealAdapter,
    "imdb_wiki": IMDBWikiAdapter,
}


def get_adapter(dataset_name: str, raw_path: str | Path) -> DatasetAdapter:
    """Factory: devuelve la instancia del adaptador apropiado."""
    if dataset_name not in ADAPTERS:
        raise ValueError(
            f"Dataset desconocido: '{dataset_name}'. "
            f"Disponibles: {list(ADAPTERS.keys())}"
        )
    return ADAPTERS[dataset_name](raw_path)
