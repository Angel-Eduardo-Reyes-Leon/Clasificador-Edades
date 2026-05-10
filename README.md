# Clasificador-Edades

Clasificador de edad facial en 3 grupos (joven / adulto / viejo) con ML clásico (SVM, XGBoost, Logistic Regression). Sin redes neuronales.

## Cómo funciona

```
imagen → detectar cara → recortar → grayscale 128x128 → extraer features (LBP+HOG+landmarks) → SVM/XGBoost
```

No le damos píxeles al modelo. Extraemos ~340 números por imagen que representan textura de piel, bordes y proporciones faciales. El modelo aprende sobre esos números.

## Estructura

```
Clasificador-Edades/
├── docs/
│   ├── DATASET_GUIDE.md         ← Cómo preparar cada dataset
│   └── TEAM_INSTRUCTIONS.md     ← Paso a paso para cada miembro
├── src/
│   ├── validate_input.py        ← Verifica tu carpeta antes de procesar
│   ├── preprocess.py            ← Detecta caras, recorta, resize
│   ├── extract_features.py      ← LBP + HOG + landmarks → CSV numérico
│   ├── validate_output.py       ← Verifica antes de subir al Drive
│   ├── merge_team.py            ← Junta outputs de todos + splits
│   ├── train.py                 ← Entrena modelos
│   └── evaluate.py              ← Métricas + audit de sesgo
├── configs/spec.yaml            ← Convenciones del equipo
├── data/                        ← Datos (no van a Git, van a Drive)
├── models/                      ← Modelos entrenados
└── reports/                     ← Gráficas y resultados
```

## Para empezar

```bash
git clone https://github.com/Angel-Eduardo-Reyes-Leon/Clasificador-Edades.git
cd Clasificador-Edades
python -m venv venv
source venv/Scripts/activate    # Windows
pip install -r requirements.txt
```

Después lee `docs/TEAM_INSTRUCTIONS.md`. Ahí está todo lo que tienes que hacer con tu dataset.

## La idea clave

Cada miembro del equipo trabaja con un dataset distinto. Cada dataset tiene formato diferente. En vez de que un script intente leer todos los formatos, **cada persona estandariza su dataset a mano**:

1. Pone todas las imágenes en una sola carpeta
2. Crea un CSV simple con filename, age, gender, ethnicity
3. Corre los scripts del pipeline

Las instrucciones para crear ese CSV por dataset están en `docs/DATASET_GUIDE.md`.

## Pipeline

| Paso | Script | Quién |
|------|--------|-------|
| 1 | (preparar carpeta + CSV) | Cada miembro, siguiendo DATASET_GUIDE.md |
| 2 | `validate_input.py` | Cada miembro |
| 3 | `preprocess.py` | Cada miembro |
| 4 | `extract_features.py` | Cada miembro |
| 5 | `validate_output.py` | Cada miembro |
| 6 | (subir a Google Drive) | Cada miembro |
| 7 | `merge_team.py` | Líder técnico |
| 8 | `train.py` | Equipo |
| 9 | `evaluate.py` | Equipo |

## Dónde va cada cosa

- **GitHub**: código, configs, manifest final (texto ligero)
- **Google Drive**: imágenes, features.csv, modelos (archivos pesados, no caben en Git)

## Datasets

| Dataset | Link | Tamaño |
|---------|------|--------|
| FairFace | https://github.com/joojs/fairface | 108k |
| UTKFace | https://susanqq.github.io/UTKFace/ | 20k |
| AgeDB | https://ibug.doc.ic.ac.uk/resources/agedb/ | 12k |
| APPA-REAL | https://chalearnlap.cvc.uab.cat/dataset/26/description/ | 7.5k |
| IMDB-Wiki | https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ | 500k |
| Faces: Age Detection from Images | https://www.kaggle.com/datasets/arashnic/faces-age-detection-dataset | 110.66 MB |

## Convenciones

Todo está en `configs/spec.yaml`. No cambiar sin avisar al equipo:
- Imágenes: 128x128, grayscale, JPG
- Buckets: young (0-25), adult (26-60), old (61+)
- Splits: 70% train, 15% val, 15% test
