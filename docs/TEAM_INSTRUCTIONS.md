# Qué hacer con tu dataset — paso a paso

Cada miembro del equipo tiene un dataset distinto. El objetivo es que todos produzcan el mismo tipo de output para que al final se pueda juntar todo.

---

## Lo que tienes que producir

Dos cosas, punto:

```
mi_dataset/
├── images/        ← todas tus imágenes sueltas aquí
└── metadata.csv   ← un CSV con filename, age, gender, ethnicity
```

Cómo crear esa carpeta y ese CSV depende de tu dataset. Ve a `docs/DATASET_GUIDE.md`, busca el tuyo, y sigue las instrucciones de ahí.

---

## Antes de empezar

Clona el repo y prepara tu entorno:

```bash
git clone https://github.com/Angel-Eduardo-Reyes-Leon/Clasificador-Edades.git
cd Clasificador-Edades
python -m venv venv
source venv/Scripts/activate    # Windows con Git Bash
pip install -r requirements.txt
```

Si algo falla al instalar, pega el error en el grupo.

---

## Paso 1 — Descarga tu dataset

Los links están en `docs/DATASET_GUIDE.md`. Descárgalo y ponlo donde quieras en tu máquina.

---

## Paso 2 — Prepara tu carpeta estandarizada

Sigue las instrucciones de `docs/DATASET_GUIDE.md` para tu dataset. Al terminar debes tener:

- `mi_dataset/images/` con todas las imágenes sueltas (sin subcarpetas)
- `mi_dataset/metadata.csv` con las etiquetas

Ponlo dentro del repo, en `data/raw/`:

```
data/raw/
└── mi_dataset/
    ├── images/
    │   ├── foto1.jpg
    │   ├── foto2.jpg
    │   └── ...
    └── metadata.csv
```

---

## Paso 3 — Verifica que esté bien

```bash
python src/validate_input.py --input data/raw/mi_dataset
```

Este script revisa:
- Que `metadata.csv` tenga las columnas correctas
- Que las imágenes referenciadas en el CSV existan en `images/`
- Que las edades sean números razonables
- Que géneros y etnias usen el vocabulario definido

Si algo falla, te dice exactamente qué. Corrige y vuelve a correr hasta que pase.

---

## Paso 4 — Procesa las imágenes

```bash
python src/preprocess.py \
    --input data/raw/mi_dataset \
    --name mi_dataset \
    --output data/processed
```

Cambia `mi_dataset` por el nombre de tu dataset (fairface, utkface, agedb, etc.).

Qué hace: toma cada imagen, detecta la cara, la recorta, la convierte a escala de grises, la redimensiona a 128x128, y la guarda con nombre prefijado en `data/processed/`. También genera `data/partial_outputs/partial_manifest_{nombre}.csv`.

Va a tardar. Si tu dataset es grande (FairFace, IMDB-Wiki), puede tomar 1-3 horas. Déjalo corriendo.

---

## Paso 5 — Extrae las features

```bash
python src/extract_features.py \
    --images data/processed \
    --manifest data/partial_outputs/partial_manifest_mi_dataset.csv \
    --output data/partial_outputs/partial_features_mi_dataset.csv
```

Qué hace: para cada imagen procesada, calcula un vector de ~340 números que describen la textura de la piel (LBP), los bordes (HOG), y las proporciones de la cara (landmarks). Esos números son lo que el modelo va a usar para aprender.

---

## Paso 6 — Verifica antes de subir

```bash
python src/validate_output.py \
    --name mi_dataset \
    --images data/processed \
    --manifest data/partial_outputs/partial_manifest_mi_dataset.csv \
    --features data/partial_outputs/partial_features_mi_dataset.csv
```

Si dice "Todo en orden", puedes subir.

---

## Paso 7 — Sube al Google Drive

Sube tres cosas a la carpeta compartida del Drive, en `partial_outputs/{tu_dataset}/`:

1. Las imágenes procesadas de tu dataset (las que empiezan con tu prefijo en `data/processed/`). Comprímelas en un ZIP primero.
2. `partial_manifest_{tu_dataset}.csv`
3. `partial_features_{tu_dataset}.csv`

Avisa en el grupo cuando termines.

---

## Después de esto

Cuando todos terminen, el líder técnico corre `merge_team.py` para juntar todo. Después ya se puede entrenar y evaluar.

No necesitas hacer nada más hasta que te avisen.

---

## Si algo falla

- Pega el error completo en el grupo. No intentes arreglar a mano.
- Si `preprocess.py` se tarda mucho, es normal. Déjalo corriendo de noche.
- Si alguna imagen no se procesa (no le detecta cara), el script la salta y sigue. Al final te dice cuántas falló.
- Si `metadata.csv` tiene errores, arréglalo en Excel y vuelve a correr `validate_input.py`.

## Tiempos estimados

- FairFace (108k imágenes): 2-3 horas
- IMDB-Wiki (50k subset): 1-2 horas
- UTKFace (20k): 45 min
- AgeDB (12k): 30 min
- APPA-REAL (7.5k): 20 min
