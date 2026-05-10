# Guía de preparación por dataset

Tu trabajo es producir **dos cosas**:

```
mi_dataset/
├── images/        ← todas las imágenes sueltas aquí
└── metadata.csv   ← un CSV con las etiquetas
```

El CSV debe tener esta estructura (mínimo `filename` y `age`):

```csv
filename,age,gender,ethnicity
foto1.jpg,22,female,white
foto2.jpg,45,male,black
foto3.jpg,67,male,unknown
```

Abajo están las instrucciones específicas para cada dataset. Busca el tuyo y sigue los pasos.

---

## FairFace

**Descarga:** https://github.com/joojs/fairface (los links de Google Drive están en el README del repo)

**Qué te bajas:** un ZIP que al extraer tiene esta estructura:

```
fairface-img-margin025-trainval/
├── train/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── val/
│   ├── 1.jpg
│   └── ...
├── fairface_label_train.csv
└── fairface_label_val.csv
```

**Cómo armar tu carpeta:**

1. Crea la carpeta `mi_dataset/images/`.

2. Copia TODAS las imágenes de `train/` y `val/` a `mi_dataset/images/`. Para evitar colisiones de nombre (ambas carpetas tienen `1.jpg`), renombra las de val. La forma más sencilla: abre una terminal en la carpeta del dataset y corre esto:

```bash
mkdir -p mi_dataset/images
cp train/*.jpg mi_dataset/images/
# Renombrar las de val para que no pisen a las de train
cd val
for f in *.jpg; do cp "$f" "../mi_dataset/images/val_$f"; done
cd ..
```

3. Ahora el CSV. FairFace ya trae CSVs con etiquetas. Ábrelos en Excel o Google Sheets. Vas a ver columnas como `file, age, gender, race, service_test`. Necesitas hacer estos cambios:

   - La columna `file` dice cosas como `train/1.jpg` o `val/1.jpg`. Necesitas que coincida con los nombres que pusiste en `images/`. Para las de train es solo el nombre (`1.jpg`). Para las de val, debe ser `val_1.jpg` (si usaste el renombrado de arriba).
   - La columna `age` viene como rango: "0-2", "3-9", "10-19", "20-29", etc. Conviértela a número usando el punto medio:
     - "0-2" → 1
     - "3-9" → 6
     - "10-19" → 14
     - "20-29" → 24
     - "30-39" → 34
     - "40-49" → 44
     - "50-59" → 54
     - "60-69" → 64
     - "more than 70" → 75
   - Renombra `race` → `ethnicity` y normaliza los valores:
     - "White" → white
     - "Black" → black
     - "East Asian" → asian
     - "Southeast Asian" → asian
     - "Indian" → asian
     - "Latino_Hispanic" → latino
     - "Middle Eastern" → other
   - `gender`: ponlo en minúsculas ("Male" → "male", "Female" → "female")
   - Elimina la columna `service_test`, no la necesitamos.

4. Guarda como `mi_dataset/metadata.csv` con las columnas: `filename,age,gender,ethnicity`

**Tip:** si sabes un poco de Python, esto se puede hacer con pandas en 15 líneas. Si no, Excel funciona perfectamente. Filtra, reemplaza, y guarda como CSV.

---

## UTKFace

**Descarga:** https://susanqq.github.io/UTKFace/ (o Kaggle: https://www.kaggle.com/datasets/jangedoo/utkface-new)

**Qué te bajas:** una carpeta con miles de JPGs sueltos. No hay CSV. Las etiquetas están codificadas en el nombre de cada archivo:

```
25_0_0_20170116174525125.jpg
│  │ │
│  │ └── etnia (0=white, 1=black, 2=asian, 3=indian, 4=other)
│  └──── género (0=male, 1=female)
└─────── edad (número entero)
```

**Cómo armar tu carpeta:**

1. Crea `mi_dataset/images/` y copia todos los JPGs ahí.

2. Para el CSV, tienes que parsear los nombres. Abre una terminal en la carpeta de imágenes y corre este script de Python (guárdalo como `crear_csv.py` y ejecútalo):

```python
import os
import csv

gender_map = {"0": "male", "1": "female"}
ethnicity_map = {"0": "white", "1": "black", "2": "asian", "3": "asian", "4": "other"}

with open("metadata.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "age", "gender", "ethnicity"])

    for fname in sorted(os.listdir("images")):
        if not fname.endswith(".jpg"):
            continue
        parts = fname.split("_")
        if len(parts) < 4:
            continue
        try:
            age = int(parts[0])
            gender = gender_map.get(parts[1], "unknown")
            ethnicity = ethnicity_map.get(parts[2], "unknown")
            if 0 <= age <= 116:
                writer.writerow([fname, age, gender, ethnicity])
        except ValueError:
            continue

print("metadata.csv creado")
```

3. Corre: `python crear_csv.py` (desde la carpeta `mi_dataset/`)

4. Abre `metadata.csv` y verifica que tenga datos. Debería tener ~20,000 filas.

---

## AgeDB

**Descarga:** https://ibug.doc.ic.ac.uk/resources/agedb/ (necesitas pedir la contraseña del ZIP por email a `s.moschoglou@imperial.ac.uk`)

**Qué te bajas:** un ZIP con JPGs. Las etiquetas están en el nombre:

```
0_MariaCallas_35_f.jpg
│ │             │  │
│ │             │  └── género (m=male, f=female)
│ │             └───── edad
│ └─────────────────── nombre de la persona
└───────────────────── ID
```

**Este dataset NO tiene etnia.** Pon `unknown` en esa columna.

**Cómo armar tu carpeta:**

1. Crea `mi_dataset/images/` y copia todos los JPGs.

2. Script para generar el CSV (guárdalo como `crear_csv.py`):

```python
import os
import csv
import re

with open("metadata.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "age", "gender", "ethnicity"])

    for fname in sorted(os.listdir("images")):
        if not fname.endswith(".jpg"):
            continue
        # Formato: ID_Nombre_edad_genero.jpg
        match = re.match(r"(\d+)_(.+)_(\d+)_([mfMF])\.jpg", fname)
        if not match:
            continue
        age = int(match.group(3))
        gender = "male" if match.group(4).lower() == "m" else "female"
        if 0 <= age <= 116:
            writer.writerow([fname, age, gender, "unknown"])

print("metadata.csv creado")
```

3. Corre: `python crear_csv.py`

---

## APPA-REAL

**Descarga:** https://chalearnlap.cvc.uab.cat/dataset/26/description/

**Qué te bajas:** un ZIP con carpetas de imágenes y CSVs:

```
appa-real/
├── train/
│   ├── 00001.jpg
│   └── ...
├── valid/
├── test/
├── gt_avg_train.csv
├── gt_avg_valid.csv
└── gt_avg_test.csv
```

Los CSVs tienen columnas como `file_name` y `real_age`. **No tiene género ni etnia.**

**Cómo armar tu carpeta:**

1. Crea `mi_dataset/images/` y copia las imágenes de `train/`, `valid/` y `test/` ahí. Si hay colisiones de nombre, prefija como hicimos con FairFace.

2. Abre los 3 CSVs (`gt_avg_train.csv`, `gt_avg_valid.csv`, `gt_avg_test.csv`) en Excel. Cada uno tiene `file_name` y `real_age`. Junta los 3 en uno solo.

3. Renombra las columnas: `file_name` → `filename`, `real_age` → `age`. Agrega columnas `gender` y `ethnicity` con valor `unknown` en todas las filas.

4. Guarda como `mi_dataset/metadata.csv`.

---

## IMDB-Wiki

**Descarga:** https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ (puedes bajar solo IMDB o solo Wiki, no necesitas ambos)

**Qué te bajas:** un .tar.gz con imágenes organizadas en subcarpetas + un archivo `.mat` (formato MATLAB) con los metadatos.

```
imdb_crop/
├── 00/
│   ├── nm0000001_rm124825600_1899-5-10_1968.jpg
│   └── ...
├── 01/
├── ...
└── imdb.mat
```

**Este es el dataset más complicado** porque el `.mat` no se puede abrir con Excel. Necesitas Python obligatoriamente.

**Cómo armar tu carpeta:**

1. Crea `mi_dataset/images/`.

2. Guarda este script como `crear_csv.py` y córrelo:

```python
import os
import csv
import numpy as np
from scipy.io import loadmat

# Cambia esta ruta al archivo .mat que descargaste
MAT_FILE = "imdb_crop/imdb.mat"
IMAGES_DIR = "imdb_crop"

mat = loadmat(MAT_FILE)
data = mat["imdb"][0, 0]  # si descargaste wiki, cambia "imdb" por "wiki"

dob = data["dob"][0]
photo_taken = data["photo_taken"][0]
gender = data["gender"][0]
full_path = [p[0] for p in data["full_path"][0]]
face_score = data["face_score"][0]
second_face_score = data["second_face_score"][0]

with open("metadata.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "age", "gender", "ethnicity"])

    copied = 0
    for i in range(len(dob)):
        try:
            birth_year = int((dob[i] - 366) / 365.25) + 1
            age = int(photo_taken[i]) - birth_year
        except (ValueError, OverflowError):
            continue

        if age < 0 or age > 100:
            continue
        if face_score[i] < 1.0:
            continue
        if not np.isnan(second_face_score[i]):
            continue

        g = "unknown"
        if not np.isnan(gender[i]):
            g = "male" if gender[i] == 1 else "female"

        # Copiar imagen a mi_dataset/images/ con nombre plano
        src = os.path.join(IMAGES_DIR, full_path[i])
        if not os.path.exists(src):
            continue

        new_name = f"imdb_{copied:06d}.jpg"
        dst = os.path.join("mi_dataset/images", new_name)
        os.makedirs("mi_dataset/images", exist_ok=True)

        import shutil
        shutil.copy2(src, dst)
        writer.writerow([new_name, age, g, "unknown"])
        copied += 1

        if copied % 5000 == 0:
            print(f"  {copied} imagenes procesadas...")

print(f"Listo. {copied} imagenes copiadas, metadata.csv creado")
```

3. Instala scipy si no la tienes: `pip install scipy numpy`
4. Corre: `python crear_csv.py`
5. Mueve `metadata.csv` a `mi_dataset/`

**Nota:** IMDB-Wiki tiene 500k+ imágenes. Si no quieres procesarlas todas, agrega un límite al script (ej. `if copied >= 50000: break`).

---

## Fotos propias o de otra fuente

Si alguien trae fotos de Google, fotos propias, o cualquier otra fuente:

1. Pon todas las imágenes en `mi_dataset/images/`.
2. Crea `metadata.csv` a mano en Excel o Google Sheets. Una fila por imagen, con el nombre exacto del archivo y la edad estimada.
3. Si no sabes la etnia o el género, pon `unknown`.

Es tedioso para muchas imágenes, pero funciona para complementar con 50-200 fotos extras.

---

## Verificación rápida

Antes de seguir con los scripts, abre tu `metadata.csv` y revisa:

- ¿La primera fila dice `filename,age,gender,ethnicity`?
- ¿Los nombres de archivo coinciden con lo que hay en `images/`?
- ¿Las edades son números enteros razonables (0-116)?
- ¿No hay filas vacías o con datos raros?

Si todo se ve bien, ya puedes correr `validate_input.py`.
