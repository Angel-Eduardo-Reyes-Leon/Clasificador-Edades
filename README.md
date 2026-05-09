# Clasificador-Edades

Sistema de clasificación de edad facial en 3 grupos (joven / adulto / viejo) usando **algoritmos clásicos de Machine Learning** (SVM, XGBoost, Logistic Regression). Sin redes neuronales para el modelo de clasificación.

---

## Tabla de contenidos

1. [Idea general del proyecto](#idea-general-del-proyecto)
2. [Estructura del repositorio](#estructura-del-repositorio)
3. [Instalación](#instalación)
4. [Pipeline completo (qué hace cada script)](#pipeline-completo-qué-hace-cada-script)
5. [Plan de trabajo en equipo](#plan-de-trabajo-en-equipo)
6. [Convenciones que TODOS deben seguir](#convenciones-que-todos-deben-seguir)
7. [Decisiones técnicas y por qué](#decisiones-técnicas-y-por-qué)
8. [Datasets recomendados](#datasets-recomendados)

---

## Idea general del proyecto

**Input**: una foto de un rostro.
**Output**: una de 3 clases — `young` (0–25 años), `adult` (26–60), `old` (61+).

**Restricción importante**: solo Machine Learning clásico, sin redes neuronales. Esto significa que **no podemos darle píxeles al modelo directamente**; tenemos que extraer features numéricas a mano (LBP, HOG, landmarks) y entrenar SVM/XGBoost sobre esas features.

### Arquitectura conceptual

```
imagen cruda → detectar cara → recortar → grayscale 128x128
                                              ↓
                               extraer features (LBP+HOG+landmarks)
                                              ↓
                                   vector de ~340 números
                                              ↓
                                  StandardScaler + PCA opcional
                                              ↓
                            entrenar SVM / XGBoost / LogisticRegression
                                              ↓
                          predicción: young | adult | old
```

---

## Estructura del repositorio

```
Clasificador-Edades/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── spec.yaml              ← CONVENCIONES DEL EQUIPO (no editar sin avisar)
│
├── src/
│   ├── __init__.py
│   ├── dataset_adapters.py    ← Lee etiquetas de cada dataset (interfaz común)
│   ├── preprocess.py          ← PASO 2: detectar cara, recortar, resize
│   ├── build_partial_manifest.py ← PASO 3: generar partial_manifest_X.csv
│   ├── extract_features.py    ← PASO 4: extraer LBP+HOG+landmarks
│   ├── validate_partial.py    ← PASO 5: verificar antes de subir
│   ├── merge_team.py          ← PASO 6: unir trabajo de todos (líder técnico)
│   ├── train.py               ← PASO 7: entrenar modelos
│   └── evaluate.py            ← PASO 8: evaluar y auditar sesgo
│
├── data/
│   ├── raw/                   ← Datasets originales (NO subir a Git)
│   ├── images/                ← Imágenes procesadas planas (NO subir a Git)
│   ├── partial_outputs/       ← CSVs parciales de cada miembro
│   └── final/                 ← manifest.csv (SÍ a Git) + features.csv (NO)
│
├── models/                    ← Modelos entrenados (.joblib)
├── reports/                   ← Métricas, gráficas, audit de sesgo
└── notebooks/                 ← Para exploración manual
```

---

## Instalación

```bash
# 1. Clonar el repo
git clone https://github.com/TU_USUARIO/Clasificador-Edades.git
cd Clasificador-Edades

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

**Notas sobre dependencias**:

- **TensorFlow** se instala como dependencia de MTCNN. Pesa ~500 MB. No es para entrenar redes — solo para detectar caras.
- **MediaPipe** se usa para landmarks faciales (468 puntos). Más fácil de instalar que dlib (no requiere cmake).
- En Windows, si `mtcnn` falla, intentar: `pip install tensorflow==2.16.1 mtcnn==0.1.1 --no-cache-dir`.
- En Mac M1/M2: `pip install tensorflow-macos` en vez de `tensorflow`.

---

## Pipeline completo (qué hace cada script)

| # | Script | Quién lo corre | Output |
|---|--------|---------------|--------|
| 1 | (descargar dataset) | Cada miembro | `data/raw/{dataset}/` |
| 2 | `preprocess.py` | Cada miembro | `data/images/{dataset}_NNNNNN.jpg` + `mapping_{dataset}.csv` |
| 3 | `build_partial_manifest.py` | Cada miembro | `partial_manifest_{dataset}.csv` |
| 4 | `extract_features.py` | Cada miembro | `partial_features_{dataset}.csv` |
| 5 | `validate_partial.py` | Cada miembro | (verifica) |
| 6 | (subir a Drive) | Cada miembro | — |
| 7 | `merge_team.py` | Líder técnico | `manifest.csv` + `features.csv` |
| 8 | `train.py` | Cualquiera | `models/*.joblib` |
| 9 | `evaluate.py` | Cualquiera | `reports/*.png` + métricas |

### Comandos completos (ejemplo con FairFace)

```bash
# PASO 2: preprocesar imágenes
python src/preprocess.py \
    --dataset fairface \
    --input data/raw/fairface \
    --output data/images

# PASO 3: construir manifest parcial
python src/build_partial_manifest.py \
    --dataset fairface \
    --raw data/raw/fairface \
    --mapping data/partial_outputs/mapping_fairface.csv \
    --output data/partial_outputs/partial_manifest_fairface.csv

# PASO 4: extraer features
python src/extract_features.py \
    --images data/images \
    --manifest data/partial_outputs/partial_manifest_fairface.csv \
    --output data/partial_outputs/partial_features_fairface.csv

# PASO 5: validar antes de subir
python src/validate_partial.py \
    --dataset fairface \
    --images data/images \
    --manifest data/partial_outputs/partial_manifest_fairface.csv \
    --features data/partial_outputs/partial_features_fairface.csv

# === Después de que TODOS terminaron y subieron al Drive ===

# PASO 6: merge (líder técnico)
python src/merge_team.py

# PASO 7: entrenar
python src/train.py

# PASO 8: evaluar
python src/evaluate.py
```

---

## Plan de trabajo en equipo

### Asignación de datasets (ejemplo)

| Miembro | Dataset | Tamaño aprox |
|---|---|---|
| Persona A | FairFace | ~108k imgs |
| Persona B | UTKFace | ~20k imgs |
| Persona C | AgeDB | ~12k imgs |
| Persona D | APPA-REAL | ~7.5k imgs |
| Persona E | IMDB-Wiki (subset) | ~50k imgs |

### Fases del proyecto

**FASE 0 — Setup (líder técnico, antes que arranquen los demás)**

- Crear repo y compartirlo
- Crear carpeta compartida en Google Drive con esta estructura:
  ```
  ProyectoEdad_Compartido/
  ├── partial_outputs/
  │   ├── fairface/
  │   ├── utkface/
  │   └── ...
  └── final/
  ```
- Asignar datasets

**FASE 1 — Trabajo individual (cada miembro, en paralelo)**

Cada quien:
1. Descarga su dataset → `data/raw/{su_dataset}/`
2. Corre `preprocess.py` → genera imágenes en `data/images/`
3. Corre `build_partial_manifest.py` → genera `partial_manifest_{su_dataset}.csv`
4. Corre `extract_features.py` → genera `partial_features_{su_dataset}.csv`
5. Corre `validate_partial.py` → debe pasar todo
6. Sube los 3 archivos (sus imágenes + 2 CSVs) al Drive en la carpeta de su dataset
7. Avisa al equipo

**FASE 2 — Merge (líder técnico)**

Cuando TODOS terminaron:
1. Bajar todo del Drive a una sola máquina
2. Correr `merge_team.py`
3. Subir `manifest.csv` y `features.csv` finales al Drive
4. Commitear `manifest.csv` al repo (es ligero, ~5 MB)

**FASE 3 — Entrenamiento y evaluación (equipo)**

Pueden dividirse:
- A: corre `train.py` con distintos hiperparámetros
- B: corre `evaluate.py` y genera todos los reportes
- C: redacta análisis del audit de sesgo
- D: prepara presentación

---

## Convenciones que TODOS deben seguir

Estas viven en `configs/spec.yaml`. **No las cambien sin avisar al equipo**:

| Cosa | Valor |
|---|---|
| Tamaño de imagen | 128 × 128 |
| Color | Grayscale |
| Formato | JPG calidad 95 |
| Naming | `{dataset}_{NNNNNN}.jpg` (6 dígitos con padding) |
| Buckets de edad | young (0–25), adult (26–60), old (61+) |
| Vocabulario etnia | white, black, asian, latino, other, unknown |
| Vocabulario género | male, female, unknown |
| Splits | 70% train / 15% val / 15% test |
| Random seed | 42 |

### Reglas de oro

1. **Nadie escribe scripts propios**. Todos usan los del repo. Si encuentran un bug, hacen PR + revisión.
2. **Nadie modifica `spec.yaml` localmente**. Si necesitan cambiar algo, lo discuten en grupo.
3. **Nadie sube imágenes sin pasar `validate_partial.py`**. Es obligatorio.
4. **Nadie asigna splits localmente**. El split lo decide `merge_team.py` sobre el conjunto completo.
5. **Avisen cuando terminen su fase**. El líder no puede hacer merge hasta que todos estén.

---

## Decisiones técnicas y por qué

### ¿Por qué carpeta plana de imágenes + CSV con metadatos?

Porque **es el patrón estándar en ML/ciencia de datos**:
- ImageNet, COCO, Open Images: todos hacen esto.
- Pandas, sklearn, PyTorch: todos esperan datos así.
- Una carpeta con subcarpetas (young/, adult/, old/) **borra información** (etnia, género, dataset original). El CSV preserva todo.

### ¿Por qué entrenamos con 7 sub-buckets si al final reportamos 3?

La clase `young` (0–25) es brutalmente heterogénea: un bebé y una persona de 24 son visualmente muy distintos. Si entrenamos directo en 3 clases, el modelo se confunde.

Estrategia: entrenamos con **age_sub** (baby, child, teen, young_adult, middle_adult, mature_adult, old) y al evaluar colapsamos a 3 clases. Esto suele dar +5–10% de accuracy en la clase `young`.

### ¿Por qué LBP + HOG + landmarks?

- **LBP** (Local Binary Patterns): captura textura local (arrugas, poros). Es la feature más predictiva de edad.
- **HOG** (Histogram of Oriented Gradients): captura bordes y orientaciones (líneas de expresión).
- **Landmarks ratios**: capturan proporciones faciales. Especialmente útiles para distinguir niños de adultos (las proporciones cambian con el desarrollo).

Esta combinación es estándar en papers de age estimation con ML clásico (Bereta 2013, Ylioinas 2013).

### ¿Por qué MediaPipe en vez de dlib?

- dlib requiere cmake, falla a menudo en Windows.
- MediaPipe se instala con un `pip install` y da 468 landmarks vs 68 de dlib.
- Funcionalmente equivalente para este uso.

### ¿Por qué NO quitar accesorios (lentes, barba) con APIs?

Aunque el profe lo sugirió, va contra el modelo:
- Las APIs de inpainting introducen píxeles sintéticos que LBP/HOG detectan como artefactos → contamina las features.
- Barba y canas correlacionan POSITIVAMENTE con edad → son señal útil, no ruido.
- En producción la gente tendrá accesorios → entrenar con accesorios respeta la distribución real.

Mejor estrategia: agregar columnas `has_glasses`, `has_beard` al manifest y reportar accuracy condicional en el reporte.

### ¿Por qué pHash para detectar duplicados?

IMDB-Wiki tiene la misma celebridad cientos de veces (mismas fotos con diferentes recortes). Si una imagen está en train Y en test, el modelo "memorizó" la respuesta y reporta accuracy inflada. pHash detecta similitud visual robusta a recompresión y resize.

---

## Datasets recomendados

| Dataset | URL | Imágenes | Notas |
|---|---|---|---|
| **FairFace** | https://github.com/joojs/fairface | 108k | El más balanceado por etnia. Imprescindible. |
| **UTKFace** | https://susanqq.github.io/UTKFace/ | 20k | Edad continua (0–116). Buena variedad. |
| **AgeDB** | https://ibug.doc.ic.ac.uk/resources/agedb/ | 12k | Etiquetas más limpias. Sin etnia. Pedir password por email. |
| **APPA-REAL** | https://chalearnlap.cvc.uab.cat/dataset/26/description/ | 7.5k | Tiene edad real Y aparente. |
| **IMDB-Wiki** | https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/ | 500k | El más grande pero ruidoso. Mayoría blancos. |
| **AAF (All-Age-Faces)** | https://github.com/JingchunCheng/All-Age-Faces-Dataset | 13k | Mayoría asiáticos, complementa los demás. |

Todos son no comerciales (uso académico). Verifiquen las licencias antes de usar en producción.

---

## Output esperado del proyecto

Al final, el equipo debe entregar:

1. **`data/final/manifest.csv`** — versionado en Git, reproducible.
2. **Modelos entrenados** (`models/*.joblib`) — al Drive.
3. **Reporte de evaluación** con:
   - Accuracy global de cada modelo
   - Confusion matrix (3 clases y 7 sub-clases)
   - Audit de sesgo: accuracy por etnia, género, y combinaciones
   - Análisis de en qué clase falla más cada modelo
   - Comparación de los 3 modelos (LogReg vs SVM vs XGBoost)

---

## Contacto y soporte

Si tienen dudas o algo no funciona:
1. Revisar este README primero
2. Revisar el docstring del script (cada uno tiene explicación detallada arriba)
3. Preguntar al equipo

**Buena suerte con el proyecto. 🤖**
