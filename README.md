---

# 🏠 Pipeline de Limpieza y Selección de Casas

Este repositorio contiene un pipeline en **Python** para la **detección de casas en imágenes satelitales o fotográficas**, incluyendo:

* Eliminación de objetos indeseados (vehículos, personas, árboles, etc.) mediante **YOLO + LaMa inpainting**.
* Detección y recorte de casas usando un modelo entrenado personalizado.
* Generación de metadatos de cada paso (objetos eliminados, máscaras y casas seleccionadas).
* Opción de ejecutar en modo **solo metadata**, sin crear imágenes ni carpetas de salida.

---

## 🚀 Instalación de dependencias

Crea y activa un entorno virtual (opcional pero recomendado) y luego instala las dependencias necesarias:

```bash
# Torch y Ultralytics (YOLO)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics

# Librerías adicionales
pip install pyyaml tqdm numpy==1.26.4 easydict==1.9.0 scikit-image scikit-learn \
            opencv-python tensorflow joblib matplotlib pandas \
            albumentations==0.5.2 hydra-core==1.1.0 \
            pytorch-lightning==1.2.9 tabulate \
            kornia==0.5.0 webdataset packaging \
            wldhx.yadisk-direct timm

# Versión compatible de hydra-core
pip install --upgrade hydra-core==1.3.2
```

---

## 📂 Estructura del proyecto

```bash
facade_model/              # Repositorio clonado con las funciones principales
├── src/facade_model/
│   ├── lama/              # Configuración y modelo de inpainting (LaMa)
│   ├── house_selector.py  # Selección y recorte de casas
│   ├── object_removal.py  # Remoción de objetos con YOLO + LaMa
│   └── ...
pretrained_models/         # Modelos entrenados (YOLO y casa)
results/                   # Resultados generados (máscaras, imágenes limpias, casas recortadas)
```

---

## ⚙️ Pasos del pipeline

### 1. **Clonar y configurar el repositorio**

```bash
git clone https://github.com/wpradab/facade_model.git
cd facade_model
pip install -e . --no-deps --force-reinstall
```

### 2. **Definir rutas de modelos**

```python
model_path = "yolov8x-seg.pt"  # Modelo YOLO para segmentación de objetos
model_house_path = "pretrained_models/best.pt"  # Modelo entrenado de casas

lama_config = "facade_model/src/facade_model/lama/configs/prediction/default.yaml"
lama_ckpt = "pretrained_models/big-lama"
```

### 3. **Cargar imagen de entrada**

```python
image_path = "ruta/a/tu_imagen.jpg"
```

### 3.1 **Extracción de metadata completa (opcional)**

Si quieres obtener **toda la metadata** del pipeline (objetos eliminados + casas detectadas) **sin generar imágenes ni carpetas de salida**, puedes usar:

```python
from facade_model import extract_facade_metadata

# 📂 Imagen de entrada
image_path = "ejemplo.jpg"

# 🚀 Ejecutar función
metadata = extract_facade_metadata(
    image_path=image_path,
    model_path=model_path,
    model_house_path=model_house_path,
    lama_config=lama_config,
    lama_ckpt=lama_ckpt,
    target_labels=target_labels,
    base_output_dir="results",
    house_label="casa",
    metadata_only=True   # ✅ Evita creación de carpetas e imágenes
)

print("✅ Metadata completa extraída:")
print(metadata)
```

---

### 4. **Remoción de objetos**

```python
from facade_model import remove_objects_from_image

target_labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck',
    'backpack', 'umbrella', 'handbag', 'suitcase', 'tree',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'chair', 'dog', 'cat', 'bird', 'couch',
    'potted plant', 'dining table', 'teddy bear'
]

metadata_remove_objects = remove_objects_from_image(
    image_path=image_path,
    model_path=model_path,
    lama_config=lama_config,
    lama_ckpt=lama_ckpt,
    target_labels=target_labels,
    base_output_dir="results",
    metadata_only=False   # ⚡ Si True, solo devuelve metadata
)
print("✅ Elementos eliminados:", metadata_remove_objects)
```

### 5. **Selección de casas**

```python
from facade_model import find_house_in_image

metadata_house = find_house_in_image(
    image_path=image_path,
    model_path=model_house_path,
    house_label="casa",   # Etiqueta usada en el entrenamiento
    results_dir="results",
    metadata_only=False   # ⚡ Si True, solo devuelve metadata
)
print("✅ Casa recortada:", metadata_house)
```

---

## 📊 Resultados esperados

En la carpeta `results/` se generan (cuando `metadata_only=False`):

* Imágenes sin objetos indeseados.
* Máscaras aplicadas por LaMa.
* Casas detectadas y recortadas junto con sus metadatos.

---

## 📝 Notas

* Usa `metadata_only=True` en cualquiera de las funciones para evitar la creación de imágenes y carpetas, obteniendo únicamente la metadata.
* El pipeline funciona en cualquier entorno Python (Linux, Windows o macOS).
* El modelo de casas (`best.pt`) debe estar entrenado previamente y ubicado en la carpeta `pretrained_models/`.
* El parámetro `house_label` debe coincidir con la etiqueta definida durante el entrenamiento.

---
