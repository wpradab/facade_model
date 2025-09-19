import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def load_yolo_model(model_path: str):
    """Carga un modelo YOLOv8."""
    return YOLO(model_path)


def get_mask_area(mask: np.ndarray) -> int:
    return int(np.sum(mask > 0))


def get_mask_center(mask: np.ndarray):
    M = cv2.moments(mask.astype(np.uint8))
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def mask_touches_border(mask: np.ndarray):
    """
    Revisa qué bordes de la máscara tocan la imagen.
    Devuelve una lista con los bordes: ["left", "right", "top", "bottom"].
    """
    h, w = mask.shape
    borders = []
    if np.any(mask[:, 0:20]):   # Izquierda
        borders.append("left")
    if np.any(mask[:, -1:-20]):  # Derecha
        borders.append("right")
    if np.any(mask[0:20, :]):   # Arriba
        borders.append("top")
    if np.any(mask[-1:-20, :]):  # Abajo
        borders.append("bottom")
    return borders



def find_house_in_image(
    image_path: str,
    model_house_path: str,
    target_labels: list = ["casa", "poste"],  # <<< ahora acepta varias etiquetas
    results_dir: str = "results",
    metadata_only: bool = False,
):
    """
    Segmenta objetos (casas y postes) y devuelve un diccionario con metadata.
    Si metadata_only=True, retorna solo la metadata sin guardar imágenes.
    """
    os.makedirs(results_dir, exist_ok=True)

    # === Cargar imagen ===
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    # === Cargar modelo ===
    model = load_yolo_model(model_house_path)

    # === Inferencia ===
    results = model(image_rgb)[0]
    names = results.names

    if results.masks is None:
        return {
            "image": image_path,
            "objects_count": {label: 0 for label in target_labels},
            "selected_objects": {label: None for label in target_labels},
            "message": "No se encontraron máscaras en la imagen."
        }

    masks = results.masks.data.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy()

    # Diccionarios para almacenar resultados por clase
    objects_by_class = {label: [] for label in target_labels}

    for i, (mask, cls_id, box) in enumerate(zip(masks, classes, boxes)):
        class_name = names[int(cls_id)]
        if class_name in target_labels:
            binary_mask = (mask * 255).astype(np.uint8)
            area = get_mask_area(binary_mask)
            center = get_mask_center(binary_mask)
            borders = mask_touches_border(binary_mask)

            objects_by_class[class_name].append({
                "id": i,
                "area": area,
                "center": center,
                "box": [float(x) for x in box],
                "touch_border": borders,
                "incomplete": len(borders) > 0
            })

    # Selección de objeto por clase
    img_center = (width // 2, height // 2)
    selected_objects = {}

    for label, objs in objects_by_class.items():
        if not objs:
            selected_objects[label] = None
            continue

        selected = None
        for obj in objs:
            if obj["center"] and abs(obj["center"][0] - img_center[0]) < width * 0.1:
                selected = obj
                break

        if selected is None:
            selected = max(objs, key=lambda x: x["area"])

        selected_objects[label] = selected

        # === Guardar crops y máscaras SOLO si no es metadata ===
        if not metadata_only and selected:
            x1, y1, x2, y2 = map(int, selected["box"])
            cropped_obj = image_bgr[y1:y2, x1:x2]

            image_name = Path(image_path).stem
            cropped_path = os.path.join(results_dir, f"{image_name}_{label}.jpg")
            mask_path = os.path.join(results_dir, f"{image_name}_{label}_mask.png")

            cv2.imwrite(cropped_path, cropped_obj)
            cv2.imwrite(mask_path, (masks[selected["id"]] * 255).astype(np.uint8))

            selected_objects[label]["mask_path"] = mask_path
            selected_objects[label]["cropped_path"] = cropped_path

    output = {
        "image": image_path,
        "objects_count": {label: len(objs) for label, objs in objects_by_class.items()},
        "selected_objects": selected_objects
    }

    return output
