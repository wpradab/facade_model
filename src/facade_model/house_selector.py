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
    if np.any(mask[:, 0]):   # Izquierda
        borders.append("left")
    if np.any(mask[:, -1]):  # Derecha
        borders.append("right")
    if np.any(mask[0, :]):   # Arriba
        borders.append("top")
    if np.any(mask[-1, :]):  # Abajo
        borders.append("bottom")
    return borders



def find_house_in_image(image_path: str, model_path: str, house_label: str = "casa", results_dir: str = "results"):
    """
    Segmenta casas y devuelve un diccionario con información de la casa seleccionada.
    """
    os.makedirs(results_dir, exist_ok=True)

    # === Cargar imagen ===
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    # === Cargar modelo ===
    model = load_yolo_model(model_path)

    # === Inferencia ===
    results = model(image_rgb)[0]
    names = results.names

    if results.masks is None:
        return {
            "image": image_path,
            "houses_count": 0,
            "selected_house": None,
            "message": "No se encontraron máscaras en la imagen."
        }

    masks = results.masks.data.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy()

    houses = []
    for i, (mask, cls_id, box) in enumerate(zip(masks, classes, boxes)):
        class_name = names[int(cls_id)]
        if class_name == house_label:
            binary_mask = (mask * 255).astype(np.uint8)
            area = get_mask_area(binary_mask)
            center = get_mask_center(binary_mask)
            borders = mask_touches_border(binary_mask)
            houses.append({
                "id": i,
                "mask": binary_mask,
                "area": area,
                "center": center,
                "box": box,
                "touch_border": borders
            })

    if not houses:
        return {
            "image": image_path,
            "houses_count": 0,
            "selected_house": None,
            "message": "No se segmentaron casas en la imagen."
        }

    # === Estrategia de selección ===
    img_center = (width // 2, height // 2)
    selected = None

    # 1. Buscar casa que contenga el centro
    for h in houses:
        if h["mask"][img_center[1], img_center[0]] > 0:
            selected = h
            break

    # 2. Si no hay en el centro → intentar 25% más abajo
    if selected is None:
        shifted_center = (width // 2, int(height * 0.75))
        for h in houses:
            if h["mask"][shifted_center[1], shifted_center[0]] > 0:
                selected = h
                break

    # 3. Si no hay ninguna → tomar la de mayor área
    if selected is None:
        selected = max(houses, key=lambda x: x["area"])

    # === Guardar resultados ===
    x1, y1, x2, y2 = map(int, selected["box"])
    cropped_house = image_bgr[y1:y2, x1:x2]

    image_name = Path(image_path).stem
    cropped_path = os.path.join(results_dir, f"{image_name}_house.jpg")
    mask_path = os.path.join(results_dir, f"{image_name}_mask.png")

    cv2.imwrite(cropped_path, cropped_house)
    cv2.imwrite(mask_path, selected["mask"])

    # Diccionario de salida
    output = {
        "image": image_path,
        "houses_count": len(houses),
        "selected_house": {
            "id": selected["id"],
            "area": selected["area"],
            "center": selected["center"],
            "box": [float(x) for x in selected["box"]],
            "touch_border": selected["touch_border"],  # lista de bordes
            "incomplete_house": len(selected["touch_border"]) > 0,  # True/False
            "mask_path": mask_path,
            "cropped_path": cropped_path
        }
    }

    return output
