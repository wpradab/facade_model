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


def border_contact_ratio(mask: np.ndarray) -> float:
    """
    Calcula qué porcentaje del perímetro de la fachada toca el borde de la imagen.
    """
    h, w = mask.shape
    border_pixels = np.zeros_like(mask)

    # Bordes de la imagen
    border_pixels[:, 0] = 1      # izquierda
    border_pixels[:, -1] = 1     # derecha
    border_pixels[0, :] = 1      # arriba
    border_pixels[-1, :] = 1     # abajo

    intersection = np.logical_and(mask > 0, border_pixels > 0)
    intersection_count = np.sum(intersection)

    mask_perimeter = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    perimeter_len = 0
    for cnt in mask_perimeter:
        perimeter_len += cv2.arcLength(cnt, True)

    if perimeter_len == 0:
        return 0.0

    return intersection_count / perimeter_len


def find_house_in_image(image_path: str, model_path: str, house_label: str = "casa", results_dir: str = "results"):
    """
    Segmenta casas y devuelve solo la sección de la casa seleccionada.

    Retorna (cropped_path, mask_path, toca_borde_bool).
    """
    # Preparar carpeta de resultados
    os.makedirs(results_dir, exist_ok=True)

    # Cargar imagen
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    # Cargar modelo
    model = load_yolo_model(model_path)

    # Inferencia
    results = model(image_rgb)[0]
    names = results.names

    if results.masks is None:
        print("No se encontraron máscaras en la imagen.")
        return None, None, False

    masks = results.masks.data.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy()

    houses = []
    for mask, cls_id, box in zip(masks, classes, boxes):
        class_name = names[int(cls_id)]
        if class_name == house_label:
            binary_mask = (mask * 255).astype(np.uint8)
            area = get_mask_area(binary_mask)
            center = get_mask_center(binary_mask)
            ratio = border_contact_ratio(binary_mask)
            touches_border = ratio > 0.25  # condición del 25%
            houses.append({
                "mask": binary_mask,
                "area": area,
                "center": center,
                "box": box,
                "touches_border": touches_border
            })

    if not houses:
        print("No se segmentaron casas en la imagen.")
        return None, None, False

    # Punto central de la imagen
    img_center = (width // 2, height // 2)

    # Buscar casa que contiene el centro
    selected = None
    for h in houses:
        if h["mask"][img_center[1], img_center[0]] > 0:
            selected = h
            break

    # Si no hay casa en el centro → probar 25% más abajo
    if selected is None:
        shifted_center = (width // 2, int(height * 0.75))
        for h in houses:
            if h["mask"][shifted_center[1], shifted_center[0]] > 0:
                selected = h
                break

    # Si tampoco se encuentra → tomar la más grande
    if selected is None:
        selected = max(houses, key=lambda x: x["area"])

    # Extraer bounding box
    x1, y1, x2, y2 = map(int, selected["box"])
    cropped_house = image_bgr[y1:y2, x1:x2]

    # Guardar resultados
    image_name = Path(image_path).stem
    cropped_path = os.path.join(results_dir, f"{image_name}_house.jpg")
    mask_path = os.path.join(results_dir, f"{image_name}_mask.png")

    cv2.imwrite(cropped_path, cropped_house)
    cv2.imwrite(mask_path, selected["mask"])

    if selected["touches_border"]:
        print(f"⚠️ La casa seleccionada en {image_name} toca el borde de la imagen (>25% del perímetro).")

    print(f"Casa seleccionada guardada en: {cropped_path}")
    print(f"Máscara guardada en: {mask_path}")

    return cropped_path, mask_path, selected["touches_border"]
