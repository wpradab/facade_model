import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from facade_model.utils import save_array_to_img
from facade_model.lama_inpaint import inpaint_img_with_lama


def load_yolo_model(model_path: str):
    """Cargar el modelo YOLO."""
    return YOLO(model_path)


def generate_mask(image_rgb, results, target_labels, output_dir, metadata_only=False):
    """
    Genera máscaras binarias para las clases de interés y devuelve la máscara final y metadata.
    Si metadata_only=True, no guarda imágenes en disco.
    """
    height, width = image_rgb.shape[:2]
    final_mask = np.zeros((height, width), dtype=np.uint8)

    if results.masks is None:
        return final_mask, [], []

    masks = results.masks.data.cpu().numpy()
    clases = results.boxes.cls.cpu().numpy()
    names = results.names
    confs = results.boxes.conf.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]

    saved_masks = []
    detected_objects = []

    for i, (mask, cls_id, conf, box) in enumerate(zip(masks, clases, confs, boxes)):
        class_name = names[int(cls_id)]
        if class_name in target_labels:
            binary_mask = (mask * 255).astype(np.uint8)

            # Aumentar tamaño de la máscara
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

            final_mask = cv2.bitwise_or(final_mask, binary_mask)

            # ✅ Solo guardar si metadata_only=False
            mask_filename = None
            if not metadata_only:
                mask_filename = os.path.join(output_dir, f"mask_{i}_{class_name}.png")
                cv2.imwrite(mask_filename, binary_mask)
                saved_masks.append(mask_filename)

            detected_objects.append({
                "id": i,
                "class": class_name,
                "confidence": float(conf),
                "bbox": [float(x) for x in box],  # [x1, y1, x2, y2]
                "mask_path": mask_filename  # será None si metadata_only=True
            })

    return final_mask, saved_masks, detected_objects



def remove_objects_from_image(
    image_path: str,
    model_path: str,
    lama_config: str,
    lama_ckpt: str,
    target_labels: list,
    base_output_dir: str = "results",
    device: str = None,
    metadata_only: bool = False,  # <<--- NUEVO PARÁMETRO
):
    """
    Detecta objetos en una imagen con YOLO.
    Si metadata_only=True, retorna solo la metadata sin modificar imágenes.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # === PREPARAR SALIDA ===
    image_name = Path(image_path).stem
    output_dir = os.path.join(base_output_dir, image_name)

    if not metadata_only:
        os.makedirs(output_dir, exist_ok=True)

    output_mask_path = os.path.join(output_dir, "mask_final.png")
    output_inpainted_path = os.path.join(output_dir, "inpainted.jpg")

    # === CARGAR IMAGEN ===
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # === CARGAR MODELO YOLO ===
    model = load_yolo_model(model_path)

    # === INFERENCIA ===
    results = model(image_rgb)[0]

    # === GENERAR MÁSCARA Y METADATA ===
    final_mask, _, detected_objects = generate_mask(
        image_rgb, results, target_labels, output_dir, metadata_only=metadata_only
    )

    output_data = {
        "image_name": image_name,
        "input_image": image_path,
        "detected_objects": detected_objects,
        "output_mask": None,
        "output_inpainted": None
    }

    # === SOLO METADATA ===
    if metadata_only:
        return output_data

    # === CONTINUA CON EL INPAINTING SOLO SI HAY MÁSCARA ===
    if np.any(final_mask):
        cv2.imwrite(output_mask_path, final_mask)

        inpainted = inpaint_img_with_lama(image_rgb, final_mask, lama_config, lama_ckpt, device=device)
        save_array_to_img(inpainted, output_inpainted_path)

        output_data["output_mask"] = output_mask_path
        output_data["output_inpainted"] = output_inpainted_path
    else:
        print("No se encontraron objetos a eliminar en la imagen.")

    return output_data

