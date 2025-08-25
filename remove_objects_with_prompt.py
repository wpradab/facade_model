import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from utils import save_array_to_img
from lama_inpaint import inpaint_img_with_lama


# === CONFIGURACIÓN ===
image_path = r"C:\Users\Dell\Documents\IGAC\repositorio\facade_model\Inpaint-Anything\158350100000000-714_2023-09_27527.jpg"
base_output_dir = 'results'
target_labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck',
    'backpack', 'umbrella', 'handbag', 'suitcase', 'tree',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'chair', 'dog', 'cat', 'bird', 'couch', 'potted plant', 'dining table', 'teddy bear', 'vegetacion'
]

# Rutas modelo LaMa
lama_config = r"C:\Users\Dell\Documents\IGAC\repositorio\facade_model\Inpaint-Anything\lama\configs\prediction\default.yaml"
lama_ckpt = r"C:\Users\Dell\Documents\IGAC\repositorio\facade_model\Inpaint-Anything\pretrained_models\big-lama"
device = "cuda" if torch.cuda.is_available() else "cpu"

# === EXTRAER NOMBRE BASE DE LA IMAGEN ===
image_name = Path(image_path).stem  # sin extensión
output_dir = os.path.join(base_output_dir, image_name)
os.makedirs(output_dir, exist_ok=True)

output_mask_path = os.path.join(output_dir, 'mask_final.png')

# === CARGAR IMAGEN ===
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
height, width = image_bgr.shape[:2]

# === CARGAR MODELO YOLO ===
# Cargar modelo YOLOv8 fine-tuned
model_path = r"C:\Users\Dell\Documents\IGAC\repositorio\facade_model\runs\segment\train\weights\best.pt"  # o "last.pt"
model = YOLO(model_path)

# model = YOLO('yolov8x-seg.pt')

# === INFERENCIA ===
results = model(image_rgb)[0]
names = results.names

# === MÁSCARA FINAL VACÍA ===
final_mask = np.zeros((height, width), dtype=np.uint8)

# === PROCESAR MÁSCARAS ===
if results.masks is not None:
    masks = results.masks.data.cpu().numpy()  # (N, H, W)
    clases = results.boxes.cls.cpu().numpy()

    for i, (mask, cls_id) in enumerate(zip(masks, clases)):
        class_name = names[int(cls_id)]

        if class_name in target_labels:
            binary_mask = (mask * 255).astype(np.uint8)

            # Aumentar el tamaño de la máscara manteniendo la forma
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))  # Puedes probar (7,7) o (9,9)
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)  # Puedes probar con 2 si quieres más expansión

            # Sumar la máscara al resultado final
            final_mask = cv2.bitwise_or(final_mask, binary_mask)

            # Guardar máscara individual
            mask_filename = os.path.join(output_dir, f"mask_{i}_{class_name}.png")
            cv2.imwrite(mask_filename, binary_mask)
            print(f"✅ Guardada individual: {mask_filename}")

    # === GUARDAR MÁSCARA FINAL ===
    cv2.imwrite(output_mask_path, final_mask)
    print(f"\n✅✅ Máscara final guardada en: {output_mask_path}")

    # === INPAINTING CON LAMA ===
    inpainted = inpaint_img_with_lama(image_rgb, final_mask, lama_config, lama_ckpt, device=device)

    # === GUARDAR IMAGEN RESULTADO ===
    out_path = os.path.join(output_dir, "inpainted.jpg")
    save_array_to_img(inpainted, out_path)
    print(f"🖼️ Imagen sin objetos guardada en: {out_path}")

else:
    print("❌ No se encontraron máscaras en la imagen.")

