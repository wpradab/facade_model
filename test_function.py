from object_removal import remove_objects_from_image
from house_selector import find_house_in_image

target_labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck',
    'backpack', 'umbrella', 'handbag', 'suitcase', 'tree',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'chair', 'dog', 'cat', 'bird', 'couch',
    'potted plant', 'dining table', 'teddy bear'
]

image_path = r"C:\Users\Dell\Downloads\ejemplos_fotos_fachadas\fotos_fachadas\fachadas\15\15835\158350100000000530019000-133157_2023-09_25091.jpg"
# model_path = r"C:\Users\Dell\Documents\IGAC\repositorio\facade_model\runs\segment\train\weights\best.pt"
model_path = r"yolov8x-seg.pt"
lama_config = r"C:\Users\Dell\Documents\IGAC\repositorio\facade_model\lama\configs\prediction\default.yaml"
lama_ckpt = r"C:\Users\Dell\Documents\IGAC\repositorio\facade_model\pretrained_models\big-lama"

mask_path, inpainted_path = remove_objects_from_image(
    image_path=image_path,
    model_path=model_path,
    lama_config=lama_config,
    lama_ckpt=lama_ckpt,
    target_labels=target_labels,
    base_output_dir="results"
)

print("Máscara:", mask_path)
print("Imagen final:", inpainted_path)


model_path = r"C:\Users\Dell\Documents\IGAC\classify_and_remove_objects\runs\segment\train\weights\best.pt"

cropped, mask = find_house_in_image(image_path, model_path, house_label="casa", results_dir="results")