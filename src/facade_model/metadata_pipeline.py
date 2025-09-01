from facade_model import remove_objects_from_image, find_house_in_image

def extract_facade_metadata(
    image_path: str,
    model_path: str,
    model_house_path: str,
    lama_config: str,
    lama_ckpt: str,
    target_labels: list,
    base_output_dir: str = "results",
    house_label: str = "casa"
):
    """
    Extrae metadata de una imagen de fachada:
      1. Detección de objetos a remover (sin guardado de imágenes).
      2. Detección de casas incompletas (sin guardado de imágenes).

    Retorna:
        dict con:
            - metadata_remove_objects
            - metadata_incomplete_house
    """

    # ==============================
    # 1. Ejecutar remoción de objetos (solo metadata)
    # ==============================
    metadata_remove_objects = remove_objects_from_image(
        image_path=image_path,
        model_path=model_path,
        lama_config=lama_config,
        lama_ckpt=lama_ckpt,
        target_labels=target_labels,
        base_output_dir=base_output_dir,
        metadata_only=True
    )

    # ==============================
    # 2. Selección de casas (solo metadata)
    # ==============================
    metadata_incomplete_house = find_house_in_image(
        image_path=image_path,
        model_path=model_house_path,
        house_label=house_label,
        results_dir=base_output_dir,
        metadata_only=True
    )

    # ==============================
    # 3. Retornar metadata combinada
    # ==============================
    return {
        "metadata_remove_objects": metadata_remove_objects,
        "metadata_incomplete_house": metadata_incomplete_house
    }
