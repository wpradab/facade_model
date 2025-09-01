"""
Facade Model Library
Author: William Prada
Description:
    Tools to process facade images:
    - Remove objects using YOLO + LaMa inpainting.
    - Detect and crop the main facade of a house.
"""

from .house_selector import find_house_in_image
from .object_removal import remove_objects_from_image
from .metadata_pipeline import extract_facade_metadata

__all__ = [
    "find_house_in_image",
    "remove_objects_from_image",
    "extract_facade_metadata"
]
