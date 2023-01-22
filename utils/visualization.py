import os
from typing import Tuple, List, Union
from uuid import UUID

import cv2
import numpy as np


def add_bbox(img: np.ndarray, xyxy: np.ndarray, name: str, color: Tuple[int, int, int], conf: float) -> np.ndarray:
    """Add bounding box with label and confidence to given image.

    Args:
        img: Images to be annotated
        xyxy: Bouding box coordinates in xyxy format
        name: Name of the class
        color: Color corresponding to the prediction
        conf: Confidence of the prediction

    Returns:
        Annotated image
    """
    img = cv2.rectangle(img, xyxy[:2], xyxy[2:], color=color, thickness=2)
    return cv2.putText(img, f"{name} {conf}", (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75,
                       color=color, thickness=2)


def write_to_disk(path: str, images: List[np.ndarray], names: List[Union[str, UUID]],
                         extension: str = ".jpg") -> None:
    """Write images to disk.

    Args:
        path: path where the images will be stored
        images: list of image objects to be written to disk
        names: list of image file names (to be written to disk)
        extension: extension of image files
    """
    # Check if path exist
    if not os.path.exists(path):
        os.makedirs(path)

    for idx, image in enumerate(images):
        filename = str(names[idx])
        filename = filename if filename.__contains__(".") else filename + extension
        cv2.imwrite(os.path.join(path, filename), image)
