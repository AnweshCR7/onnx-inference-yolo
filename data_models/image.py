"""Image data class"""
from typing import Optional, List, Tuple, Union, Any
from uuid import UUID

import cv2
import numpy as np
from pydantic import BaseModel

from data_models.detected_objects import DetectedObject


class Image(BaseModel):
    """Image object"""
    id: Union[str, UUID]
    np_data: np.ndarray
    plot_data: Optional[Any]
    ratio: Optional[float]
    dwdh: Optional[Tuple[float]]
    objects: Optional[List[DetectedObject]]

    def letterbox(self, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32) -> np.ndarray:
        """Resize image by applying letterbox.

        Args:
            new_shape: New dimension of images
            color: Letterbox background color
            auto: Minimum rectangle
            scaleup: Only scale down, do not scale up (for better val mAP)
            stride: Stride for scaleup
        """

        shape = self.np_data.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        im = self.np_data
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(self.np_data, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        resized_img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                         value=color)  # add border
        self.ratio = r
        self.dwdh = (dw, dh)

        return resized_img

    class Config:
        """Poetry config settings."""
        arbitrary_types_allowed = True
