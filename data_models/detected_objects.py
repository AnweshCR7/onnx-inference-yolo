from typing import List, Any, Optional, Tuple
import numpy as np

from data_models.bounding_box import BoundingBox

yolo_classnames = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                   'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                   'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
                   'frisbee',
                   'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard',
                   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                   'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                   'cell phone',
                   'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear',
                   'hair drier', 'toothbrush')


class DetectedObject:
    """Class to store License plate character"""

    def __init__(self, cls: int, conf: float, bbox: BoundingBox, classnames: Tuple[str] = yolo_classnames) -> None:
        self.cls: int = cls
        self.object_class: Optional[str] = classnames[cls]
        self.conf: float = conf
        self.bbox: BoundingBox = bbox

    def get_annotation_for_bbox(self):
        """Returns all attributes required for annotation"""
        return self.bbox.get_coordinates(), self.cls, self.conf

    @staticmethod
    def rescale_prediction(xyxy, dwdh, ratio) -> List[Any]:
        """Rescale prediction back to original image size.

        Args:
            image: image whose characters need to be rescaled

        Returns:
            Rescaled image predictions
        """
        xyxy = np.array(xyxy)
        xyxy -= np.array(dwdh * 2)
        xyxy /= ratio

        return list(xyxy)
