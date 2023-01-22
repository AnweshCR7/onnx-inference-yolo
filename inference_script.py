from typing import Dict, Tuple

import cv2
from loguru import logger
import random
from data_models.images_input import Images
import numpy as np

from data_models.onnx_object_detection import OnnxObjectDetection
from utils.visualization import write_to_disk

yolo_classnames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']

yolo_colors: Dict[str, Tuple[int, int, int]] = {cls_name: [random.randint(0, 255) for _ in range(3)] for k, cls_name in
                                                enumerate(yolo_classnames)}

yolov7_tiny = "/Users/anwesh.marwade@pon.com/Downloads/yolo/yolov7-tiny-dynamic-batch.onnx"
input_folder = "/Users/anwesh.marwade@pon.com/Downloads/yolo/images"
output_folder = "/Users/anwesh.marwade@pon.com/Downloads/test_op"

if __name__ == '__main__':

    yolov7 = OnnxObjectDetection(weight_path=yolov7_tiny, classnames=yolo_classnames)

    images = Images(images=Images.read_from_folder(path=input_folder, ext="jpg"))

    for i, batch in enumerate(images.create_batch(batch_size=4)):
        logger.info(f"Processing batch: {i} containing {len(batch)} image(s)...")

        raw_out = yolov7.predict_object_detection(input_data=batch.to_onnx_input(image_size=yolov7.input_size))
        batch.init_detected_objects(raw_out)

        annotations = batch.annotate_objects(input_size=yolov7.input_size, letterboxed_image=True, class_colors=yolo_colors)
        write_to_disk(path=output_folder, images=annotations,
                             names=batch.get_image_ids())
