# onnx-inference-yolo
ONNX inference pipeline for YOLO Object Detection Model

Working with ML models there a lot of different frameworks to train and execute models, potential compilers to improve runtime of interences and the story goes on. When it comes to inference runtime optimization (including optimization of some potentially very costly pre-processing) the hardware architectures on which the models are deployed makes a significant difference.

Often there is need for interoperability between these different tools. E.g. when training a model with algorithm A in framework X you are often locked into the eco-system of that framework which might not have the best prediction accuracy, lowest execution runtime or better other “quality attributes” than when training a model with the same algorithm A but a different framework Y. Furthermore, it could be a completely different story for a different algorithm. The reason for this is that the low level implementation of the algorithms differ or that they use a slightly different set of operators. Additionally, the choice of framework could also be based on the development experience with framework Y which might be way better than the one provided by framework X! 

ONNX to the rescue! This repository contains scripts to perform inference on a YOLO-v7 object detection model using just a `.onnx` file. It takes an object oriented approach (pun un-intended) to perform object detection on provided images. Any YOLO model in onnx format can be used for inference. A couple of them are provided below. Use the dynamic batch checkpoint for working with > 1 image per batch.

[`yolov7-tiny.onnx`](https://github.com/AnweshCR7/onnx-inference-yolo/blob/main/weights/yolov7-tiny/yolov7-tiny.onnx)
[`yolov7-tiny-dynamic-batch.onnx`](https://github.com/AnweshCR7/onnx-inference-yolo/blob/main/weights/yolov7-tiny/yolov7-tiny-dynamic-batch.onnx)

TODOs
- [x] Upload a couple of onnx weights
- [x] Add annotation script
- [ ] Fix input/output paths and model path using argparser 
- [ ] Add inference for segmentation 
- [ ] Improve docstrings
