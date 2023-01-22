from typing import Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
from loguru import logger


class OnnxBase:
    """ONNX Base class."""
    weight_path: str = None
    session: ort.InferenceSession = None

    input_size: Tuple[int, int] = None
    input_name: str = None
    output_name: str = None

    def __init__(self, weight_path: str) -> None:
        """Initialize class.

        Args:
            weight_path: Location of the weight file
        """
        self.weight_path: str = weight_path

        if weight_path.__contains__("onnx"):
            logger.info(f"Loading ONNX model from: {weight_path}")
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else [
                'CPUExecutionProvider']
            self.session = ort.InferenceSession(self.weight_path, providers=providers)

            # Model settings
            img_size_h = self.session.get_inputs()[0].shape[2]
            img_size_w = self.session.get_inputs()[0].shape[3]
            self.input_size = (img_size_w, img_size_h)
            self.input_name: str = self.session.get_inputs()[0].name
            self.output_name: str = self.session.get_outputs()[0].name
        else:
            logger.error(f"This type of model is not supported!")

    def predict(self, input_data: np.ndarray) -> Union[np.ndarray, None]:
        """OCR model predict code, independent of weight format.

        Args:
            input_data: Input data

        Returns:
            Resulting predictions
        """
        if isinstance(self.session, ort.InferenceSession):
            return self.onnx_inference(session=self.session, input_data=input_data, output_name=self.output_name,
                                       input_name=self.input_name)
        else:
            logger.error(f"Predict for type {type(self.session)} is not supported!")
            return None

    @staticmethod
    def onnx_inference(session: ort.InferenceSession, input_data: np.ndarray,
                       output_name: str = 'output', input_name: str = 'images') -> np.ndarray:
        """General onnx inference session.

        Args:
            session: ONNX inference Session
            input_data: Input data, batched images
            output_name: ONNX output name
            input_name: ONNX input name

        Returns:
            Array of predictions
        """
        return session.run(
            None, {
                input_name: input_data
            }
        )
