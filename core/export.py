"""Deep learning model export."""
from pathlib import Path
from typing import Union

import tf2onnx

from tools.common.log import make_logger


class Export:
    """A deep learning model export."""

    def __init__(self, onnx_path: Union[Path, str], model):
        """Create an exporter."""
        self._model = model
        self._onnx_path = onnx_path

    def create_onnx(self):
        """Convert a model and save to an onnx file."""
        logger = make_logger(__name__)
        logger.info("Exporting...")
        tf2onnx.convert.from_keras(
            self._model,
            input_signature=(self._model.input_signature,),
            opset=13,
            output_path=self._onnx_path,
        )
        logger.info("Done.")
