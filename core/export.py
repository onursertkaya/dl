from pathlib import Path
from typing import Union

import tf2onnx


class Export:
    def __init__(self, onnx_path: Union[Path, str], model):
        self._model = model
        self._onnx_path = onnx_path

    def create_onnx(self):
        model_proto, _ = tf2onnx.convert.from_keras(
            self._model,
            input_signature=(self._model.input_signature,),
            opset=13,
            output_path=self._onnx_path,
        )
