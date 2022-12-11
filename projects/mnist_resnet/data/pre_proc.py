"""Mnist pre-processor."""
import tensorflow as tf

from interfaces.pre_proc import PreProcessing


class MnistPreProc(PreProcessing):
    """Preprocessing pipeline for mnist."""

    def __call__(self, x_train, x_eval):
        """Apply preprocessing operations to input."""
        x_train, x_eval = x_train / 255.0, x_eval / 255.0

        # append c (as in nhwc) to the end as a new axis
        x_train = x_train[..., tf.newaxis].astype("float32")
        x_eval = x_eval[..., tf.newaxis].astype("float32")

        return x_train, x_eval
