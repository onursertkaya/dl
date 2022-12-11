"""Preprocessing for neural network input."""
import abc


class PreProcessing(abc.ABC):
    """Preprocessing for neural network input."""

    def __init__(self):
        """Create a preprocessor."""

    @abc.abstractmethod
    def __call__(self, x_train, x_eval):
        """Apply the preprocessing operations to the input."""
