"""Cifar100 pre-processor."""
from interfaces.pre_proc import PreProcessing


class Cifar100PreProc(PreProcessing):
    """Preprocessing pipeline for cifar100."""

    def __call__(self, x_train, x_eval):
        """Apply preprocessing operations to input."""
        return x_train / 255.0, x_eval / 255.0
