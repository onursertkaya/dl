"""Module for implementing all Resnet Block variants.

This module allows just a bit of code duplication in exchange for
clarity, as understanding the nuances in the papers can be challenging.

Each variant is implemented as a class. Docstrings include the papers
where the block has been introduced.


= Heritage =

ResNet
- information flow to subsequent layers
https://arxiv.org/pdf/1512.03385.pdf


Residual unit
- BN-relu-conv ordering
https://arxiv.org/pdf/1603.05027v3.pdf


Aggregated Residual Transformations for Deep Neural Networks
- "resNeXt"
- group aggregation
https://arxiv.org/pdf/1611.05431.pdf


ResNet strikes back
- re-exploration of training regimes
https://arxiv.org/pdf/2110.00476.pdf


A convnet for 2020s
- modernization with other lines of research
  - "patchify" 4x4@4 conv
  - resnext layers
  - inverted bottlenecks (mbconv)
  - 7x7 depthwise convs
https://arxiv.org/pdf/2201.03545v1.pdf


Scaling up your kernels to 31x31
- replknet
https://arxiv.org/pdf/2203.06717.pdf


More convnets in the 2020s
- 51x51 sparse kernels
https://arxiv.org/pdf/2207.03620v1.pdf

"""
import abc
from typing import final

import tensorflow as tf

DATA_FORMAT = "channels_last"  # NHWC
CHANNEL_AXIS = 3 if DATA_FORMAT == "channels_last" else 1
PADDING_SAME = "SAME"


@final
class InputBlock(tf.keras.layers.Layer):
    """Resnet input block for downsampling.

    https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, num_filters=64, use_batch_norm=True):
        """Initialize layers."""
        super().__init__(name="conv1")
        self.conv = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(
                7,
                7,
            ),
            strides=(
                2,
                2,
            ),
            padding=PADDING_SAME,
            data_format=DATA_FORMAT,
            activation=None,
            use_bias=not use_batch_norm,
            name="conv",
        )

        self.batch_norm = (
            tf.keras.layers.BatchNormalization(name="bn")
            if use_batch_norm
            else lambda x, training: x
        )

    def call(self, input_tensor, training):
        """Forward pass."""
        t = self.conv(input_tensor)
        t = self.batch_norm(t, training=training)
        return tf.nn.relu(t)


@final
class InitialPoolBlock(tf.keras.layers.Layer):
    """Initial pooling block for further downsampling.

    https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(
        self,
    ):
        """Initialize layers."""
        super().__init__(name="initial_maxpool")
        self.maxpool = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding=PADDING_SAME,
            data_format=DATA_FORMAT,
            name="maxpool",
        )

    def call(self, input_tensor, training):
        """Forward pass."""
        return self.maxpool(input_tensor)


class _ResnetBlock(tf.keras.layers.Layer, abc.ABC):
    """Abstract base class for Resnet block variants.

    This class only defines the "trunk", a.k.a. non-residual
    convolutions of the ResNet blocks.

    https://arxiv.org/pdf/1512.03385.pdf

    """

    @abc.abstractmethod
    def __init__(
        self, num_filters: int, use_batch_norm: bool, downsample: bool, name: str
    ):
        """Initialize layers."""
        assert num_filters in [64, 128, 256, 512]
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(
                3,
                3,
            ),
            strides=(
                2,
                2,
            )
            if downsample
            else (1, 1),
            padding=PADDING_SAME,
            data_format=DATA_FORMAT,
            activation=None,
            use_bias=not use_batch_norm,
            name="conv1",
        )

        self.batch_norm1 = (
            tf.keras.layers.BatchNormalization(name="bn1")
            if use_batch_norm
            else lambda x, training: x
        )

        self.conv2 = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(
                3,
                3,
            ),
            strides=(
                1,
                1,
            ),
            padding=PADDING_SAME,
            data_format=DATA_FORMAT,
            activation=None,
            use_bias=not use_batch_norm,
            name="conv2",
        )

        self.batch_norm2 = (
            tf.keras.layers.BatchNormalization(name="bn2")
            if use_batch_norm
            else lambda x, training: x
        )

    def call(self, layer_in, training):
        raise RuntimeError(f"{str(__class__)} can't be used directly.")


@final
class ResnetBlockA(_ResnetBlock):
    """Block type-A.

    https://arxiv.org/pdf/1512.03385.pdf

    If the block is responsible for downsampling, this variant's
    residual connection performs zero-padding on input tensor in channel
    axis + maxpooling(*) in order to match the number of output
    channels.

    * https://github.com/keras-team/keras/issues/2608
    * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/resnet.py#L301


                 |-----------------|                  |--------------------|
    H x W x C -> | channel padding | -> H x W x 2C -> | max_pool @stride 2 | -> H/2 x W/2 x 2C
                 |-----------------|                  |--------------------|

    """

    def __init__(
        self, name: str, num_filters: int, use_batch_norm=True, downsample=False
    ):
        """Initialize layers."""
        super().__init__(
            num_filters=num_filters,
            use_batch_norm=use_batch_norm,
            downsample=downsample,
            name=name,
        )

        self.maybe_skip_concat = lambda x: x
        self.maybe_skip_pool = lambda x: x

        if downsample:
            # TODO: this is probably implemented as
            # tf.zeros_like shows up as cast -> mul -> cast in the onnx
            # file, find a less complex solution.
            self.maybe_skip_concat = lambda x: tf.concat(
                [x, tf.zeros_like(x, dtype=x.dtype)], axis=CHANNEL_AXIS
            )

            self.maybe_skip_pool = tf.keras.layers.MaxPool2D(
                pool_size=(3, 3),
                strides=(2, 2),
                padding=PADDING_SAME,
                data_format=DATA_FORMAT,
                name="maxpool",
            )

    def call(self, layer_in, training):
        """Forward pass."""
        t = self.conv1(layer_in)
        t = self.batch_norm1(t, training=training)
        t = tf.nn.relu(t)
        t = self.conv2(t)
        t = self.batch_norm2(t, training=training)

        skip = self.maybe_skip_concat(layer_in)
        skip = self.maybe_skip_pool(skip)

        t += skip
        return tf.nn.relu(t)


@final
class ResnetBlockB(_ResnetBlock):
    """Block type-B.

    https://arxiv.org/pdf/1512.03385.pdf

    If the block is responsible for downsampling, this variant's
    residual connection performs projection with a 1x1
    (convolution + batchnorm) in order to match the number of output
    channels.

                 |---------------------------------------------|
    H x W x C -> | 2C conv kernels of size 1 x 1 x C @stride 2 | -> H/2 x W/2 x 2C
                 |---------------------------------------------|

    """

    def __init__(
        self, name: str, num_filters: int, use_batch_norm=True, downsample=False
    ):
        """Initialize layers."""
        super().__init__(
            num_filters=num_filters,
            use_batch_norm=use_batch_norm,
            downsample=downsample,
            name=name,
        )

        self.maybe_skip_conv = lambda x: x
        self.maybe_skip_batchnorm = lambda x, training: x

        if downsample:
            self.maybe_skip_conv = tf.keras.layers.Conv2D(
                filters=num_filters,
                kernel_size=(
                    1,
                    1,
                ),
                strides=(
                    2,
                    2,
                ),
                padding=PADDING_SAME,
                data_format=DATA_FORMAT,
                activation=None,
                use_bias=not use_batch_norm,
                name="conv_skip",
            )

            self.maybe_skip_batchnorm = (
                tf.keras.layers.BatchNormalization(name="bn_skip")
                if use_batch_norm
                else lambda x, training: x
            )

    def call(self, layer_in, training):
        """Forward pass."""
        t = self.conv1(layer_in)
        t = self.batch_norm1(t, training=training)
        t = tf.nn.relu(t)
        t = self.conv2(t)
        t = self.batch_norm2(t, training=training)

        skip = self.maybe_skip_conv(layer_in)
        skip = self.maybe_skip_batchnorm(skip, training=training)

        t += skip
        return tf.nn.relu(t)


@final
class ResnetBlockC(_ResnetBlock):
    """Block type-C.

    https://arxiv.org/pdf/1512.03385.pdf

    This variant has 1x1 projection shortcut convolution regardless of
    its downsampling status.

    """

    def __init__(self, name: str, num_filters, use_batch_norm=True, downsample=False):
        """Initialize layers."""
        super().__init__(
            num_filters=num_filters,
            use_batch_norm=use_batch_norm,
            downsample=downsample,
            name=name,
        )

        self.skip_conv = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(
                1,
                1,
            ),
            strides=(
                2,
                2,
            )
            if downsample
            else (1, 1),
            padding=PADDING_SAME,
            data_format=DATA_FORMAT,
            activation=None,
            use_bias=not use_batch_norm,
            name="conv_skip",
        )
        self.maybe_skip_batchnorm = (
            tf.keras.layers.BatchNormalization(name="bn_skip")
            if use_batch_norm
            else lambda x, training: x
        )

    def call(self, layer_in, training):
        """Forward pass."""
        t = self.conv1(layer_in)
        t = self.batch_norm1(t, training=training)
        t = tf.nn.relu(t)
        t = self.conv2(t)
        t = self.batch_norm2(t, training=training)

        skip = self.skip_conv(layer_in)
        skip = self.maybe_skip_batchnorm(skip, training=training)

        t += skip
        return tf.nn.relu(t)


@final
class BottleneckBlock(tf.keras.layers.Layer):
    """Bottleneck block for deeper variants.

    https://arxiv.org/pdf/1512.03385.pdf

    Note that this implementation does not inherit from _ResnetBlock
    as it has a specialized pipeline which is different than the
    A, B, C variants.

    Used in ResNet-{50, 101, 152}."""

    def __init__(
        self, name: str, num_filters: int, use_batch_norm=True, downsample=False
    ):
        """Initialize layers."""
        assert num_filters in [64, 128, 256, 512]
        super().__init__(name=name)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(
                1,
                1,
            ),
            strides=(
                2,
                2,
            )
            if downsample
            else (
                1,
                1,
            ),
            padding=PADDING_SAME,
            data_format=DATA_FORMAT,
            activation=None,
            use_bias=not use_batch_norm,
            name="conv1",
        )

        self.batch_norm1 = (
            tf.keras.layers.BatchNormalization(name="bn1")
            if use_batch_norm
            else lambda x, training: x
        )

        self.conv2 = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(
                3,
                3,
            ),
            strides=(
                1,
                1,
            ),
            padding=PADDING_SAME,
            data_format=DATA_FORMAT,
            activation=None,
            use_bias=not use_batch_norm,
            name="conv2",
        )
        self.batch_norm2 = (
            tf.keras.layers.BatchNormalization(name="bn2")
            if use_batch_norm
            else lambda x, training: x
        )

        self.conv3 = tf.keras.layers.Conv2D(
            filters=num_filters * 4,
            kernel_size=(
                1,
                1,
            ),
            strides=(
                1,
                1,
            ),
            padding=PADDING_SAME,
            data_format=DATA_FORMAT,
            activation=None,
            use_bias=not use_batch_norm,
            name="conv3",
        )

        self.batch_norm3 = (
            tf.keras.layers.BatchNormalization(name="bn3")
            if use_batch_norm
            else lambda x, training: x
        )

        self.maybe_skip_conv = lambda x: x
        self.maybe_skip_batchnorm = lambda x, training: x
        if downsample:
            self.maybe_skip_conv = tf.keras.layers.Conv2D(
                filters=num_filters * 4,
                kernel_size=(
                    1,
                    1,
                ),
                strides=(
                    2,
                    2,
                )
                if downsample
                else (
                    1,
                    1,
                ),
                padding=PADDING_SAME,
                data_format=DATA_FORMAT,
                activation=None,
                use_bias=not use_batch_norm,
                name="conv_skip",
            )

            self.maybe_skip_batchnorm = (
                tf.keras.layers.BatchNormalization(name="bn_skip")
                if use_batch_norm
                else lambda x, training: x
            )

    def call(self, layer_in, training):
        """Forward pass."""
        t = self.conv1(layer_in)
        t = self.batch_norm1(t, training)
        t = tf.nn.relu(t)
        t = self.conv2(t)
        t = self.batch_norm2(t, training)
        t = tf.nn.relu(t)
        t = self.conv3(t)
        t = self.batch_norm3(t, training)

        skip = self.maybe_skip_conv(layer_in)
        skip = self.maybe_skip_batchnorm(skip, training=training)

        t += skip
        return tf.nn.relu(t)


@final
class ResidualUnit(tf.keras.layers.Layer):
    """An enhanced residual block.

    From the paper 'Identity Mappings in Deep Residual Networks'.
    https://arxiv.org/pdf/1603.05027v3.pdf
    """


@final
class ResNext(tf.keras.layers.Layer):
    """Aggregated residual unit.

    From the paper 'Aggregated Residual Transformations for Deep Neural Networks'
    https://arxiv.org/pdf/1611.05431.pdf
    """
