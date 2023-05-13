"""Module for implementing various R-CNN variants.

Starting from Fast R-CNN (as it is the first
end-to-end DL-only method)
https://arxiv.org/abs/1504.08083

Continuing with Faster R-CNN
https://arxiv.org/abs/1506.01497

"""


import tensorflow as tf

DATA_FORMAT = "channels_last"  # NHWC
PADDING_SAME = "SAME"


class RoiPooling(tf.keras.layers.Layer):
    """Roi Pooling layer.

    First proposed in Fast R-CNN paper.
    """

    def __init__(self, out_width: int, out_height: int):
        self._out_w = out_width
        self._out_h = out_height

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[1, 10, 10, 1], dtype=tf.float32),  # last feature map
            tf.TensorSpec(shape=[128, 4], dtype=tf.float32),        # 128 sampled rois (or N=2 & 64)
        )
    )
    def call(self, conv_feature_map: tf.Tensor, roi_edges: tf.Tensor):
        n, h, w, c = conv_feature_map.shape
        left = roi_edges[:, 0]
        right = roi_edges[:, 1]
        top = roi_edges[:, 2]
        bottom = roi_edges[:, 3]

        left, right, top, bottom = w * left, w * right, h * top, h * bottom
        left, right, top, bottom = (
            _round_to_int(v) for v in (left, right, top, bottom)
        )
        x = tf.TensorArray(tf.float32, size=128)
        for i in tf.range(2):
            roi_feature_map = conv_feature_map[
                :,
                top[i]:bottom[i],
                left[i]:right[i],
                :
            ]
            roi_h, roi_w  = tf.shape(roi_feature_map)[1], tf.shape(roi_feature_map)[2]  # tensor.shape does not work.
            h_idxs = _get_split_idxs(roi_h, 2)
            v_idxs = _get_split_idxs(roi_w, 2)
            horiz_split_subtensors = tf.split(roi_feature_map, h_idxs, axis=1)
            vert_split_subtensors = [tf.split(hor, v_idxs, axis=2) for hor in horiz_split_subtensors]
            nnn = tf.concat([tf.reduce_max(b, axis=(1,2)) for a in vert_split_subtensors for b in a], axis=0)
            mmm = tf.reshape(nnn, (n, 2, 2, c))
            x.write(i, mmm)
            tf.print(mmm)
        return x

    # def call_eager(self, conv_feature_map_and_roi_edges: List[tf.Tensor]):
    #     """Reference implementation with python-heavy statements.

    #     Runs correctly in eager mode, does not compile into graph.
    #     """
    #     conv_feature_map = conv_feature_map_and_roi_edges[0]
    #     roi_edges = conv_feature_map_and_roi_edges[1]
    #     n, h, w, c = conv_feature_map.shape
    #     cropped_rois = []
    #     for edges in roi_edges:
    #         left, right, top, bottom = edges

    #         left, right, top, bottom = w * left, w * right, h * top, h * bottom
    #         left, right, top, bottom = (
    #             _round_to_int(v) for v in (left, right, top, bottom)
    #         )
    #         roi_feature_map = conv_feature_map[
    #             :,
    #             top:bottom,
    #             left:right,
    #             :,
    #         ]
    #         _, roi_h, roi_w, _ = roi_feature_map.shape
    #         h_idxs = _get_split_idxs(roi_h, self._out_h)
    #         v_idxs = _get_split_idxs(roi_w, self._out_w)
    #         horiz_split_subtensors = tf.split(roi_feature_map, h_idxs, axis=1)
    #         vert_split_subtensors = [tf.split(hor, v_idxs, axis=2) for hor in horiz_split_subtensors]
    #         x = tf.concat([tf.reduce_max(b, axis=(1,2)) for a in vert_split_subtensors for b in a], axis=0)
    #         q = tf.reshape(tf.constant(x), (n, self._out_h, self._out_w, c))
    #         cropped_rois.append(q)
    #         # cropped = [tf.reduce_max(st) for st in vert_split_subtensors]
    #         # print(cropped)
    #         # cropped_rois.append(cropped)

    #     return cropped_rois

@tf.function
def _round_to_int(val):
    return tf.cast(tf.math.round(val), tf.int32)


@tf.function
def _get_split_idxs(total_extent: int, num_splits: int):
    cuts = num_splits - 1
    x = total_extent / num_splits
    xup = tf.cast(tf.math.ceil(x), tf.int32)
    m = xup - 1 if xup*cuts >= total_extent else xup
    r = total_extent - cuts * m
    return ([m] * cuts) + [r]

# class RegionProposalNet(tf.keras.layers.Layer):
#     """Region Proposal Network.

#     First proposed in Faster R-CNN paper, replacing the
#     region proposal algorithms.
#     """

#     def __init__(self, num_features: int):
#         self._conv = tf.keras.layers.Conv2D(
#             filters=num_features,
#             kernel_size=(3, 3),
#             strides=(1, 1),
#             padding=PADDING_SAME,
#             data_format=DATA_FORMAT,
#             activation=None,
#             use_bias=True,
#             name=name,
#         )


class BoxRegression(tf.keras.layers.Layer):
    pass


class BoxClassification(tf.keras.layers.Layer):
    pass
