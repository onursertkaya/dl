import unittest

import tensorflow as tf

from interfaces.metrics import WeightedObjectiveFunction
from projects.mnist_resnet.objective.mnist_metrics import MnistLoss

# TODO: fix the tests


class TestMnistLoss(unittest.TestCase):
    def test_single_loss_term(self):
        loss_fn = MnistLoss(
            [
                WeightedObjectiveFunction(
                    func=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                    metric=tf.keras.Metrics.Mean(""),
                    weight=0.6,
                )
            ]
        )
        loss_val = loss_fn([0, 1, 0, 0], [0.1, 0.7, 0.1, 0.1])

        self.assertAlmostEqual(loss_val, 0.168189)

    def test_multi_loss_term(self):
        loss_fn = MnistLoss(
            [
                WeightedObjectiveFunction(
                    tf.keras.losses.BinaryCrossentropy(from_logits=False), 0.6
                ),
                WeightedObjectiveFunction(tf.keras.losses.MeanAbsoluteError(), 0.4),
            ]
        )

        loss_val = loss_fn([0, 1, 0, 0], [0.1, 0.7, 0.1, 0.1]).numpy()
        # BCE = 0.168189
        # MAE = ( 3 * |0.1 - 0.0| + | 0.7 - 1.0 | ) / 4
        # MAE = 0.15

        # 0.6 * BCE + 0.4 * MAE = 0.1609134

        self.assertAlmostEqual(loss_val, 0.1609134)


if __name__ == "__main__":
    unittest.main()
