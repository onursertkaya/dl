"""Tests for objective."""

import unittest

import tensorflow as tf

from interfaces.objective import ObjectiveFormulation, ObjectiveTerm


class TestObjectiveTerm(unittest.TestCase):
    """Test ObjectiveTerm."""

    def test_creation(self):
        """Test construction of instances."""
        objective_term = ObjectiveTerm.build(
            "dummy_mae",
            loss_func=tf.keras.losses.MeanAbsoluteError,
            weight=1.2,
        )
        self.assertIsInstance(objective_term, ObjectiveTerm)
        self.assertIsInstance(objective_term.loss_metric, tf.keras.metrics.Mean)
        self.assertEqual(objective_term.loss_metric.name, "dummy_mae")
        self.assertAlmostEqual(objective_term.weight, 1.2)
        self.assertEqual(objective_term.loss_func, tf.keras.losses.MeanAbsoluteError)


class TestLossFormulation(unittest.TestCase):
    """Test mnist objective implementation."""

    def test_single_loss_term(self):
        """Test the case of single loss term in the formulation."""
        objective = ObjectiveFormulation(
            (
                ObjectiveTerm.build(
                    name="",
                    loss_func=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                    weight=0.6,
                ),
            ),
            cumulative_loss_metric=tf.keras.metrics.Mean("total_loss"),
        )
        loss_val = objective.calculate([0, 1, 0, 0], [0.1, 0.7, 0.1, 0.1])

        self.assertAlmostEqual(loss_val.numpy(), 0.168189 * 0.6)
        self.assertAlmostEqual(
            objective.terms[0].loss_metric.result().numpy(), 0.168189
        )
        self.assertAlmostEqual(
            objective.cumulative_loss_metric.result().numpy(), 0.168189 * 0.6
        )

    def test_multi_loss_term(self):
        """Test the case of multiple loss terms in the formulation."""
        objective = ObjectiveFormulation(
            terms=(
                ObjectiveTerm.build(
                    name="",
                    loss_func=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                    weight=0.6,
                ),
                ObjectiveTerm.build(
                    name="",
                    loss_func=tf.keras.losses.MeanAbsoluteError(),
                    weight=0.4,
                ),
            ),
            cumulative_loss_metric=tf.keras.metrics.Mean(""),
        )

        loss_val = objective.calculate([0, 1, 0, 0], [0.1, 0.7, 0.1, 0.1]).numpy()
        # BCE = 0.168189
        # MAE = ( 3 * |0.1 - 0.0| + | 0.7 - 1.0 | ) / 4
        # MAE = 0.15

        # 0.6 * BCE + 0.4 * MAE = 0.1609134

        self.assertAlmostEqual(loss_val, 0.1609134)
