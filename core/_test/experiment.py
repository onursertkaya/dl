"""Tests for Experiment class."""
import unittest
from unittest.mock import Mock, patch

from core.experiment import Experiment


class TestExperiment(unittest.TestCase):
    """Tests for Experiment class."""

    # pylint: disable=too-many-arguments
    @patch("core.experiment.make_logger")
    @patch("core.experiment.Summary")
    @patch("core.experiment.ExperimentSettings")
    @patch("core.experiment.TaskGroup")
    @patch("core.experiment.Loader")
    @patch("core.experiment.AssembledModel")
    @patch("core.experiment.Experiment._perform_post_epoch_ops")
    @patch("core.experiment.Experiment._evaluate_one_epoch")
    @patch("core.experiment.Experiment._train_one_epoch")
    @patch("core.experiment.Experiment._setup_env")
    def test_create_and_start(
        self,
        mock_setup_env: Mock,
        mock_train_one_epoch: Mock,
        mock_evaluate_one_epoch: Mock,
        mock_post_epoch_ops: Mock,
        mock_model: Mock,
        mock_loader: Mock,
        mock_taskgroup: Mock,
        mock_settings: Mock,
        mock_summary: Mock,
        mock_make_logger: Mock,
    ):
        """Test an experiment end-to-end."""
        mock_settings.return_value.restore_from = None
        mock_settings.return_value.epochs = 1
        mock_settings.return_value.debug = False
        mock_taskgroup.return_value.training = None
        exp = Experiment(
            mock_settings(),
            mock_taskgroup(),
            mock_loader(),
            mock_model(),
        )
        mock_setup_env.assert_called_once_with(False)
        mock_summary.assert_called_once()
        mock_make_logger.assert_called_once()

        exp.start()
        mock_train_one_epoch.assert_called_once()
        mock_evaluate_one_epoch.assert_called_once()
        mock_post_epoch_ops.assert_called_once()
