import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential

from model import build_and_train_model, build_model, get_optimizer
from utils import Config, GAConfig, ModelConfig


class TestModel(unittest.TestCase):

    def setUp(self):
        self.config = Config(
            ga=GAConfig(
                population_size=50,
                cxpb=0.5,
                mutpb=0.2,
                ngen=100,
                n_processes=4
            ),
            model=ModelConfig(
                hl1=10,
                hl2=10,
                activation='relu',
                optimizer='adam',
                lr=0.001,
                batch_size=16
            )
        )
        self.df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, 4.0],
            'y': [1.0, 2.0, 3.0, 4.0],
            'label': [0, 1, 0, 1]
        })
        self.initial_weights = np.random.rand(10 + 10 + 1 + 1).tolist()

    @patch('model.build_model')
    @patch('model.Sequential.fit')
    @patch('model.Sequential.save')
    def test_build_and_train_model(self, mock_save, mock_fit, mock_build_model):
        mock_model = MagicMock(spec=Sequential)
        mock_build_model.return_value = mock_model
        mock_fit.return_value = MagicMock(history={'accuracy': [0.8], 'val_accuracy': [
                                          0.75], 'loss': [0.5], 'val_loss': [0.55]})

        build_and_train_model(self.initial_weights, self.df, self.config)

        mock_build_model.assert_called_once()
        mock_fit.assert_called_once()
        mock_save.assert_called_once()

    def test_build_model(self):
        model = build_model(self.config)
        self.assertIsInstance(model, Sequential)
        self.assertEqual(len(model.layers), 3)

    def test_get_optimizer(self):
        optimizer = get_optimizer('adam', 0.001)
        self.assertEqual(optimizer.__class__.__name__, 'Adam')

        optimizer = get_optimizer('sgd', 0.001)
        self.assertEqual(optimizer.__class__.__name__, 'SGD')

        optimizer = get_optimizer('rmsprop', 0.001)
        self.assertEqual(optimizer.__class__.__name__, 'RMSprop')

        optimizer = get_optimizer('unknown', 0.001)
        self.assertEqual(optimizer.__class__.__name__, 'Adam')


if __name__ == '__main__':
    unittest.main()
