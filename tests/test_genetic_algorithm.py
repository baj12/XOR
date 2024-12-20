import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from deap import tools

from genetic_algorithm import GeneticAlgorithm, eval_individual
from utils import Config, GAConfig, ModelConfig


class TestGeneticAlgorithm(unittest.TestCase):

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
        self.X_train = np.random.rand(100, 2)
        self.X_val = np.random.rand(20, 2)
        self.y_train = np.random.randint(0, 2, 100)
        self.y_val = np.random.randint(0, 2, 20)

    @patch('genetic_algorithm.build_model')
    def test_calculate_total_weights(self, mock_build_model):
        mock_model = MagicMock()
        mock_model.layers = [
            MagicMock(get_weights=lambda: [np.random.rand(10, 10), np.random.rand(10)])]
        mock_build_model.return_value = mock_model

        ga = GeneticAlgorithm(self.config, self.X_train,
                              self.X_val, self.y_train, self.y_val)
        total_weights = ga.calculate_total_weights()
        self.assertEqual(total_weights, 110)

    @patch('genetic_algorithm.build_model')
    @patch('genetic_algorithm.algorithms.eaSimple')
    def test_run(self, mock_eaSimple, mock_build_model):
        mock_model = MagicMock()
        mock_model.layers = [
            MagicMock(get_weights=lambda: [np.random.rand(10, 10), np.random.rand(10)])]
        mock_build_model.return_value = mock_model
        mock_eaSimple.return_value = ([], tools.Logbook())

        ga = GeneticAlgorithm(self.config, self.X_train,
                              self.X_val, self.y_train, self.y_val)
        pop, log = ga.run()

        mock_build_model.assert_called_once()
        mock_eaSimple.assert_called_once()
        self.assertIsInstance(pop, list)
        self.assertIsInstance(log, tools.Logbook)

    @patch('genetic_algorithm.build_model')
    @patch('genetic_algorithm.get_optimizer')
    def test_eval_individual(self, mock_get_optimizer, mock_build_model):
        mock_model = MagicMock()
        mock_model.evaluate.return_value = (0.5, 0.8)
        mock_build_model.return_value = mock_model

        individual = np.random.rand(110).tolist()
        fitness = eval_individual(
            individual, self.config, self.X_train, self.X_val, self.y_train, self.y_val)
        self.assertEqual(fitness, (0.8,))


if __name__ == '__main__':
    unittest.main()
