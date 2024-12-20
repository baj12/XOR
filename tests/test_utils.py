import os
import tempfile
import unittest
from unittest.mock import mock_open, patch

import pandas as pd
import yaml

from utils import (Config, GAConfig, ModelConfig, load_config, plot_results,
                   validate_file)


class TestUtils(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.test_dir.cleanup)

    def test_load_config(self):
        config_data = {
            'ga': {
                'population_size': 50,
                'cxpb': 0.5,
                'mutpb': 0.2,
                'ngen': 100,
                'n_processes': 4
            },
            'model': {
                'hl1': 10,
                'hl2': 10,
                'activation': 'relu',
                'optimizer': 'adam',
                'lr': 0.001,
                'batch_size': 16
            }
        }
        config_yaml = yaml.dump(config_data)
        config_path = os.path.join(self.test_dir.name, 'config.yaml')
        with open(config_path, 'w') as f:
            f.write(config_yaml)

        config = load_config(config_path)
        self.assertEqual(config.ga.population_size, 50)
        self.assertEqual(config.model.hl1, 10)

    def test_validate_file(self):
        csv_data = "x,y,label\n1.0,2.0,0\n3.0,4.0,1"
        csv_path = os.path.join(self.test_dir.name, 'test.csv')
        with open(csv_path, 'w') as f:
            f.write(csv_data)

        df = validate_file(csv_path)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ['x', 'y', 'label'])

    @patch("matplotlib.pyplot.show")
    def test_plot_results(self, mock_show):
        from deap import tools
        logbook = tools.Logbook()
        logbook.record(gen=0, avg=0.5, max=1.0)
        logbook.record(gen=1, avg=0.6, max=1.1)

        plot_results(logbook)
        mock_show.assert_called_once()

    def test_validate_file_invalid(self):
        with self.assertRaises(ValueError):
            validate_file("non_existent_file.csv")

        invalid_csv_data = "a,b,c\n1,2,3"
        invalid_csv_path = os.path.join(self.test_dir.name, 'invalid.csv')
        with open(invalid_csv_path, 'w') as f:
            f.write(invalid_csv_data)

        with self.assertRaises(ValueError):
            validate_file(invalid_csv_path)


if __name__ == '__main__':
    unittest.main()
