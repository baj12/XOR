import unittest
from src.genetic_algorithm import run_genetic_algorithm


class TestGeneticAlgorithm(unittest.TestCase):
    def test_run_genetic_algorithm(self):
        population, log, hof = run_genetic_algorithm()
        self.assertIsNotNone(population)
        self.assertIsNotNone(log)
        self.assertIsNotNone(hof)
        self.assertGreater(len(population), 0)
        self.assertGreater(len(hof), 0)


if __name__ == '__main__':
    unittest.main()
