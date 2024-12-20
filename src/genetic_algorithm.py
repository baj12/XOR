# genetic_algorithm.py

import logging
import multiprocessing as mp
import random
from contextlib import contextmanager
from functools import partial

import matplotlib.pyplot as plt
import numpy as np  # Added for handling weights
import pandas as pd
from deap import algorithms, base, creator, tools
# from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential  # Added for model manipulation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from model import build_model, get_optimizer
from utils import Config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@contextmanager
def managed_pool(processes):
    """Context manager for proper pool cleanup"""
    pool = mp.Pool(processes=processes)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()


class GeneticAlgorithm:
    def __init__(self, config: Config, X_train, X_val, y_train, y_val):
        """
        Initialize the Genetic Algorithm with configuration and data.

        Parameters:
        - config (Config): Configuration object containing GA and model parameters.
        - X_train (np.ndarray): Training features.
        - X_val (np.ndarray): Validation features.
        - y_train (np.ndarray): Training labels.
        - y_val (np.ndarray): Validation labels.
        """
        self.config = config
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.model = build_model(config)
        self.total_weights = self.calculate_total_weights()
        self.setup_deap()

    def calculate_total_weights(self) -> int:
        """
        Calculate the total number of weights based on the model architecture.

        Returns:
        - total_weights (int): Total number of weights in the model.
        """
        model = self.model
        total_weights = 0
        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                for w in weights:
                    total_weights += w.size
        logger.debug(f"Total weights calculated: {total_weights}")
        return total_weights

    def setup_deap(self):
        """
        Set up the DEAP framework for the Genetic Algorithm.
        """
        # Prevent redefining creator classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness,
                           weights=(1.0,))  # Maximize accuracy
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, -1.0, 1.0)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_float, n=self.total_weights)
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)

        # Register genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian,
                              mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def run(self):
        """
        Execute the Genetic Algorithm.

        Returns:
        - pop (list): Final population.
        - log (Logbook): Logbook containing statistics of the evolution.
        """
       # Use functools.partial to pass necessary data to eval_individual
        eval_func = partial(eval_individual, config=self.config,
                            X_train=self.X_train, X_val=self.X_val,
                            y_train=self.y_train, y_val=self.y_val)
        self.toolbox.register("evaluate", eval_func)

        pop = self.toolbox.population(n=self.config.ga.population_size)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        with managed_pool(processes=self.config.ga.n_processes) as pool:
            self.toolbox.register("map", pool.map)
            logger.info("Starting Genetic Algorithm execution.")
            pop, log = algorithms.eaSimple(
                population=pop,
                toolbox=self.toolbox,
                cxpb=self.config.ga.cxpb,
                mutpb=self.config.ga.mutpb,
                ngen=self.config.ga.ngen,
                stats=stats,
                halloffame=hof,
                verbose=True
            )
            logger.info("Genetic Algorithm execution completed.")

        return pop, log


def eval_individual(individual, config: Config, X_train, X_val, y_train, y_val) -> tuple:
    """
    Evaluate an individual's fitness based on validation accuracy.

    Parameters:
    - individual (list): List of weights representing an individual.
    - config (Config): Configuration object containing model parameters.
    - X_train (np.ndarray): Training features.
    - X_val (np.ndarray): Validation features.
    - y_train (np.ndarray): Training labels.
    - y_val (np.ndarray): Validation labels.

    Returns:
    - fitness (tuple): Validation accuracy as a tuple.
    """
    logger.info("Starting evaluation of individual.")
    try:
        # Build the model
        model = build_model(config)

        # Reshape individual to match model weights
        weight_shapes = [
            w.shape for layer in model.layers for w in layer.get_weights() if w.size > 0]
        weight_tuples = []
        idx = 0
        for shape in weight_shapes:
            size = np.prod(shape)
            weights = np.array(individual[idx:idx+size]).reshape(shape)
            weight_tuples.append(weights)
            idx += size
        model.set_weights(weight_tuples)
        logger.debug("Initial weights set successfully.")

        # Compile the model to reset optimizer state
        optimizer = get_optimizer(config.model.optimizer, config.model.lr)
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy', metrics=['accuracy'])
        logger.debug("Model compiled successfully after setting weights.")

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=config.model.batch_size,
            validation_data=(X_val, y_val),
            verbose=0
        )
        logger.debug("Model training completed.")

        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        logger.debug(f"Validation Accuracy: {val_accuracy}")

        return (val_accuracy,)

    except Exception as e:
        logger.error(f"Error during individual evaluation: {e}", exc_info=True)
        return (0.0,)
