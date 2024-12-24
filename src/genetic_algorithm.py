# genetic_algorithm.py

import json
import logging
import multiprocessing as mp
import os  # Add this import
import random
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np  # Added for handling weights
import pandas as pd
import tensorflow as tf
from deap import algorithms, base, creator, tools
from memory_profiler import memory_usage, profile
from pympler import asizeof
from tensorflow.keras.backend import clear_session
# from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential  # Added for model manipulation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from model import build_model, get_optimizer
from utils import Config

# tf.config.threading.set_intra_op_parallelism_threads(2)
# tf.config.threading.set_inter_op_parallelism_threads(2)

logger = logging.getLogger(__name__)


@contextmanager
def managed_pool(processes):
    """Context manager for proper pool cleanup"""
    pool = mp.Pool(processes=processes, initializer=init_worker_logging)
    try:
        yield pool
    finally:
        pool.close()
        pool.join()


def init_worker_logging():
    """
    Initializes logging for each worker process.
    """
    import logging
    import sys

    # Configure logging for the worker
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to capture all debug messages
        format='%(asctime)s [PID %(process)d] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('debug.log', mode='a')
        ]
    )
    # Suppress DEBUG logs from specific third-party libraries if needed
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


@profile
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
        self.pool = mp.Pool(processes=self.config.ga.n_processes)
        self.toolbox.register("map", self.pool.map)
        self.fitness_history = []  # To store fitness of all individuals per generation

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

    def record_fitness(self, population, generation):
        logger.debug(f"Recording fitness for generation {generation}.")
        generation_fitness = [ind.fitness.values[0] for ind in population]
        self.fitness_history.append({
            'generation': generation,
            'fitness': generation_fitness
        })
        logger.info(
            f"size of self.fitness_history: {asizeof.asizeof(self.fitness_history)} bytes")

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

        # Determine verbose level based on logger level
        log_level = logger.getEffectiveLevel()
        if log_level <= logging.DEBUG:
            verbose = False
        elif log_level <= logging.INFO:
            verbose = False
        else:
            verbose = False

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        logger.debug(mp.get_start_method())
        master_logbook = tools.Logbook()
        # Define headers as per your stats
        master_logbook.header = ["gen", "avg", "std", "min", "max"]
        with managed_pool(processes=self.config.ga.n_processes) as pool:
            self.toolbox.register("map", pool.map)
            pid = os.getpid()
            logger.debug(
                f"Starting Genetic Algorithm execution. process {pid}")
            for gen in range(1, self.config.ga.ngen + 1):
                logger.debug(f"Generation {gen} started.")

                pop, logbook = algorithms.eaSimple(
                    population=pop,
                    toolbox=self.toolbox,
                    cxpb=self.config.ga.cxpb,
                    mutpb=self.config.ga.mutpb,
                    ngen=1,
                    stats=stats,
                    halloffame=hof,
                    verbose=verbose
                )
                self.record_fitness(pop, gen)
                for record in logbook:
                    record['gen'] = gen
                master_logbook.extend(logbook)
                logger.debug(
                    f"size of master_logbook: {asizeof.asizeof(master_logbook)} bytes")
                logger.debug(
                    f"Generation {gen} completed process {pid}.")
            logger.debug(
                f"Genetic Algorithm execution completed. process {pid}")
            logger.info(
                f"size of self.fitness_history - 2: {asizeof.asizeof(self.fitness_history)} bytes")

        current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fitness_filename = f'fitness_history_{current_date}.json'
        fitness_filepath = os.path.join("results", fitness_filename)

        # Save fitness history to a JSON file in the results directory
        with open(fitness_filepath, 'w') as f:
            json.dump(self.fitness_history, f)

        # Retrieve the best individual from Hall of Fame
        best_individual = hof[0] if hof else None
        logger.info(
            f"Best Individual: {best_individual}, Fitness: {best_individual.fitness.values if best_individual else 'N/A'}")

        return best_individual, master_logbook


@profile
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
    pid = os.getpid()
    # Determine verbose level based on logger level
    log_level = logger.getEffectiveLevel()
    if log_level <= logging.DEBUG:
        verbose = 0
    elif log_level <= logging.INFO:
        verbose = 0
    else:
        verbose = 0

    logger.info(f"{pid} Starting evaluation of individual.")
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

        logger.debug(f"{pid} Initial weights set successfully.")

        # Compile the model to reset optimizer state
        optimizer = get_optimizer(config.model.optimizer, config.model.lr)
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy', metrics=['accuracy'])
        logger.debug(
            f"{pid} Model compiled successfully after setting weights.")

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=config.model.batch_size,
            validation_data=(X_val, y_val),
            verbose=verbose
        )

        logger.debug(
            f"size of history: {asizeof.asizeof(history)} bytes")
        logger.debug(f"{pid} Model training completed.")
        filepath = "results"
        os.makedirs(filepath, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4()
        full_filepath = os.path.join(
            filepath, f"ga_results_{timestamp}_{unique_id}.keras")

        model.save(full_filepath)
        logger.info(f"{pid} Trained model saved to {full_filepath}.")

        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        logger.debug(f"{pid} Validation Accuracy: {val_accuracy}")
        tf.keras.backend.clear_session()
        del model

        return (val_accuracy,)

    except Exception as e:
        logger.error(
            f"{pid} Error during individual evaluation: {e}", exc_info=True)
        return (0.0,)
