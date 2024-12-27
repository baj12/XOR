# genetic_algorithm.py

# because of some bugs in M2/M3 apple silicon and metal
# i had to play with paralleization. in the end multiprocessing with pools

import concurrent.futures
import gc
import json
import logging
import multiprocessing as mp
import os
import random
import signal
import sys
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
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

# this disables GPU
# tf.config.set_visible_devices([], 'GPU')


def handler(signum, frame):
    location = f"File \"{frame.f_code.co_filename}\", line {frame.f_lineno}, in {frame.f_code.co_name}"
    tb = ''.join(traceback.format_stack(frame))
    error_message = f"GA execution timed out!\nLocation: {location}\nStack Trace:\n{tb}"
    raise TimeoutError(error_message)


signal.signal(signal.SIGALRM, handler)
# signal.alarm(3600)


logger = logging.getLogger(__name__)

# Initialize a global counter
fitness_counter = 0


# @contextmanager
# def managed_pool(processes):
#     """Context manager for proper pool cleanup"""
#     pool = mp.Pool(processes=processes, initializer=init_worker_logging)
#     try:
#         yield pool
#     finally:
#         pool.close()
#         pool.join()


@contextmanager
def managed_pool(max_workers):
    executor = ProcessPoolExecutor(
        max_workers=max_workers, initializer=init_worker_logging)
    try:
        yield executor
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


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
        self.counters = defaultdict(int)

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

    def record_fitness(self, population, generation, filename_base='fitness', batch=10):
        logger = logging.getLogger()
        logger.debug(f"Recording fitness started {generation}.")

        pid = os.getpid()
        filename = f"{filename_base}_{pid}.log"

        # Increment the counter for the current PID
        self.counters[pid] += 1
        current_count = self.counters[pid]
        logger.debug(
            f"Process {pid}: record_fitness counter = {current_count}.")

        generation_fitness = [ind.fitness.values[0] for ind in population]
        self.fitness_history.append({
            'generation': generation,
            'fitness': generation_fitness
        })
        logger.info(
            f"size of self.fitness_history: {asizeof.asizeof(self.fitness_history)} bytes")
        # Store fitness history to file in batches
        if self.counters[pid] % batch == 0:
            try:
                log_directory = '/Users/bernd/python/XOR/logs/'
                os.makedirs(log_directory, exist_ok=True)
                filepath = os.path.join(log_directory, filename)

                with open(filename, 'a') as f:
                    f.write(f"{self.fitness_history}\n")
                logging.debug(
                    f"Process {pid}: Appended fitness: {self.fitness_history} to {filename}")
                # Clear the fitness_history after saving to file
                self.fitness_history.clear()
                logging.debug(
                    f"Process {pid}: Cleared fitness_history after saving.")

            except Exception as e:
                logging.error(
                    f"Process {pid}: Failed to write to {filename}: {e}")

    def run(self):
        """
        Execute the Genetic Algorithm.

        Returns:
        - pop (list): Final population.
        - log (Logbook): Logbook containing statistics of the evolution.
        """
       # Use functools.partial to pass necessary data to eval_individual
        # eval_func = partial(eval_individual, config=self.config,
        #                     X_train=self.X_train, X_val=self.X_val,
        #                     y_train=self.y_train, y_val=self.y_val)
        # self.toolbox.register("evaluate", eval_func)

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
        timeout = self.config.ga.max_time_per_ind
        pid = os.getpid()
        with managed_pool(max_workers=self.config.ga.n_processes) as executor:
            logger.debug(
                f"Starting Genetic Algorithm execution. Process ID: {pid}")

            # Initialize population
            pop = self.toolbox.population(n=self.config.ga.population_size)

            logger.debug(f"pop created.")
            # Evaluate the entire population
            futures = {executor.submit(eval_individual, ind, self.config,
                                       self.X_train, self.X_val,
                                       self.y_train, self.y_val): ind for ind in pop}
            logger.debug(f"futures created.")

            fitnesses = []
            count = 0
            killed = 0

            # Iterate over futures as they complete
            try:
                for future in as_completed(futures, timeout=timeout+1.0):
                    # Retrieve the individual associated with the future
                    individual = futures[future]

                    count += 1
                    logger.debug(f"future. {count}")
                    f = future.result()
                    logger.debug(f"future results {count}: {f}")
                    individual.fitness.values = f
            except TimeoutError:
                killed += 1
                logger.error(
                    f"Evaluation timed out.{pid}. {count}. {killed}")
            except Exception as e:
                logger.error(
                    f"Error during evaluation for individual {ind}: {e}")
            finally:
                for future in futures:
                    if not future.done():
                        individual = futures[future]
                        individual.fitness.values = (0.0, )
                        future.cancel()
                        logger.warning(
                            f"Total evaluations timed out and were cancelled: {killed}")

            # Optional: Update Hall of Fame if needed
            hof.update(pop)

            # Log statistics
            record = stats.compile(pop)
            master_logbook.record(gen=0, **record)

            logger.debug(
                f"size of hof: {asizeof.asizeof(hof)} bytes")
            logger.debug(
                f"size of self.toolbox: {asizeof.asizeof(self.toolbox)} bytes")

            for gen in range(1, self.config.ga.ngen + 1):
                logger.debug(f"Generation {gen} started.")

                # Select the next generation individuals
                offspring = self.toolbox.select(pop, len(pop))
                # Clone the selected individuals
                offspring = list(map(self.toolbox.clone, offspring))

                # Apply crossover and mutation on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.config.ga.cxpb:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in offspring:
                    if random.random() < self.config.ga.mutpb:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [
                    ind for ind in offspring if not ind.fitness.valid]

                # Evaluate the entire population
                futures = {executor.submit(eval_individual, ind, self.config,
                                           self.X_train, self.X_val,
                                           self.y_train, self.y_val): ind for ind in invalid_ind}
                count = 0
                killed = 0
                try:
                    for future in as_completed(futures, timeout=timeout+1):
                        count += 1
                        individual = futures[future]
                        f = future.result(timeout=timeout)
                        individual.fitness.values = f
                        logger.debug(f"Evaluation {pid}. {count} {f}")

                except TimeoutError:
                    killed += 1
                    logger.error(
                        f"Evaluation timed out.{pid}. {count}. {killed}")
                except Exception as e:
                    logger.error(
                        f"Error during evaluation for individual {ind}: {e}")
                finally:
                    for future in futures:
                        if not future.done():
                            individual = futures[future]
                            individual.fitness.values = (0.0, )
                            future.cancel()
                            logger.warning(
                                f"Total evaluations timed out and were cancelled: {killed}")

                # Replace population with offspring
                pop[:] = offspring
                del offspring

                # Update Hall of Fame
                hof.update(pop)

                # Compile and record statistics
                record = stats.compile(pop)
                master_logbook.record(gen=gen, **record)

                gc.collect()

            logger.debug(f"self.record_fitness {gen} starting.")
            self.record_fitness(pop, gen)
            for record in master_logbook:
                record['gen'] = gen
            master_logbook.extend(master_logbook)
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

        # Compile the model to reset optimizer state
        optimizer = get_optimizer(config.model.optimizer, config.model.lr)
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy', metrics=['accuracy'])
        logger.debug(
            f"{pid} Model compiled successfully after setting weights.")

        # Train the model
        model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=config.model.batch_size,
            validation_data=(X_val, y_val),
            verbose=verbose
        )

        # logger.debug(
        #     f"size of history: {asizeof.asizeof(history)} bytes")
        logger.debug(f"{pid} Model training completed.")
        filepath = "results"
        os.makedirs(filepath, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4()

        full_filepath = os.path.join(
            filepath, f"ga_results_{timestamp}_{unique_id}.keras")

        model.save(full_filepath)
        logger.debug(f"{pid} Trained model saved to {full_filepath}.")

        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        logger.debug(f"{pid} Validation Accuracy: {val_accuracy}")
        tf.keras.backend.clear_session()
        del model
        gc.collect()

        return (val_accuracy,)

    except Exception as e:
        logger.error(
            f"{pid} Error during individual evaluation: {e}", exc_info=True)
        return (0.0,)
