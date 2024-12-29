"""
genetic_algorithm.py

because of some bugs in M2/M3 apple silicon and metal which cause at minimum
the macs with M3Pro and tensorflow-metal==1.1.0 to hang at various points.
i had to play with parallelization. in the end concurrent.futures does the trick.
Here we can implement a max wait time for an individual run (individual of the GA)
Of note, we train individual over 10 iterations, which mutation / etc is used?

there is some advanced logging going on that allows logging parallel processes

commented codes allows changing tensorflow number of parallel processes and GPU usage

record fitness was implemented during a search of a memory leak. 
What might be better is a link to a SQL data base
"""

"""
Genetic Algorithm Implementation for [Your Project Name]

This script implements a Genetic Algorithm (GA) to optimize [describe what is being optimized, e.g., neural network architectures, parameters, etc.].
The GA utilizes DEAP (Distributed Evolutionary Algorithms in Python) for evolutionary operations.

Classes:
    GeneticAlgorithm: Encapsulates the GA process, including initialization, evaluation, selection, crossover, and mutation.

Functions:
    evaluate_individual(individual, config, X_train, X_val, y_train, y_val):
        Evaluates the fitness of an individual based on model performance.
"""


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
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential  # Added for model manipulation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from model import build_model, get_optimizer
from utils import Config, get_total_size
tf.config.threading.set_intra_op_parallelism_threads(
    1)  # Set the number of threads for TensorFlow

tf.config.threading.set_inter_op_parallelism_threads(
    1)  # Set the number of threads for TensorFlow


# this disables GPU
# tf.config.set_visible_devices([], 'GPU')


def handler(signum, frame):
    """
    Signal handler to raise a TimeoutError when GA execution exceeds the allowed time.
    """

    location = f"File \"{frame.f_code.co_filename}\", line {frame.f_lineno}, in {frame.f_code.co_name}"
    tb = ''.join(traceback.format_stack(frame))
    error_message = f"GA execution timed out!\nLocation: {location}\nStack Trace:\n{tb}"
    raise TimeoutError(error_message)


# signal a time-out after 1 hour
signal.signal(signal.SIGALRM, handler)
# signal.alarm(3600)


logger = logging.getLogger(__name__)

# Initialize a global counter
fitness_counter = 0


@contextmanager
def managed_pool(max_workers):
    """
    Context manager to handle multiprocessing pool with proper initialization and cleanup.
    used with futures to handle timeouts

    Args:
        processes (int): Number of worker processes.

    Yields:
        mp.Pool: A multiprocessing pool.
    """
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
        # levele should be set by the main process
        # level=logging.INFO,  # Set to DEBUG to capture all debug messages
        format='%(asctime)s [PID %(process)d] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('debug.log', mode='a')
        ]
    )
    # Suppress DEBUG logs from specific third-party libraries if needed
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


class GeneticAlgorithm:
    """
    Genetic Algorithm class to manage the evolutionary process.

    Attributes:
        config (dict): Configuration parameters from config.yaml.
        X_train (np.ndarray): Training feature data.
        X_val (np.ndarray): Validation feature data.
        y_train (np.ndarray): Training labels.
        y_val (np.ndarray): Validation labels.
        toolbox (deap.Toolbox): DEAP toolbox with registered genetic operators.
        fitness_history (dict): Records fitness statistics per generation.
    """

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

                with open(filepath, 'a') as f:
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
        # Initialize population and Hall of Fame
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

        # stats to keep track of
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        master_logbook = tools.Logbook()
        # Define headers as per your stats
        master_logbook.header = ["gen", "avg", "std", "min", "max"]

        # debugging information
        logger.debug(mp.get_start_method())
        pid = os.getpid()
        timeout = self.config.ga.max_time_per_ind

        # Parallel environment for evaluating individuals
        with managed_pool(max_workers=self.config.ga.n_processes) as executor:
            logger.info(
                f"Starting Genetic Algorithm execution. Process ID: {pid}")

            # Initialize population
            pop = self.toolbox.population(n=self.config.ga.population_size)
            logger.debug(f"pop created.")

            # Evaluate the entire population
            futures = {executor.submit(eval_individual, ind, self.config,
                                       self.X_train, self.X_val,
                                       self.y_train, self.y_val): ind for ind in pop}
            logger.debug(f"futures created.")

            count = 0
            killed = 0

            # Initial population evaluation
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

        # generations after the parent gen
        for gen in range(1, self.config.ga.ngen + 1):
            logger.info(f"Generation {gen} started.")

            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            logger.debug(f"number of offsprings :{len(offspring)}.")
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            logger.debug(
                f"number of offsprings after map :{len(offspring)}.")

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
            # in case the offspring didn't change we don't need to re-evaluate
            invalid_ind = [
                ind for ind in offspring if not ind.fitness.valid]
            with managed_pool(max_workers=self.config.ga.n_processes) as executor:

                # Evaluate the entire population
                # Setup future parallel execution with per individual time-out
                futures = {executor.submit(eval_individual, ind, self.config,
                                           self.X_train, self.X_val,
                                           self.y_train, self.y_val): ind for ind in offspring}
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
                        f"Error during evaluation for individual {gen}: {e}")
                finally:
                    cn = 0
                    fn = 0
                    for future in futures:
                        fn += 1
                        if not future.done():
                            cn += 1
                            individual = futures[future]
                            f = (0.0,)
                            logger.debug(f"Evaluation {pid}. {count} {f}")
                            individual.fitness.values = f
                            future.cancel()
                            logger.warning(
                                f"Total evaluations timed out and were cancelled: fn:{fn} - {killed} cn: {cn}")

            # check that not all individuals have 0.0 fitness, which would mean that the evaluation timed out
            if all(ind.fitness.values == (0.0,) for ind in offspring):
                logger.error(
                    "All individuals have 0.0 fitness, indicating that the evaluation timed out.")
                raise RuntimeError(
                    "Evaluation timed out: All individuals have 0.0 fitness.")

            # Replace population with offspring
            pop[:] = offspring
            del offspring

            # Update Hall of Fame
            hof.update(pop)

            # Compile statistics about the new population
            try:
                record = stats.compile(pop)
            except ValueError as e:
                logger.error(f"Error during stats compilation: {e}")
                for i, ind in enumerate(pop):
                    logger.error(
                        f"Individual {i} shape: {np.shape(ind)}, value: {ind}")
                raise e

            # Compile and record statistics
            record = stats.compile(pop)
            master_logbook.record(gen=gen, **record)

            logger.debug(f"Generation {gen} Statistics: {record}")
            self.record_fitness(pop, gen)

            # local_vars = locals()
            # for var_name, var_value in local_vars.items():
            #     logger.debug(
            #         f" {count} -- Mysize of {var_name} : {sys.getsizeof(var_value)} bytes")
            # for var_name, var_value in vars(self).items():
            #     logger.debug(
            #         f"Mysize variable {var_name} : {sys.getsizeof(var_value)} bytes")

        # After recording statistics
        for entry in master_logbook:
            logger.debug(entry)
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


# @profile
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

    if (individual.fitness.valid):
        logger.debug(f"Individual already evaluated. Skipping.")
        return individual.fitness.values

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
            epochs=config.ga.epochs,
            batch_size=config.model.batch_size,
            validation_data=(X_val, y_val),
            verbose=verbose
        )

        logger.debug(f"{pid} Model training completed.")
        filepath = "results"
        os.makedirs(filepath, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4()

        full_filepath = os.path.join(
            filepath, f"ga_results_{timestamp}_{unique_id}.keras")

        # mod ger.info(f"{pid} Trained model saved to {full_filepath}.")

        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"{pid} Validation Accuracy: {val_accuracy}")
        tf.keras.backend.clear_session()
        del model
        gc.collect()

        return (val_accuracy,)

    except Exception as e:
        logger.error(
            f"{pid} Error during individual evaluation: {e}", exc_info=True)
        return (0.0,)
