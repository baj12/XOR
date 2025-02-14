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


# from memory_profiler import memory_usage, profile


import gc
import contextlib
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
from pympler import asizeof
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential  # Added for model manipulation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from model import build_model, get_optimizer
from utils import Config, get_total_size, save_model_and_history
from plotRawData import plot_train_test_with_decision_boundary


tf.config.threading.set_intra_op_parallelism_threads(
    1)  # Set the number of threads for TensorFlow

tf.config.threading.set_inter_op_parallelism_threads(
    1)  # Set the number of threads for TensorFlow


# this disables GPU
# tf.config.set_visible_devices([], 'GPU')


def log_gpu_usage():
    """Log GPU usage information."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # Get GPU memory info
                memory_info = tf.config.experimental.get_memory_info(gpu.name)
                logger.debug(f"GPU {gpu.name} memory usage: {memory_info}")
        except:
            logger.debug("Could not get GPU memory info")
    else:
        logger.warning("No GPU devices available")


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
    """Improved process pool management with proper resource cleanup"""
    ctx = mp.get_context('spawn')  # Use spawn context explicitly
    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        mp_context=ctx
    )

    try:
        yield executor
    finally:
        # Graceful shutdown sequence
        executor.shutdown(wait=True, cancel_futures=True)

        # Clean up any remaining processes
        for child in mp.active_children():
            try:
                child.terminate()
                child.join(timeout=1.0)
            except Exception as e:
                logger.debug(f"Error cleaning up child process: {e}")

        # Clear TensorFlow session
        tf.keras.backend.clear_session()


@contextlib.contextmanager
def timeout_context(seconds):
    def handler(signum, frame):
        raise TimeoutError("Evaluation timed out")
    # Register a function to raise a TimeoutError on the signal
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)


def evaluate_population(self, population, timeout=60):
    """
    Evaluate all individuals in the population with timeout protection.
    """
    pid = os.getpid()
    logger.debug(f"pop created.")

    try:
        with managed_pool(max_workers=self.config.ga.n_processes) as executor:
            # Setup future parallel execution with per individual time-out
            futures = {
                executor.submit(
                    eval_individual,
                    individual=ind,
                    config=self.config,
                    X_train=self.X_train,
                    X_val=self.X_val,
                    y_train=self.y_train,
                    y_val=self.y_val,
                    df=self.df
                ): ind for ind in population
            }
            logger.debug(f"futures created.")

            count = 0
            results = []

            try:
                for future in as_completed(futures, timeout=timeout+1):
                    count += 1
                    # Get the individual associated with this future
                    ind = futures[future]
                    try:
                        fitness = future.result(timeout=timeout)
                        ind.fitness.values = fitness  # Assign fitness to the individual
                        results.append((ind, fitness))
                        logger.debug(f"future. {count}")
                        logger.debug(f"future results {count}: {fitness}")
                    except TimeoutError:
                        logger.warning(
                            f"Evaluation timeout for individual {count}")
                        # Assign worst fitness on timeout
                        ind.fitness.values = (0.0,)
                    except Exception as e:
                        logger.error(
                            f"Error evaluating individual {count}: {e}")
                        # Assign worst fitness on error
                        ind.fitness.values = (0.0,)

            except concurrent.futures.TimeoutError:
                logger.warning(
                    "Global timeout reached during population evaluation")
                # Assign worst fitness to any remaining individuals
                for future, ind in futures.items():
                    if not future.done():
                        ind.fitness.values = (0.0,)

            return population

    except Exception as e:
        logger.error(f"Error during population evaluation: {e}")
        # Ensure all individuals have fitness values
        for ind in population:
            if not hasattr(ind, 'fitness') or not ind.fitness.valid:
                ind.fitness.values = (0.0,)
        return population


def init_worker():
    """Initialize worker process."""
    tf.keras.backend.clear_session()
    # Allow GPU but limit memory growth to prevent memory conflicts
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Optionally, limit GPU memory per process
            # tf.config.set_logical_device_configuration(
            #     gpus[0],
            #     [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]  # 1GB limit
            # )
        except RuntimeError as e:
            logger.warning(f"GPU configuration failed: {e}")


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

    def __init__(self, config: Config, X_train, X_val, y_train, y_val, df, paths):
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
        self.metrics = {
            'training_time': [],
            'memory_usage': [],
            'convergence': [],
            'parameters': [],
            'capacity_utilization': []
        }
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.df = df  # Store the full DataFrame
        self.model = build_model(config.model)
        self.total_weights = self.calculate_total_weights()
        self.setup_deap()
        self.pool = mp.Pool(processes=self.config.ga.n_processes)
        self.toolbox.register("map", self.pool.map)
        self.fitness_history = []  # To store fitness of all individuals per generation
        self.counters = defaultdict(int)
        self.pool = None
        self.processes = []
        self.paths = paths

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
        return total_weights

    def cleanup(self):
        """Cleanup method for GA resources"""
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)

        for process in self.processes:
            try:
                process.terminate()
                process.join(timeout=0.5)
            except Exception as e:
                logger.debug(f"Error cleaning up GA process: {e}")

        tf.keras.backend.clear_session()

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

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

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"fitness_history_{timestamp}.json"
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
                filepath = os.path.join(self.paths.results, filename)
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

    def train_best_individual(self, individual):
        model = build_model(self.config.model)

        # Set weights from individual
        weight_shapes = [w.shape for layer in model.layers
                         for w in layer.get_weights() if w.size > 0]
        weight_tuples = []
        idx = 0
        for shape in weight_shapes:
            size = np.prod(shape)
            weights = np.array(individual[idx:idx+size]).reshape(shape)
            weight_tuples.append(weights)
            idx += size
        model.set_weights(weight_tuples)

        # Prepare training data with noise features
        if self.config.experiment.noise_dimensions > 0:
            noise_cols = [
                f'noise_{i+1}' for i in range(self.config.experiment.noise_dimensions)]
            X_train_full = np.column_stack([
                self.X_train,
                self.df[noise_cols].values[: len(self.X_train)]
            ])
            X_val_full = np.column_stack([
                self.X_val,
                self.df[noise_cols].values[len(self.X_train): len(
                    self.X_train) + len(self.X_val)]
            ])
        else:
            X_train_full = self.X_train
            X_val_full = self.X_val

        # Train model with full feature set
        history = model.fit(
            X_train_full, self.y_train,
            epochs=self.config.ga.epochs,
            batch_size=self.config.model.batch_size,
            validation_data=(X_val_full, self.y_val),
            verbose=1
        )

        return model, history

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
                                       self.y_train, self.y_val, self.df): ind for ind in pop}
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
                                           self.y_train, self.y_val, self.df): ind for ind in offspring}
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
        # current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # fitness_filename = f'fitness_history_{current_date}.json'
        # fitness_filepath = os.path.join("results", fitness_filename)

        # Save fitness history to a JSON file in the results directory
        # with open(fitness_filepath, 'w') as f:
        #    json.dump(self.fitness_history, f)

        # Retrieve the best individual from Hall of Fame
        best_individual = hof[0] if hof else None
        logger.info(
            f"Best Individual: {best_individual}, Fitness: {best_individual.fitness.values if best_individual else 'N/A'}")

        # After evolution completes, train and save the best model
        # best_individual = tools.selBest(population, k=1)[0]
        best_model, best_history = self.train_best_individual(best_individual)

        # Save the best model and its training history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_model_and_history(best_model, best_history, self.paths, timestamp)
        # Generate and save the final classification plot
        plot_path = f"{self.paths.plots}/final_classification.png"

        try:
            plot_train_test_with_decision_boundary(
                model=best_model,
                X_train=self.X_train,
                X_test=self.X_val,
                y_train=self.y_train,
                y_test=self.y_val,
                df=self.df,  # Pass the full DataFrame
                config=self.config,  # Pass the configuration
                save_path=plot_path
            )
            logger.info(f"Final classification plot saved to {plot_path}")
        except Exception as e:
            logger.error(
                f"Failed to generate final classification plot: {str(e)}", exc_info=True)

        return best_individual, master_logbook


# @profile
def eval_individual(individual, config, X_train, X_val, y_train, y_val, df):
    """
    Evaluate a single individual.

    Parameters:
        individual: The individual to evaluate
        config: Configuration object
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        df: Full DataFrame containing all features
    """
    try:
        tf.keras.backend.clear_session()

        # Build model with correct input shape
        model = build_model(config.model)

        # Reshape individual to match model weights
        weight_shapes = [w.shape for layer in model.layers
                         for w in layer.get_weights() if w.size > 0]
        weight_tuples = []

        idx = 0
        for shape in weight_shapes:
            size = np.prod(shape)
            weights = np.array(individual[idx:idx+size]).reshape(shape)
            weight_tuples.append(weights)
            idx += size

        model.set_weights(weight_tuples)

        # Prepare data with noise features if specified
        if config.experiment.noise_dimensions > 0:
            noise_cols = [
                f'noise_{i+1}' for i in range(config.experiment.noise_dimensions)]
            noise_data = df[noise_cols].values[: len(X_train)]
            X_train_full = np.column_stack([X_train, noise_data])

            noise_data_val = df[noise_cols].values[len(
                X_train): len(X_train) + len(X_val)]
            X_val_full = np.column_stack([X_val, noise_data_val])

        else:
            X_train_full = X_train
            X_val_full = X_val

        # Train and evaluate
        history = model.fit(
            X_train_full, y_train,
            validation_data=(X_val_full, y_val),
            epochs=config.ga.epochs,
            batch_size=config.model.batch_size,
            verbose=0
        )

        val_accuracy = history.history['val_accuracy'][-1]
        return (val_accuracy,)

    except Exception as e:
        logger.error(
            f"Error during individual evaluation: {str(e)}", exc_info=True)
        return (0.0,)


def evaluate_individual(self, individual):
    """Evaluate individual with metric tracking."""
    start_time = time.time()
    memory_start = get_process_memory()

    fitness = super().evaluate_individual(individual)

    if self.config.metrics['tracking']['training']['track_time']:
        self.metrics['training_time'].append(time.time() - start_time)

    if self.config.metrics['tracking']['training']['track_memory']:
        self.metrics['memory_usage'].append(
            get_process_memory() - memory_start)

    return fitness
