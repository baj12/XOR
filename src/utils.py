import json
import logging
import multiprocessing
import os
import pickle
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

logger = logging.getLogger(__name__)


@contextmanager
def managed_multiprocessing():
    try:
        yield
    finally:
        # Clean up any remaining multiprocessing resources
        cleanup_processes()
        tf.keras.backend.clear_session()
        # Clean up any remaining multiprocessing resources
        for p in multiprocessing.active_children():
            p.terminate()
            p.join(timeout=1.0)


@dataclass
class GAConfig:
    population_size: int
    cxpb: float
    mutpb: float
    ngen: int
    n_processes: int
    max_time_per_ind: float
    epochs: int


@dataclass
class ModelConfig:
    hidden_layers: List[int]
    neurons_per_layer: int
    skip_connections: Optional[str]
    activation: str
    optimizer: str
    lr: float
    batch_size: int
    input_dim: int = 2  # default to 2 for basic XOR


@dataclass
class ExperimentConfig:
    id: str
    description: str
    noise_dimensions: int
    class_separation: float


@dataclass
class DataConfig:
    input_dim: int
    class_distribution: float
    dataset_size: int


@dataclass
class GAConfig:
    population_size: int
    cxpb: float
    mutpb: float
    ngen: int
    n_processes: int
    max_time_per_ind: float
    epochs: int


@dataclass
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    ga: GAConfig
    model: ModelConfig
    metrics: dict


def cleanup_processes():
    """Cleanup any remaining processes."""
    logger.debug("Initiating process cleanup.")

    import psutil
    current_process = psutil.Process()
    children = current_process.children(recursive=True)

    for child in children:
        try:
            logger.debug(f"Terminating process: {child.pid}")
            child.terminate()
        except psutil.NoSuchProcess:
            logger.debug(f"Process {child.pid} no longer exists")
            pass
        except Exception as e:
            logger.error(f"Error terminating process {child.pid}: {e}")

    # Wait for processes to terminate
    gone, alive = psutil.wait_procs(children, timeout=3)

    # Force kill any remaining processes
    for p in alive:
        try:
            logger.debug(f"Force killing process: {p.pid}")
            p.kill()
        except psutil.NoSuchProcess:
            pass
        except Exception as e:
            logger.error(f"Error killing process {p.pid}: {e}")

    # Clear any TensorFlow sessions
    try:
        tf.keras.backend.clear_session()
        logger.debug("TensorFlow session cleared")
    except Exception as e:
        logger.error(f"Error clearing TensorFlow session: {e}")

    logger.debug("Process cleanup completed.")


def configure_logging(log_level=None):
    import logging
    import os
    import sys

    if not log_level:
        log_level = os.getenv('LOG_LEVEL', 'DEBUG')

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {log_level}")
        sys.exit(1)
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {log_level}")
        sys.exit(1)

    # Clear existing handlers to prevent duplicates
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s [PID %(process)d] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('debug.log', mode='a')  # Adds FileHandler
        ]
    )
    logger = logging.getLogger(__name__)
    logger.debug("Logging configured successfully.")

    # Suppress DEBUG logs from specific third-party libraries
    libraries_to_suppress = ['matplotlib']
    for lib in libraries_to_suppress:
        logging.getLogger(lib).setLevel(logging.WARNING)

    return logger


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    experiment_config = ExperimentConfig(**config_dict['experiment'])
    data_config = DataConfig(**config_dict['data'])
    ga_config = GAConfig(**config_dict['ga'])
    model_config = ModelConfig(**config_dict['model'])

    return Config(
        experiment=experiment_config,
        data=data_config,
        ga=ga_config,
        model=model_config,
        metrics=config_dict.get('metrics', {})
    )


def plot_results(logbook):
    """
    Plot the results of the genetic algorithm.

    Parameters:
    - logbook (Logbook): Logbook containing statistics of the evolution.
    """
    generations = logbook.select("gen")
    avg_fitness = logbook.select("avg")
    max_fitness = logbook.select("max")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Plot Average and Max Fitness
    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_fitness, label='Average Fitness', color='blue')
    plt.plot(generations, max_fitness, label='Max Fitness', color='red')
    plt.title('Genetic Algorithm Progress Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend(loc='lower right')
    plt.grid(True)
    plot_filename = f"plots/fitness_over_generations_{timestamp}.png"
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to prevent blocking
    logger.debug(f"Plot saved to {plot_filename}")


def validate_file(filepath):
    """
    Validates that the provided filepath is a CSV file with the required structure:
    - Three columns: 'x', 'y', 'label'
    - 'x' and 'y' are floats
    - 'label' is an integer

    Parameters:
    - filepath (str): Path to the CSV file.

    Returns:
    - df (DataFrame): Validated pandas DataFrame.

    Raises:
    - ValueError: If any validation fails.
    """
    # Check if file exists
    if not os.path.isfile(filepath):
        raise ValueError(f"File '{filepath}' does not exist.")

    # Check file extension
    if not filepath.lower().endswith('.csv'):
        raise ValueError(f"File '{filepath}' is not a CSV file.")

    try:
        df = pd.read_csv(filepath)

        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("The CSV file is empty.")

        # Check number of columns
        expected_columns = ['x', 'y', 'label']
        if list(df.columns) != expected_columns:
            raise ValueError(
                f"CSV file must have exactly three columns: {expected_columns}. Found columns: {list(df.columns)}")

        # Validate data types
        if not pd.api.types.is_float_dtype(df['x']):
            raise ValueError("Column 'x' must contain float values.")
        if not pd.api.types.is_float_dtype(df['y']):
            raise ValueError("Column 'y' must contain float values.")
        if not pd.api.types.is_integer_dtype(df['label']):
            raise ValueError("Column 'label' must contain integer values.")
        # Print summary information
        logger.info("\nDataset Summary:")
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Total columns: {len(df.columns)}")
        logger.info("\nColumn info:")
        logger.info(df.dtypes)
        logger.debug("\nFirst 5 rows preview:")
        logger.debug(df.head())

        return df

    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")


def save_results(best_individual, logbook, filepath='results/ga_results.pkl'):
    """
    Saves the best individual and logbook to a file using pickle.

    Parameters:
    - best_individual: The best individual from the GA run.
    - logbook: The logbook containing GA run statistics.
    - filepath (str): Path where the results will be saved.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # unique_id = uuid.uuid4()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filepath = os.path.join(
        filepath, f'ga_results_best_{timestamp}.pkl')

    logger.debug(f"saving to {full_filepath}")
    with open(full_filepath, 'wb') as f:
        pickle.dump(
            {'best_individual': best_individual, 'logbook': logbook}, f)
    logger.info(f"Results saved to {full_filepath}")


def load_results(filepath='results/ga_results.pkl'):
    """
    Loads the best individual and logbook from a file.

    Parameters:
    - filepath (str): Path from where the results will be loaded.

    Returns:
    - data (dict): Dictionary containing 'best_individual' and 'logbook'.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No results found at {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    logger.debug(f"Results loaded from {filepath}")
    return data


def get_total_size(obj, seen=None):
    """Recursively finds the total size of an object including its contents."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_total_size(v, seen) for v in obj.values()])
        size += sum([get_total_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_total_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_total_size(i, seen) for i in obj])
    return size


def get_all_child_processes():
    try:
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)
        logger.debug(f"Found {len(children)} child process(es).")
        for child in children:
            logger.debug(f"Child Process PID={child.pid}, Name={child.name()}")
        return children
    except psutil.NoSuchProcess:
        logger.error("Current process does not exist.")
        return []
    except Exception as e:
        logger.error(
            f"An error occurred while retrieving child processes: {e}")
        return []


def terminate_child_processes(children):
    for child in children:
        try:
            logger.debug(
                f"Terminating child process PID={child.pid}, Name={child.name()}")
            child.terminate()
        except psutil.NoSuchProcess:
            logger.warning(f"Process PID={child.pid} does not exist.")
        except psutil.AccessDenied:
            logger.error(
                f"Access denied when trying to terminate PID={child.pid}.")
        except Exception as e:
            logger.error(f"Error terminating PID={child.pid}: {e}")


def kill_child_processes(children):
    for child in children:
        try:
            logger.debug(
                f"Killing child process PID={child.pid}, Name={child.name()}")
            child.kill()
        except psutil.NoSuchProcess:
            logger.warning(f"Process PID={child.pid} does not exist.")
        except psutil.AccessDenied:
            logger.error(f"Access denied when trying to kill PID={child.pid}.")
        except Exception as e:
            logger.error(f"Error killing PID={child.pid}: {e}")


def write_config_to_text(config: dict, output_path: str):
    """
    Writes configuration parameters to a plain text file.

    Args:
        config (dict): Configuration parameters.
        output_path (str): Path to the output text file.
    """
    try:
        with open(output_path, 'w') as file:
            file.write("Configuration Parameters:\n")
            file.write("==========================\n")
            _write_dict(config, file)
        logger.debug(f"Configuration parameters written to {output_path}.")
    except Exception as e:
        logger.error(f"Failed to write configuration to {output_path}: {e}")
        raise


def _write_dict(d: dict, file, indent: int = 0):
    """
    Recursively writes a dictionary to a file with indentation.

    Args:
        d (dict): The dictionary to write.
        file (file object): The file object to write to.
        indent (int): Current indentation level.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            file.write('    ' * indent + f"{key}:\n")
            _write_dict(value, file, indent + 1)
        else:
            file.write('    ' * indent + f"{key}: {value}\n")
