import json
import logging
import multiprocessing as mp
import os
import pickle
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
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
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    ga: GAConfig
    model: ModelConfig
    metrics: dict
    use_gpu: bool = False  # Add GPU toggle as a default field


class ExperimentPaths:
    def __init__(self, config_name: str):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f"experiments/{config_name}_{timestamp}"

        # Create attribute for each path
        self.base_dir = base_dir
        self.models = f"{base_dir}/models"
        self.config = f"{base_dir}/config"
        self.results = f"{base_dir}/results"
        self.plots = f"{base_dir}/plots"
        self.logs = f"{base_dir}/logs"

        # Create all directories
        for path in [self.models, self.config, self.results, self.plots, self.logs]:
            os.makedirs(path, exist_ok=True)
            logger.debug(f"Created directory: {path}")


def get_output_dirs(config_name):
    """
    Creates and returns standardized output directory structure.

    Args:
        config_name (str): Name of the configuration file (without extension)

    Returns:
        dict: Dictionary containing paths for different outputs
    """
    base_dir = f"experiments/{config_name}"
    dirs = {
        'models': f"{base_dir}/models",
        'plots': f"{base_dir}/plots",
        'results': f"{base_dir}/results",
        'logs': f"{base_dir}/logs",
        'configs': f"{base_dir}/configs"
    }

    # Create all directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def cleanup_processes():
    """Enhanced process cleanup with resource tracking"""
    logger.debug("Initiating process cleanup.")

    # Clean up TensorFlow
    tf.keras.backend.clear_session()

    # Clean up multiprocessing resources
    active_children = mp.active_children()
    for child in active_children:
        try:
            logger.debug(f"Terminating process: {child.pid}")
            child.terminate()
            child.join(timeout=0.5)

            # Force kill if still alive
            if child.is_alive():
                logger.debug(f"Force killing process: {child.pid}")
                child.kill()
                child.join(timeout=0.5)
        except Exception as e:
            logger.debug(f"Error during process cleanup: {e}")

    # Reset multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass


def configure_logging(log_level=None, log_path=None):
    """
    Configure logging to output to both console and file.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_path: Path to log file. If None, only console logging is enabled
    """
    import logging
    import os
    import sys

    if not log_level:
        log_level = os.getenv('LOG_LEVEL', 'DEBUG')

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {log_level}")
        sys.exit(1)

    # Clear existing handlers to prevent duplicates
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

   # Update the formatter to include filename and line number
    log_formatter = logging.Formatter(
        '%(asctime)s [PID %(process)d] %(levelname)s: %(filename)s:%(lineno)d - %(message)s'
    )

    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # Configure file handler if log_path is provided
    if log_path:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    # Set the logging level
    root_logger.setLevel(numeric_level)

    # Suppress DEBUG logs from specific third-party libraries
    libraries_to_suppress = ['matplotlib', 'PIL', 'tensorflow']
    for lib in libraries_to_suppress:
        logging.getLogger(lib).setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.debug("Logging configured successfully.")
    if log_path:
        logger.debug(f"Logging to file: {log_path}")

    return logger


def save_results(self, best_individual, logbook, dirs):
    """
    Save GA results using standardized directory structure
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"{dirs['results']}/ga_results_{timestamp}.pkl"

    with open(results_path, 'wb') as f:
        pickle.dump({
            'best_individual': best_individual,
            'logbook': logbook,
            'config_name': self.config_name,
            'timestamp': timestamp
        }, f)


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
    # plot_filename = f"plots/fitness_over_generations_{timestamp}.png"
    plt.savefig(f"{paths.plots}/fitness_over_generations.png")
    plt.close()  # Close the plot to prevent blocking
    logger.debug(f"Plot saved to {plot_filename}")


def validate_file(filepath, config: Config):
    """
    Validates and preprocesses the CSV file based on model configuration.

    Parameters:
        filepath (str): Path to the CSV file
        config (Config): Configuration object containing model specifications
    """
    if not os.path.isfile(filepath):
        raise ValueError(f"File '{filepath}' does not exist.")

    if not filepath.lower().endswith('.csv'):
        raise ValueError(f"File '{filepath}' is not a CSV file.")

    try:
        df = pd.read_csv(filepath)

        if df.empty:
            raise ValueError("The CSV file is empty.")

        # Check for required base columns
        required_columns = ['x', 'y', 'label']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(
                f"CSV file must contain the base columns: {required_columns}")

        # Calculate expected number of input features from config
        expected_inputs = config.model.input_dim

        # If noise dimensions are specified in experiment config
        noise_dims = config.experiment.noise_dimensions

        # Select appropriate columns based on config
        if noise_dims > 0:
            # Include noise columns
            feature_columns = ['x', 'y'] + \
                [f'noise_{i+1}' for i in range(noise_dims)]
            if len(feature_columns) != expected_inputs:
                raise ValueError(
                    f"Model expects {expected_inputs} input features but data has "
                    f"{len(feature_columns)} features ({feature_columns})"
                )
        else:
            # Only x, y columns
            feature_columns = ['x', 'y']
            if expected_inputs != 2:
                raise ValueError(
                    f"Model expects {expected_inputs} input features but data has "
                    f"only x, y columns"
                )

        # Validate all expected columns exist
        missing_cols = [
            col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Select features and label columns
        df = df[feature_columns + ['label']]

        # Validate data types
        for col in feature_columns:
            if not pd.api.types.is_float_dtype(df[col]):
                raise ValueError(f"Column '{col}' must contain float values.")

        if not pd.api.types.is_integer_dtype(df['label']):
            raise ValueError("Column 'label' must contain integer values.")

        # Print summary information
        logger.info("\nDataset Summary:")
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Input features: {feature_columns}")
        logger.info(f"Model input dimension: {expected_inputs}")
        logger.info("\nColumn info:")
        logger.info(df.dtypes)
        logger.debug("\nFirst 5 rows preview:")
        logger.debug(df.head())

        return df

    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")

# def save_results(best_individual, logbook, filepath='results/ga_results.pkl'):
    """
    Saves the best individual and logbook to a file using pickle.

    Parameters:
    - best_individual: The best individual from the GA run.
    - logbook: The logbook containing GA run statistics.
    - filepath (str): Path where the results will be saved.
    """
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
    """


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

# utils.py


def save_model_and_history(model, history, paths, timestamp=None):
    """
    Save trained model and plot training history.

    Args:
        model: Trained Keras model
        history: Training history
        paths: ExperimentPaths object
        timestamp: Optional timestamp for file naming
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_path = f"{paths.models}/model_{timestamp}.keras"
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Plot and save training history
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    accuracy_plot_path = f"{paths.plots}/accuracy_{timestamp}.png"
    plt.savefig(accuracy_plot_path)
    plt.close()

    plt.figure()
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = f"{paths.plots}/loss_{timestamp}.png"
    plt.savefig(loss_plot_path)
    plt.close()


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
