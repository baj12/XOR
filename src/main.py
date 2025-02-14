# main.py

import argparse
import logging
import multiprocessing as mp
import os
import signal  # Add this import
import sys
from datetime import datetime

import psutil
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split

from genetic_algorithm import GeneticAlgorithm, managed_pool
from model import build_and_train_model
from plotRawData import plot_train_test_with_decision_boundary
from utils import (Config, cleanup_processes, configure_logging,
                   get_all_child_processes, get_output_dirs,
                   kill_child_processes, load_config, load_results,
                   managed_multiprocessing, plot_results, save_results,
                   terminate_child_processes, validate_file,
                   write_config_to_text)

# Setup logger
logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Received shutdown signal. Cleaning up...")
    cleanup_processes()
    tf.keras.backend.clear_session()
    sys.exit(0)


# Register the signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# logging.basicConfig(
#     level=logging.DEBUG,  # Set to DEBUG to capture all levels of log messages
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     handlers=[
#         logging.StreamHandler()  # Logs will be printed to the console
#     ]
# )
# logger = logging.getLogger(__name__)
# # Suppress DEBUG messages from matplotlib.font_manager
# logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def signal_handler(signum, frame):
    logger.info("Received shutdown signal. Cleaning up...")
    cleanup_processes()
    tf.keras.backend.clear_session()
    sys.exit(0)


# Register the signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run XOR neural network training with genetic algorithm.')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--log',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the logging level'
    )
    return parser.parse_args()


def run_experiment(config: Config, data_file: str):
    """Run experiment with given configuration."""
    # Generate data with noise
    data = generate_xor_data(
        n_samples=config.data.dataset_size,
        noise_dim=config.experiment.noise_dimensions,
        class_separation=config.data.class_distribution
    )

    # Create experiment directory
    experiment_dir = f"results/experiment_{config.experiment.id}"
    os.makedirs(experiment_dir, exist_ok=True)

    # Save experiment metadata
    metadata = {
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'data_shape': data.shape
    }
    with open(f"{experiment_dir}/metadata.yaml", 'w') as f:
        yaml.dump(metadata, f)

    # Run genetic algorithm
    ga = GeneticAlgorithm(config, X_train, X_val, y_train, y_val)
    best_individual, logbook = ga.run()

    # Save results
    save_results(best_individual, logbook,
                 f"{experiment_dir}/ga_results.pkl")

    # Generate plots
    plot_results(logbook, f"{experiment_dir}/plots")


def main():
    args = parse_arguments()
    configure_logging(args.log)
    logger.debug("Starting main function.")
    config_name = os.path.basename(args.config).replace('.yaml', '')
    # Get standardized directory structure
    dirs = get_output_dirs(config_name)

    try:
        # Load configuration
        config = load_config(args.config)
        # Create config dictionary for writing to text file
        config_dict = {
            'experiment': vars(config.experiment),
            'data': vars(config.data),
            'ga': vars(config.ga),
            'model': vars(config.model),
            'metrics': config.metrics
        }
        # Get expected data filepath based on config filename
        config_name = os.path.basename(args.config).replace('.yaml', '')
        expected_data_file = f"data/raw/{config_name}_data.csv"

        # Check if data file exists and matches config
        if not os.path.exists(expected_data_file):
            logger.error(f"Data file not found: {expected_data_file}")
            logger.error("Please generate data first using data_generator.py")
            sys.exit(1)

        # Load and validate data
        try:
            df = validate_file(expected_data_file)
            logger.debug(
                f"Successfully validated the input file: '{expected_data_file}'")
        except ValueError as ve:
            logger.error(f"Validation Error: {ve}")
            sys.exit(1)

        # Write configuration parameters to a plain text file
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        # config_output_path = f"plots/config_parameters_{current_date}.txt"
        config_output_path = f"{dirs['configs']}/parameters_{current_date}.txt"
        os.makedirs(os.path.dirname(config_output_path), exist_ok=True)
        try:
            write_config_to_text(config_dict, config_output_path)
            logger.debug(
                f"Configuration parameters written to {config_output_path}.")
        except Exception as e:
            logger.error(f"Failed to write configuration to text file: {e}")
            sys.exit(1)

        # Split data
        X = df[['x', 'y']].values
        y = df['label'].values
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info("TensorFlow GPU support: %s", tf.test.is_built_with_cuda())
        logger.info("TensorFlow GPU available: %s",
                    tf.config.list_physical_devices('GPU'))
        # Run genetic algorithm
        try:
            ga = GeneticAlgorithm(config, X_train, X_test, Y_train, Y_test)
            best_individual, logbook = ga.run()
            logger.debug("Genetic Algorithm completed successfully.")
        except Exception as e:
            logger.error(f"Error during Genetic Algorithm execution: {e}")
            cleanup_processes()
            sys.exit(1)

        # Rest of your code...

    except Exception as e:
        logger.error(f"Error in main: {e}")
        cleanup_processes()
        sys.exit(1)
    finally:
        cleanup_processes()


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        main()
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
    finally:
        cleanup_processes()
